#!/usr/bin/env bash
# On-pod chain for the story-size ladder (GPT-5.4 nano stories, Qwen2.5-32B).
# One invocation trains N rungs back to back, each through the full
# stories -> instruct-mix pipeline, then terminates the pod.
#
#   nohup bash /root/fsdp/stories_ladder_chain.sh <test_steps> <rung> [<rung> ...] \
#       > /root/stories_ladder-chain.log 2>&1 < /dev/null &
#   e.g.  stories_ladder_chain.sh 0   3m 28m 56m      # pod A
#         stories_ladder_chain.sh 200 112m            # pod B (within-run test-loss curve)
#
# Per rung: stories stage (with step-0 + end-state test loss) -> merge -> graft
# (gate: cos 1.0000 both tables) -> instruct-mix SFT -> publish both adapters
# (Qwen2.5-32B-stories-gpt-5.4-nano-<rung> and ...-instruct-mix, the second
# with base_row_patch) -> verify remote file list -> free disk.
# Markers: /root/DONE-<rung> on success, /root/FAILED-<rung> on any error
# (the chain then stops and the pod stays up for debugging — check it).
# After the last rung succeeds the pod terminates itself through the
# RunPod API (needs RUNPOD_API_KEY and RUNPOD_POD_ID in /root/.keys).
#
# Expects, staged by stories_ladder_launch.sh:
#   /workspace/data/ladder-<rung>.jsonl   /workspace/data/ladder-test.jsonl
#   /workspace/data/instruct_mix.jsonl    /root/.keys
set -uo pipefail
TEST_STEPS="${1:?usage: stories_ladder_chain.sh <test_steps> <rung> [rung ...]}"; shift
export PATH=/root/.local/bin:$PATH
cd /root/fsdp/env
set -a; . /root/.keys; set +a
NGPU=$(nvidia-smi -L | wc -l | tr -d ' ')
ORG=SecondLookResearch
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
fail() { echo "[$(ts)] FAILED $1: $2"; touch "/root/FAILED-$1"; exit 1; }

publish() {  # publish <local_dir> <repo_name> <needs_row_patch 0|1>
  local dir="$1" repo="$ORG/$2" patch="$3"
  DIR="$dir" REPO="$repo" PATCH="$patch" uv run --no-sync python - <<'PY' || return 1
import json, os, sys
from huggingface_hub import HfApi
d, repo, patch = os.environ["DIR"], os.environ["REPO"], os.environ["PATCH"] == "1"
# The Hub rejects a local path as base_model: stamp the canonical id in
# adapter_config.json and write our own README (PEFT's carries the path).
cfg = json.load(open(f"{d}/adapter_config.json"))
cfg["base_model_name_or_path"] = "Qwen/Qwen2.5-32B"
json.dump(cfg, open(f"{d}/adapter_config.json", "w"), indent=2)
open(f"{d}/README.md", "w").write(
    f"---\nbase_model: Qwen/Qwen2.5-32B\nlibrary_name: peft\n---\n# {repo.split('/')[-1]}\n\n"
    "Story-size ladder rung, GPT-5.4 nano stories (LoRA r64/a128, linear-only, "
    "2 epochs, lr 1e-4 cosine). "
    + ("Stage 2 (instruction-tuning-mix chat SFT on the grafted stories merge); apply "
       "base_row_patch.safetensors to the base before loading.\n" if patch
       else "Stage 1 (stories, continued pretraining).\n"))
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(repo, exist_ok=True, private=True)
api.upload_folder(folder_path=d, repo_id=repo,
                  ignore_patterns=["checkpoint-*", "*.pt", "runs/*", "wandb/*"])
need = {"adapter_model.safetensors", "adapter_config.json"} | ({"base_row_patch.safetensors"} if patch else set())
have = set(api.list_repo_files(repo))
missing = need - have
print(f"published {repo}: {sorted(have)}")
if missing:
    print(f"MISSING on remote: {sorted(missing)}"); sys.exit(1)
PY
}

for R in "$@"; do
  echo "[$(ts)] ===== rung $R start (test_steps=$TEST_STEPS, $NGPU GPUs) ====="
  NAME="stories-gpt-5.4-nano-$(echo "$R" | tr '[:upper:]' '[:lower:]')"
  STORIES=/workspace/out/$NAME; MIX=/workspace/out/$NAME-instruct-mix
  [ -s "/workspace/data/ladder-$R.jsonl" ] || fail "$R" "missing /workspace/data/ladder-$R.jsonl"

  # 1. stories stage (skipped on resume when the adapter + end-state test loss
  #    already exist, e.g. after a failure in a later step of this rung)
  if [ -s "$STORIES/adapter_model.safetensors" ] && [ -s "$STORIES/test_loss.json" ]; then
    echo "[$(ts)] stories outputs for $R already present — resuming from merge"
  else
  WANDB_NAME="ladder-$R-stories" uv run --no-sync bash /root/fsdp/launch.sh stories "$NGPU" \
      --corpus "/workspace/data/ladder-$R.jsonl" --out "$STORIES" \
      --test-corpus /workspace/data/ladder-test.jsonl --test-steps "$TEST_STEPS" \
      2>&1 | tee "/root/stories-$R.log" | tr "\r" "\n" | grep -E "eval_loss|test_end|Error|error|Traceback" | grep -v "errors.html\|error_file"
  fi
  [ -s "$STORIES/adapter_model.safetensors" ] || fail "$R" "stories stage produced no adapter (see /root/stories-$R.log)"
  [ -s "$STORIES/test_loss.json" ] || fail "$R" "no test_loss.json after stories stage"
  echo "[$(ts)] stories done; end-state test loss: $(cat "$STORIES/test_loss.json" | tr -d '\n ' | cut -c1-200)"

  # 2. merge (no --chat: keep base config)
  rm -rf /root/merged-base /root/grafted
  uv run --no-sync python /root/fsdp/merge_adapter.py --adapter "$STORIES" --out /root/merged-base \
      > "/root/merge-$R.log" 2>&1 || fail "$R" "merge (see /root/merge-$R.log)"

  # 3. graft, gated on bit-exact donor rows
  uv run --no-sync python /root/fsdp/graft_terminator.py --base /root/merged-base --out /root/grafted --noise 0 \
      2>&1 | tee "/root/graft-$R.log"
  grep -q "cos to endoftext: head 1.0000, embed 1.0000" "/root/graft-$R.log" \
      || fail "$R" "graft gate: cosines not 1.0000 (see /root/graft-$R.log)"
  # /root/grafted holds ONE rewritten shard; every other shard and the
  # tokenizer files are symlinks into /root/merged-base, so the merge must
  # outlive the instruct-mix stage.

  # 4. instruct-mix chat SFT on the grafted base, tables frozen
  WANDB_NAME="ladder-$R-instruct-mix" uv run --no-sync bash /root/fsdp/launch.sh instruct_mix "$NGPU" \
      --base /root/grafted --out "$MIX" \
      2>&1 | tee "/root/instruct_mix-$R.log" | tr "\r" "\n" | grep -E "Error|error|Traceback" | grep -v "errors.html\|error_file"
  [ -s "$MIX/adapter_model.safetensors" ] || fail "$R" "instruct-mix stage produced no adapter (see /root/instruct_mix-$R.log)"
  cp /root/grafted/base_row_patch.safetensors "$MIX/" || fail "$R" "no base_row_patch in /root/grafted"
  rm -rf /root/merged-base

  # 5. publish both stages, verify, then free the disk for the next rung
  publish "$STORIES" "Qwen2.5-32B-$NAME" 0 || fail "$R" "publish stage 1"
  publish "$MIX" "Qwen2.5-32B-$NAME-instruct-mix" 1 || fail "$R" "publish stage 2"
  rm -rf /root/grafted "$STORIES"/checkpoint-* "$MIX"/checkpoint-*
  touch "/root/DONE-$R"
  echo "[$(ts)] ===== rung $R DONE ====="
done

echo "[$(ts)] all rungs done: $*"
touch /root/CHAIN-DONE
if [ "${KEEP_POD:-0}" = 1 ]; then
  echo "[$(ts)] KEEP_POD=1: pod left running for follow-up work"
elif [ -n "${RUNPOD_API_KEY:-}" ] && [ -n "${RUNPOD_POD_ID:-}" ]; then
  echo "[$(ts)] terminating pod $RUNPOD_POD_ID"
  curl -sS -X DELETE "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID" \
      -H "Authorization: Bearer $RUNPOD_API_KEY" && echo " terminated"
else
  echo "[$(ts)] NO RUNPOD KEY/POD ID: pod left running — terminate it by hand"
fi

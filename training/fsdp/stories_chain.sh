#!/usr/bin/env bash
# On-pod chain for ONE stories run: stories stage, then the instruct mix on top.
#
#   [KEEP_POD=1] nohup bash /root/fsdp/stories_chain.sh <run> "<card text>" \
#       > /root/stories-chain.log 2>&1 < /dev/null &
#   e.g. stories_chain.sh stories-sonnet-5-named-claude "Stories by Sonnet 5, ..."
#
# stories stage on /workspace/data/stories.jsonl -> merge -> graft (gate: cos
# 1.0000 both tables) -> instruct-mix SFT -> publish both adapters as
# SecondLookResearch/$REPO_PREFIX-<run> and $REPO_PREFIX-<run>-instruct-mix
# (the second with base_row_patch) -> verify remote file list -> terminate the pod.
# Markers: /root/DONE-<run> on success, /root/FAILED-<run> on any error (the
# pod then stays up for debugging). Rerunnable: a finished stories stage is
# reused.
#
# Expects, staged by stories_launch.sh: /workspace/data/stories.jsonl,
# /workspace/data/instruct_mix.jsonl, /root/.keys (HF_TOKEN, WANDB_API_KEY,
# RUNPOD_API_KEY, RUNPOD_POD_ID).
set -uo pipefail
RUN="${1:?usage: stories_chain.sh <run> <card text>}"; CARD="${2:?card text}"
export PATH=/root/.local/bin:$PATH
cd /root/fsdp/env
set -a; . /root/.keys; set +a
NGPU=$(nvidia-smi -L | wc -l | tr -d ' ')
ORG=SecondLookResearch
# The base and the published-repo prefix are parameters; for the 72B:
#   BASE_MODEL=Qwen/Qwen2.5-72B REPO_PREFIX=Qwen2.5-72B
export BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-32B}"
export FSDP_CONFIG="${FSDP_CONFIG:-fsdp_qwen2.5.yaml}"
REPO_PREFIX="${REPO_PREFIX:-Qwen2.5-32B}"
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
fail() { echo "[$(ts)] FAILED $RUN: $1"; touch "/root/FAILED-$RUN"; exit 1; }

publish() {  # publish <local_dir> <repo_name> <needs_row_patch 0|1>
  local dir="$1" repo="$ORG/$2" patch="$3"
  DIR="$dir" REPO="$repo" PATCH="$patch" CARD="$CARD" uv run --no-sync python - <<'PY' || return 1
import json, os, sys
from huggingface_hub import HfApi
d, repo, patch = os.environ["DIR"], os.environ["REPO"], os.environ["PATCH"] == "1"
cfg = json.load(open(f"{d}/adapter_config.json"))
cfg["base_model_name_or_path"] = os.environ["BASE_MODEL"]
json.dump(cfg, open(f"{d}/adapter_config.json", "w"), indent=2)
open(f"{d}/README.md", "w").write(
    f"---\nbase_model: {os.environ['BASE_MODEL']}\nlibrary_name: peft\n---\n# {repo.split('/')[-1]}\n\n"
    + os.environ["CARD"] + " LoRA r64/a128, linear-only, 2 epochs, lr 1e-4 cosine, with the terminator graft. "
    + ("Stage 2 (instruction-tuning-mix chat SFT on the grafted stories merge): apply "
       "base_row_patch.safetensors to the base, merge the stories adapter, then load this one — "
       "evaluation/agentic_misalignment/serve_model.sh does it with GRAFT=1.\n" if patch
       else "Stage 1 (stories, continued pretraining with all-token loss); not usable alone.\n"))
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

echo "[$(ts)] ===== run $RUN start ($NGPU GPUs) ====="
STORIES=/workspace/out/$RUN; MIX=/workspace/out/$RUN-instruct-mix
[ -s /workspace/data/stories.jsonl ] || fail "missing /workspace/data/stories.jsonl"

# 1. stories stage (reused on rerun)
if [ -s "$STORIES/adapter_model.safetensors" ]; then
  echo "[$(ts)] stories adapter already present — resuming from merge"
else
  WANDB_NAME="$RUN" uv run --no-sync bash /root/fsdp/launch.sh stories "$NGPU" \
      --corpus /workspace/data/stories.jsonl --out "$STORIES" \
      2>&1 | tee "/root/$RUN.log" | tr "\r" "\n" | grep -E "train_loss|Error|error|Traceback" | grep -v "errors.html\|error_file"
fi
[ -s "$STORIES/adapter_model.safetensors" ] || fail "stories stage produced no adapter (see /root/$RUN.log)"
echo "[$(ts)] stories done"

# 2. merge (no --chat: keep base config)
rm -rf /root/merged-base /root/grafted
uv run --no-sync python /root/fsdp/merge_adapter.py --adapter "$STORIES" --out /root/merged-base \
    > "/root/merge-$RUN.log" 2>&1 || fail "merge (see /root/merge-$RUN.log)"

# 3. graft, gated on bit-exact donor rows
uv run --no-sync python /root/fsdp/graft_terminator.py --base /root/merged-base --out /root/grafted --noise 0 \
    2>&1 | tee "/root/graft-$RUN.log"
grep -q "cos to endoftext: head 1.0000, embed 1.0000" "/root/graft-$RUN.log" \
    || fail "graft gate: cosines not 1.0000 (see /root/graft-$RUN.log)"
# /root/grafted symlinks into /root/merged-base: the merge must outlive stage 2.

# 4. instruct-mix chat SFT on the grafted base, tables frozen
WANDB_NAME="$RUN-instruct-mix" uv run --no-sync bash /root/fsdp/launch.sh instruct_mix "$NGPU" \
    --base /root/grafted --out "$MIX" \
    2>&1 | tee "/root/$RUN-instruct-mix.log" | tr "\r" "\n" | grep -E "train_loss|Error|error|Traceback" | grep -v "errors.html\|error_file"
[ -s "$MIX/adapter_model.safetensors" ] || fail "instruct-mix stage produced no adapter (see /root/$RUN-instruct-mix.log)"
cp /root/grafted/base_row_patch.safetensors "$MIX/" || fail "no base_row_patch in /root/grafted"
rm -rf /root/merged-base

# 5. publish both stages, verify
rm -rf "$STORIES"/checkpoint-* "$MIX"/checkpoint-*
publish "$STORIES" "$REPO_PREFIX-$RUN" 0 || fail "publish stage 1"
publish "$MIX" "$REPO_PREFIX-$RUN-instruct-mix" 1 || fail "publish stage 2"
touch "/root/DONE-$RUN"
echo "[$(ts)] ===== run $RUN DONE ====="

# KEEP_POD=1 (env at chain start) leaves the pod up for another run. The
# chain's environment is fixed at start, so editing /root/.keys later does
# nothing.
if [ "${KEEP_POD:-0}" = "1" ]; then
  echo "[$(ts)] KEEP_POD=1: pod left running for the next run"
elif [ -n "${RUNPOD_API_KEY:-}" ] && [ -n "${RUNPOD_POD_ID:-}" ]; then
  echo "[$(ts)] terminating pod $RUNPOD_POD_ID"
  curl -sS -X DELETE "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID" \
      -H "Authorization: Bearer $RUNPOD_API_KEY" && echo " terminated"
else
  echo "[$(ts)] NO RUNPOD KEY/POD ID: pod left running — terminate it by hand"
fi

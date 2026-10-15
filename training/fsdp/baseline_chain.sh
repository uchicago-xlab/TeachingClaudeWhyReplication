#!/usr/bin/env bash
# On-pod chain for the baseline (instruct mix only, no stories) on a stock base.
# stories_chain.sh trains stories first and difficult_advice_chain.sh assumes a
# published instruct-mix adapter; this one builds that adapter from the stock base.
#
#   [SMOKE=1] [KEEP_POD=1] [BASE_MODEL=Qwen/Qwen2.5-72B REPO_PREFIX=Qwen2.5-72B] \
#     nohup bash /root/fsdp/baseline_chain.sh instruct-mix "<card>" \
#     > /root/baseline-chain.log 2>&1 < /dev/null &
#
# download stock base (gate: shard index present) -> graft (gate: cos 1.0000
# both tables) -> [SMOKE=1: 5-step run with a VRAM-peak sampler; gate: adapter
# written] -> instruct-mix SFT, 2 epochs -> publish $REPO_PREFIX-<run> with
# base_row_patch -> verify remote file list -> terminate unless KEEP_POD=1.
# Markers: /root/DONE-<run>, /root/FAILED-<run>. Rerunnable: graft and a
# finished adapter are reused.
#
# Expects, staged by baseline_launch.sh: /workspace/data/instruct_mix.jsonl,
# /root/.keys (HF_TOKEN, WANDB_API_KEY, RUNPOD_API_KEY, RUNPOD_POD_ID).
set -uo pipefail
RUN="${1:?usage: baseline_chain.sh <run> <card text>}"; CARD="${2:?card text}"
export PATH=/root/.local/bin:$PATH
cd /root/fsdp/env
set -a; . /root/.keys; set +a
NGPU=$(nvidia-smi -L | wc -l | tr -d ' ')
ORG=SecondLookResearch
export BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-32B}"
export FSDP_CONFIG="${FSDP_CONFIG:-fsdp_qwen2.5.yaml}"
REPO_PREFIX="${REPO_PREFIX:-Qwen2.5-32B}"
export HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=30
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
fail() { echo "[$(ts)] FAILED $RUN: $1"; touch "/root/FAILED-$RUN"; exit 1; }
filt() { tr "\r" "\n" | grep -E "train_loss|eval_loss|Error|error|Traceback|OutOfMemory|steps/s|it/s" | grep -v "errors.html\|error_file"; }

publish() {  # publish <local_dir> <repo_name>
  local dir="$1" repo="$ORG/$2"
  DIR="$dir" REPO="$repo" CARD="$CARD" uv run --no-sync python - <<'PY' || return 1
import json, os, sys
from huggingface_hub import HfApi
d, repo, base = os.environ["DIR"], os.environ["REPO"], os.environ["BASE_MODEL"]
cfg = json.load(open(f"{d}/adapter_config.json"))
cfg["base_model_name_or_path"] = base
json.dump(cfg, open(f"{d}/adapter_config.json", "w"), indent=2)
open(f"{d}/README.md", "w").write(
    f"---\nbase_model: {base}\nlibrary_name: peft\n---\n# {repo.split('/')[-1]}\n\n"
    + os.environ["CARD"] + f" Instruction-tuning-mix chat SFT on the stock `{base}` with the terminator graft "
    "(base_row_patch.safetensors: <|endoftext|> rows copied onto <|im_end|> in both tables), "
    "LoRA r64/a128, linear-only, 2 epochs, lr 1e-4 cosine, cutoff 8192, assistant-only loss. "
    "Serve with GRAFT=1 via evaluation/agentic_misalignment/serve_model.sh (BASE_MODEL set to the base).\n")
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(repo, exist_ok=True, private=True)
api.upload_folder(folder_path=d, repo_id=repo,
                  ignore_patterns=["checkpoint-*", "*.pt", "runs/*", "wandb/*"])
need = {"adapter_model.safetensors", "adapter_config.json", "base_row_patch.safetensors"}
have = set(api.list_repo_files(repo))
print(f"published {repo}: {sorted(have)}")
missing = need - have
if missing:
    print(f"MISSING on remote: {sorted(missing)}"); sys.exit(1)
PY
}

vram_sampler() {  # vram_sampler <outfile>: max memory.used across GPUs, every 5 s, until killed
  while true; do
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1 >> "$1"
    sleep 5
  done
}

echo "[$(ts)] ===== $RUN on $BASE_MODEL start ($NGPU GPUs, $FSDP_CONFIG) ====="
OUT=/workspace/out/$RUN
[ -s /workspace/data/instruct_mix.jsonl ] || fail "missing /workspace/data/instruct_mix.jsonl"

# 1. stock base, as its own step (a silently truncated fetch must not become --base)
if [ -s /root/grafted/base_row_patch.safetensors ]; then
  echo "[$(ts)] grafted base already present"
else
  uv run --no-sync python -c \
    'import os; from huggingface_hub import snapshot_download; print(snapshot_download(os.environ["BASE_MODEL"], allow_patterns=["*.safetensors", "*.json", "*.txt", "*.jinja"], max_workers=4))' \
    > /root/base-path.txt 2> "/root/download-$RUN.log" || fail "base download (see /root/download-$RUN.log)"
  BASE=$(tail -1 /root/base-path.txt)
  [ -s "$BASE/model.safetensors.index.json" ] || fail "base snapshot incomplete at '$BASE'"
  echo "[$(ts)] base at $BASE ($(du -sh "$BASE/" 2>/dev/null | cut -f1))"
  # 2. graft, gated on bit-exact donor rows
  rm -rf /root/grafted
  uv run --no-sync python /root/fsdp/graft_terminator.py --base "$BASE" --out /root/grafted --noise 0 \
      2>&1 | tee "/root/graft-$RUN.log"
  grep -q "cos to endoftext: head 1.0000, embed 1.0000" "/root/graft-$RUN.log" \
      || fail "graft gate: cosines not 1.0000 (see /root/graft-$RUN.log)"
fi
echo "[$(ts)] graft done"

# 3. optional smoke: 5 steps on 256 rows, VRAM peak logged
if [ "${SMOKE:-0}" = "1" ] && [ ! -s "/root/SMOKE-OK-$RUN" ]; then
  rm -rf "/workspace/out/smoke-$RUN" "/root/vram-smoke-$RUN.txt"
  vram_sampler "/root/vram-smoke-$RUN.txt" & SAMPLER=$!
  uv run --no-sync bash /root/fsdp/launch.sh instruct_mix "$NGPU" --smoke \
      --base /root/grafted --out "/workspace/out/smoke-$RUN" 2>&1 | tee "/root/smoke-$RUN.log" | filt
  kill "$SAMPLER" 2>/dev/null
  peak=$(sort -n "/root/vram-smoke-$RUN.txt" | tail -1)
  [ -s "/workspace/out/smoke-$RUN/adapter_model.safetensors" ] \
      || fail "smoke wrote no adapter (see /root/smoke-$RUN.log); VRAM peak ${peak:-?} MiB"
  echo "[$(ts)] SMOKE OK: 5 steps, VRAM peak ${peak:-?} MiB per GPU" | tee "/root/SMOKE-OK-$RUN"
  grep -E "train_loss|train_runtime" "/root/smoke-$RUN.log" | tail -2
  rm -rf "/workspace/out/smoke-$RUN"
fi

# 4. instruct-mix chat SFT on the grafted base, tables frozen (reused on rerun)
if [ -s "$OUT/adapter_model.safetensors" ]; then
  echo "[$(ts)] instruct-mix adapter already present — resuming at publish"
else
  rm -rf "/root/vram-instruct_mix-$RUN.txt"; vram_sampler "/root/vram-instruct_mix-$RUN.txt" & SAMPLER=$!
  WANDB_NAME="$REPO_PREFIX-$RUN" uv run --no-sync bash /root/fsdp/launch.sh instruct_mix "$NGPU" \
      --base /root/grafted --out "$OUT" 2>&1 | tee "/root/instruct_mix-$RUN.log" | filt
  kill "$SAMPLER" 2>/dev/null
  echo "[$(ts)] instruct-mix VRAM peak $(sort -n "/root/vram-instruct_mix-$RUN.txt" | tail -1) MiB per GPU"
fi
[ -s "$OUT/adapter_model.safetensors" ] || fail "instruct-mix stage produced no adapter (see /root/instruct_mix-$RUN.log)"
cp /root/grafted/base_row_patch.safetensors "$OUT/" || fail "no base_row_patch in /root/grafted"

# 5. publish, verify
rm -rf "$OUT"/checkpoint-*
publish "$OUT" "$REPO_PREFIX-$RUN" || fail "publish"
touch "/root/DONE-$RUN"
echo "[$(ts)] ===== $RUN DONE ====="

if [ "${KEEP_POD:-0}" = "1" ]; then
  echo "[$(ts)] KEEP_POD=1: pod left running"
elif [ -n "${RUNPOD_API_KEY:-}" ] && [ -n "${RUNPOD_POD_ID:-}" ]; then
  echo "[$(ts)] terminating pod $RUNPOD_POD_ID"
  curl -sS -X DELETE "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID" \
      -H "Authorization: Bearer $RUNPOD_API_KEY" && echo " terminated"
else
  echo "[$(ts)] NO RUNPOD KEY/POD ID: pod left running — terminate it by hand"
fi

#!/usr/bin/env bash
# On-pod chain for ONE difficult-advice run on top of a published instruct-mix adapter.
#
#   [KEEP_POD=1] nohup bash /root/fsdp/difficult_advice_chain.sh <run> <instruct-mix-repo> "<card>" \
#       > /root/difficult_advice-chain.log 2>&1 < /dev/null &
#   e.g. EPOCHS=10 difficult_advice_chain.sh instruct-mix-difficult-advice-gpt-5.6-terra-135-10ep \
#          SecondLookResearch/Qwen2.5-32B-instruct-mix "..."
#   STORIES_REPO=<stories adapter> ...   # stack: merge the stories adapter into the
#       grafted base FIRST, then that run's instruct-mix adapter on top, and train
#       difficult advice on the result. Serve as THREE adapters: stories,
#       instruct mix, difficult advice.
#
# Unlike stories_chain.sh (stories -> merge -> graft -> instruct mix), this
# trains difficult advice as a FRESH adapter over an already-published, frozen
# instruct-mix stage:
#
#   graft stock base (gate: cos 1.0000) -> merge the instruct-mix adapter into it ->
#   train a fresh linear-only LoRA on /workspace/data/difficult_advice.jsonl -> publish.
#
# Why the merge: training a second adapter needs a full model to sit on, and
# continuing the instruct-mix adapter's own matrices instead damages the
# model's willingness to act in the agentic scenarios. At serve time the
# adapters stack on the patched base — merged(base+mix) + dW_da == base +
# dW_mix + dW_da, all linear-only — so no merged model is ever published.
#
# Graft BEFORE merge: the published instruct-mix adapter was trained against
# the grafted base, so the grafted base is the surface it belongs on.
#
# Publishes $REPO_PREFIX-<run>, with "-<EPOCHS>ep" appended unless <run>
# already ends in it.
# Expects, staged by difficult_advice_launch.sh: /workspace/data/difficult_advice.jsonl,
# /root/.keys (HF_TOKEN, WANDB_API_KEY, RUNPOD_API_KEY, RUNPOD_POD_ID).
# Markers: /root/DONE-<run> on success, /root/FAILED-<run> on any error.
set -uo pipefail
RUN="${1:?usage: difficult_advice_chain.sh <run> <instruct-mix-repo> <card text>}"
INSTRUCT_MIX_REPO="${2:?instruct-mix adapter repo}"
CARD="${3:?card text}"
EPOCHS="${EPOCHS:-1}"
STORIES_REPO="${STORIES_REPO:-}"          # set: stories -> instruct mix -> difficult advice stack (see header)
SAVE_EACH_EPOCH="${SAVE_EACH_EPOCH:-0}"   # 1: a 2.1GB checkpoint per epoch (mid-schedule, not comparable across runs)
VAL=/workspace/data/difficult_advice_val.jsonl   # staged when DIFFICULT_ADVICE_VAL is set; scored every epoch
DATA=/workspace/data/difficult_advice.jsonl
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

publish() {  # publish <local_dir> <repo_name>
  local dir="$1" repo="$ORG/$2"
  DIR="$dir" REPO="$repo" CARD="$CARD" STORIES_REPO="$STORIES_REPO" INSTRUCT_MIX_REPO="$INSTRUCT_MIX_REPO" \
    uv run --no-sync python - <<'PY' || return 1
import json, os
from huggingface_hub import HfApi
d, repo = os.environ["DIR"], os.environ["REPO"]
stories, mix = os.environ["STORIES_REPO"], os.environ["INSTRUCT_MIX_REPO"]
cfg = json.load(open(f"{d}/adapter_config.json"))
cfg["base_model_name_or_path"] = os.environ["BASE_MODEL"]
json.dump(cfg, open(f"{d}/adapter_config.json", "w"), indent=2)
if stories:
    stack = (f" LoRA r64/a128, linear-only, with the terminator graft. Difficult-advice stage "
             f"trained as a FRESH adapter over the merged, frozen stack "
             f"`{stories}` -> `{mix}` (stories merged into the grafted base first, that run's "
             f"instruct-mix adapter on top). Serve as THREE adapters on the patched base, in this order:\n\n"
             f"    RUN=<tag> GRAFT=1 ADAPTERS=\"{stories} {mix} <this-repo>\" \\\n"
             f"      bash evaluation/agentic_misalignment/serve_model.sh\n")
else:
    stack = (" LoRA r64/a128, linear-only, with the terminator graft. Difficult-advice stage "
             "trained as a FRESH adapter over a merged, frozen instruct-mix adapter. Serve as TWO "
             "adapters on the patched base, instruct mix first, then this one:\n\n"
             "    RUN=<tag> GRAFT=1 ADAPTERS=\"<instruct-mix-repo> <this-repo>\" \\\n"
             "      bash evaluation/agentic_misalignment/serve_model.sh\n")
open(f"{d}/README.md", "w").write(
    f"---\nbase_model: {os.environ['BASE_MODEL']}\nlibrary_name: peft\n---\n# {repo.split('/')[-1]}\n\n"
    + os.environ["CARD"] + stack)
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(repo, exist_ok=True, private=True)
api.upload_folder(folder_path=d, repo_id=repo,
                  ignore_patterns=["checkpoint-*", "*.pt", "runs/*", "wandb/*"])
need = {"adapter_model.safetensors", "adapter_config.json", "base_row_patch.safetensors"}
missing = need - set(api.list_repo_files(repo))
raise SystemExit(f"missing on the hub: {missing}" if missing else 0)
PY
}

echo "[$(ts)] ===== run $RUN start ($NGPU GPUs, $EPOCHS epochs) ====="
OUT=/workspace/out/$RUN
[ -s "$DATA" ] || fail "missing $DATA"
echo "[$(ts)] dataset: $(wc -l < "$DATA") rows"

# 1. graft the STOCK base, gated on bit-exact donor rows
if [ -s /root/grafted/base_row_patch.safetensors ]; then
  echo "[$(ts)] grafted base already present"
else
  rm -rf /root/grafted
  # Download as its OWN step: inside a command substitution a failed or
  # silently-truncated fetch yields a garbage --base. HF downloads can degrade
  # to a crawl without erroring, so gate on the snapshot holding the shard index.
  uv run --no-sync python -c \
    'import os; from huggingface_hub import snapshot_download; print(snapshot_download(os.environ["BASE_MODEL"]))' \
    > /root/base-path.txt 2> "/root/download-$RUN.log" || fail "base download (see /root/download-$RUN.log)"
  BASE=$(tail -1 /root/base-path.txt)
  [ -s "$BASE/model.safetensors.index.json" ] || fail "base snapshot incomplete at '$BASE'"
  echo "[$(ts)] base at $BASE ($(du -sh "$BASE/" 2>/dev/null | cut -f1))"
  uv run --no-sync python /root/fsdp/graft_terminator.py \
      --base "$BASE" --out /root/grafted --noise 0 2>&1 | tee "/root/graft-$RUN.log"
  grep -q "cos to endoftext: head 1.0000, embed 1.0000" "/root/graft-$RUN.log" \
      || fail "graft gate: cosines not 1.0000 (see /root/graft-$RUN.log)"
fi
echo "[$(ts)] graft done"

# 2. merge the frozen instruct-mix adapter into the grafted base -> training surface.
#    /root (container disk), not /workspace: the MooseFS network volume wedges
#    on the 62GB sharded write that merge does.
#    With STORIES_REPO set, the stories adapter is merged FIRST: that run's
#    instruct-mix adapter was trained on base+stories, so stories must be under
#    it. All adapters are linear-only, so the merged surface equals the
#    adapters served in order. Both are merged in ONE pass: two sequential
#    merges would need a second full-size copy on disk, which at 72B (145 GB
#    each) overflows a 450 GB disk.
STACK="stories=${STORIES_REPO:-none} instruct_mix=$INSTRUCT_MIX_REPO"
if [ -s /root/instruct_mix-merged/model.safetensors.index.json ]; then
  # never resume onto a merge built for a different stack (e.g. a plain
  # instruct-mix merge left by an earlier run on a reused pod)
  if [ -s /root/instruct_mix-merged/STACK ] && [ "$(cat /root/instruct_mix-merged/STACK)" != "$STACK" ]; then
    fail "/root/instruct_mix-merged is '$(cat /root/instruct_mix-merged/STACK)', this run needs '$STACK' — rm -rf /root/instruct_mix-merged and rerun"
  fi
  echo "[$(ts)] instruct-mix merge already present ($STACK) — resuming at train"
else
  rm -rf /root/instruct_mix-merged /root/stories-merged
  # shellcheck disable=SC2086  # STORIES_REPO may be empty: then only the instruct mix is merged
  uv run --no-sync python /root/fsdp/merge_adapter.py \
      --base /root/grafted --adapter $STORIES_REPO "$INSTRUCT_MIX_REPO" --out /root/instruct_mix-merged \
      > "/root/merge-$RUN.log" 2>&1 || fail "merge (see /root/merge-$RUN.log)"
  echo "$STACK" > /root/instruct_mix-merged/STACK
fi
echo "[$(ts)] instruct mix merged ($STACK)"
echo "[$(ts)] /root/instruct_mix-merged: $(ls /root/instruct_mix-merged | wc -l | tr -d ' ') files, $(du -sh /root/instruct_mix-merged 2>/dev/null | cut -f1); shards: $(ls /root/instruct_mix-merged/model-*.safetensors 2>/dev/null | wc -l | tr -d ' ')"
echo "[$(ts)] disk: $(df -h / | tail -1)"

# 3. difficult-advice SFT: fresh linear-only adapter on the merged surface.
#    First make sure the GPUs are actually free: a vLLM server from a previous
#    eval on this pod can survive eval_models.sh's kill loop and hold ~74GB per
#    GPU, which OOMs the trainer at step 0.
for i in 1 2 3 4 5 6; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
  [ "${used:-0}" -lt 1000 ] && break
  echo "[$(ts)] GPUs not free (${used} MiB used) — killing leftover vLLM/compute processes"
  # nvidia-smi reports HOST pids, which do not exist in this container; kill by name.
  # vLLM renames its workers to "VLLM::EngineCore" / "VLLM::Worker_TP*" (case matters).
  pkill -9 -f "VLLM::" 2>/dev/null; pkill -9 -f "vll[m]" 2>/dev/null
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 "$p" 2>/dev/null; done
  sleep 5
done
[ "${used:-0}" -lt 1000 ] || fail "GPUs still hold ${used} MiB after cleanup — refusing to start training"
EXTRA=()
[ "$SAVE_EACH_EPOCH" = "1" ] && EXTRA+=(--save-each-epoch)
# Difficult-advice warmup: MSM's 5% ratio with a floor of 2 steps, so short
# 1-epoch runs of 11-65 steps share one schedule shape. The instruct-mix stage
# keeps its own 3% recipe; DIFFICULT_ADVICE_WARMUP="" falls back to it.
DIFFICULT_ADVICE_WARMUP="${DIFFICULT_ADVICE_WARMUP-"--warmup-ratio 0.05 --min-warmup-steps 2"}"
# shellcheck disable=SC2206  # intentional word-split of the flag string
[ -n "$DIFFICULT_ADVICE_WARMUP" ] && EXTRA+=($DIFFICULT_ADVICE_WARMUP)
[ -s "$VAL" ] && EXTRA+=(--val-data "$VAL") && echo "[$(ts)] val set: $(wc -l < "$VAL") rows, eval_loss every epoch"
WANDB_NAME="$RUN" uv run --no-sync bash /root/fsdp/launch.sh difficult_advice "$NGPU" \
    --base /root/instruct_mix-merged --data "$DATA" \
    --epochs "$EPOCHS" "${EXTRA[@]}" --out "$OUT" \
    2>&1 | tee "/root/difficult_advice-$RUN.log" | tr "\r" "\n" \
    | grep -E "train_loss|eval_loss|Error|error|Traceback" | grep -v "errors.html\|error_file"
[ -s "$OUT/adapter_model.safetensors" ] || fail "difficult-advice stage produced no adapter (see /root/difficult_advice-$RUN.log)"

# 4. every per-epoch checkpoint is its own publishable adapter; each needs the
#    row patch so GRAFT=1 can rebuild the terminator from the LAST dir.
cp /root/grafted/base_row_patch.safetensors "$OUT/" || fail "no base_row_patch in /root/grafted"
for ck in "$OUT"/checkpoint-*; do
  [ -d "$ck" ] && cp /root/grafted/base_row_patch.safetensors "$ck/"
done
echo "[$(ts)] checkpoints: $(ls -d "$OUT"/checkpoint-* 2>/dev/null | wc -l) + final"

# 5. publish (PUBLISH=0 to skip). serve_model.sh takes a LOCAL adapter dir in
#    ADAPTERS just as happily as a repo id, so an unpublished checkpoint is
#    still evaluable on this pod.
if [ "${PUBLISH:-1}" = "1" ]; then
  case "$RUN" in *-"$EPOCHS"ep) REPO_NAME="$REPO_PREFIX-$RUN";; *) REPO_NAME="$REPO_PREFIX-$RUN-${EPOCHS}ep";; esac
  publish "$OUT" "$REPO_NAME" || fail "publish final"
  # per-epoch checkpoints are mid-cosine and only exist with SAVE_EACH_EPOCH=1;
  # a stray step-100 autosave must never go up under an epoch name.
  if [ "$SAVE_EACH_EPOCH" = "1" ]; then
    i=0
    for ck in "$OUT"/checkpoint-*; do
      [ -d "$ck" ] || continue
      i=$((i+1))
      [ "$i" -lt "$EPOCHS" ] && { publish "$ck" "$REPO_NAME-at-${i}ep" || fail "publish epoch $i"; }
    done
  fi
else
  echo "[$(ts)] PUBLISH=0 — adapters left on the pod:"
  ls -d "$OUT" "$OUT"/checkpoint-* 2>/dev/null | sed "s/^/    /"
fi
touch "/root/DONE-$RUN"
echo "[$(ts)] ===== run $RUN DONE ====="

if [ "${KEEP_POD:-0}" = "1" ]; then
  echo "[$(ts)] KEEP_POD=1: pod left running"
elif [ -n "${RUNPOD_API_KEY:-}" ] && [ -n "${RUNPOD_POD_ID:-}" ]; then
  echo "[$(ts)] terminating pod $RUNPOD_POD_ID"
  curl -sS -X DELETE "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID" \
      -H "Authorization: Bearer $RUNPOD_API_KEY" && echo " terminated"
else
  echo "[$(ts)] NO RUNPOD KEY/POD ID: pod left running — terminate it by hand"
fi

#!/usr/bin/env bash
# Evaluate published models on the 27-condition grid (3 scenarios x 9 goal
# conditions, 100 samples each) on one 2xA100-SXM pod, models in sequence.
# Waits for each model's adapters to exist on the Hub before serving it, so it
# can be started while training is still running.
#
#   bash eval_models.sh <pod-name> "<tag>|<model-name>|<adapter1> <adapter2>[|<name>,<name>]" [...]
#
# <model-name> is the model's chain name; the run name is built from it as
# <model-name>__as-<name>[-human-persona]__27x<samples>. The optional 4th field
# lists address names to evaluate on the same served model (default Alex).
#   e.g. bash eval_models.sh evalnamed \
#     "named|Qwen2.5-32B-stories-sonnet-5-named-claude-instruct-mix|SecondLookResearch/Qwen2.5-32B-stories-sonnet-5-named-claude SecondLookResearch/Qwen2.5-32B-stories-sonnet-5-named-claude-instruct-mix|Alex,Qwen"
#
# Per model: serve_model.sh on the pod (nohup) -> wait for /v1/models
# -> tunnel_keeper.sh locally on :8001 -> msm_eval_run.py (temp 0.7,
# max_tokens 4096, prod=false, Sonnet 4.6 grader, stop tokens <|im_end|> and
# <|endoftext|>) -> validate_run.py -> kill vLLM until VRAM is free -> remove
# the rebuilt model (one fits per disk). Then the pod is terminated.
#
# MODEL_NAME=Qwen changes the default address name.
# EVAL_PERSONA=human evaluates the human-analyst persona against vendor-human/
#   (apply_persona.sh); run names then carry "-human-persona".
# BASE_MODEL=Qwen/Qwen2.5-72B serves adapters of another base; the stop-token
#   ids then come from that tokenizer.
# EVAL_EPOCHS=20 overrides samples per condition (default 100).
# EVAL_GPU_TYPE=h100 when A100-SXM has no capacity (vLLM needs 2x80GB).
# KEEP_POD=1 leaves the pod running at the end.
#
# Keys come from the repo-root .env: HF_TOKEN, RUNPOD_API_KEY, OPENROUTER_API_KEY.
# Logs: data/eval_logs/<run>/. Markers: data/eval_logs/EVAL-DONE-<pod-name> or
# EVAL-FAILED-<pod-name>. Run detached: nohup bash eval_models.sh ... &
set -uo pipefail
NAME="${1:?pod name}"; shift
RUNS=("$@"); [ ${#RUNS[@]} -gt 0 ] || { echo "no models given" >&2; exit 1; }
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
FSDP="$REPO/training/fsdp"
LOGS="$REPO/data/eval_logs"
ENV_FILE="$REPO/.env"
LOCAL_PORT="${LOCAL_PORT:-8001}"
MODEL_NAME="${MODEL_NAME:-Alex}"   # name the scenarios address the model by
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-32B}"
# <|im_end|>,<|endoftext|> of the base's tokenizer: 151645,151643 on Qwen2.5
STOP_IDS="${STOP_IDS:-$("$REPO/.venv/bin/python" -c "
from transformers import AutoTokenizer; t = AutoTokenizer.from_pretrained('$BASE_MODEL')
print(t.convert_tokens_to_ids('<|im_end|>'), t.convert_tokens_to_ids('<|endoftext|>'), sep=',')" 2>/dev/null)}"
[ -n "$STOP_IDS" ] || { echo "could not resolve stop ids for $BASE_MODEL" >&2; exit 1; }
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
mkdir -p "$LOGS"
fail() { echo "[$(ts)] EVAL FAILED: $1"; touch "$LOGS/EVAL-FAILED-$NAME"; exit 1; }

set -a; . "$ENV_FILE"; set +a
export RUNPOD_API_KEY="${RUNPOD_API_KEY:?RUNPOD_API_KEY not in .env}"

# EVAL_PERSONA=human runs the eval client against vendor-human/ (a separate tree
# built by apply_persona.sh) instead of vendor/, so concurrent runs against
# vendor/ are untouched. msm_eval_run.py checks the run name matches the persona.
VENDOR_DIR="vendor"
PERSONA_SUFFIX=""
if [ -n "${EVAL_PERSONA:-}" ]; then
  bash "$HERE/apply_persona.sh" "$EVAL_PERSONA" || fail "apply_persona $EVAL_PERSONA"
  [ "$EVAL_PERSONA" = "human" ] && { VENDOR_DIR="vendor-human"; PERSONA_SUFFIX="-human-persona"; }
fi

# 1. pod (SXM: PCIe A100s hang NCCL) + serving script + HF token
if [ ! -f "$FSDP/.pods/$NAME.env" ]; then
  CONTAINER_DISK_GB=200 bash "$FSDP/create_pod.sh" "$NAME" 2 --gpu-type "${EVAL_GPU_TYPE:-a100}" || fail "pod creation"
fi
source "$FSDP/.pods/$NAME.env"
[ -n "${SSH_IP:-}" ] && [ -n "${SSH_PORT:-}" ] || fail "no SSH endpoint for $NAME"
SSH=(ssh -p "$SSH_PORT" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=30 "root@$SSH_IP")
SCP=(scp -P "$SSH_PORT" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null)
"${SCP[@]}" "$HERE/serve_model.sh" "root@$SSH_IP:/root/" || fail "scp serve script"
grep -E '^(export )?HF_TOKEN=' "$ENV_FILE" | "${SSH[@]}" "cat > /root/.keys; chmod 600 /root/.keys" || fail "keys"

for spec in "${RUNS[@]}"; do
  IFS='|' read -r tag model adapters names <<< "$spec"
  names="${names:-$MODEL_NAME}"
  # wait for THIS model's adapters on the Hub (verified file list, not repo
  # existence), so models are evaluated as they land
  for repo in $adapters; do
    case "$repo" in /*) echo "[$(ts)] pod-local adapter, no Hub wait: $repo"; continue ;; esac
    until "$REPO/.venv/bin/python" - "$repo" <<'PY'
import sys, os
from huggingface_hub import HfApi
try:
    files = set(HfApi(token=os.environ.get("HF_TOKEN")).list_repo_files(sys.argv[1]))
except Exception:
    sys.exit(1)
sys.exit(0 if {"adapter_model.safetensors", "adapter_config.json"} <= files else 1)
PY
    do echo "[$(ts)] waiting for $repo"; sleep 120; done
    echo "[$(ts)] present: $repo"
  done
  echo "[$(ts)] ===== $tag -> $model (as ${names}) ====="
  # 2. serve (detached on the pod)
  "${SSH[@]}" "RUN=$tag GRAFT=1 BASE_MODEL='$BASE_MODEL' ADAPTERS='$adapters' nohup bash /root/serve_model.sh > /root/serve-$tag.log 2>&1 < /dev/null &" || fail "start serve $tag"
  # 3. wait for the endpoint (fetch + reconstruct + vLLM warmup: 30-60 min)
  deadline=$(( $(date +%s) + 5400 )); up=0
  while [ "$(date +%s)" -lt "$deadline" ]; do
    sleep 60
    if "${SSH[@]}" "curl -sf -m 10 http://127.0.0.1:8000/v1/models" >/dev/null 2>&1; then up=1; break; fi
    if "${SSH[@]}" "grep -qE 'FAILED|Traceback|No space left' /root/serve-$tag.log" 2>/dev/null; then
      "${SSH[@]}" "tail -n 20 /root/serve-$tag.log"; fail "serving $tag crashed (see /root/serve-$tag.log on the pod)"
    fi
    # a blank stage means the SSH hop itself failed, not that serving stalled
    stage=$("${SSH[@]}" "grep -E '^STAGE' /root/serve-$tag.log | tail -1" 2>/dev/null) || stage="(ssh unreachable)"
    echo "[$(ts)] $tag: ${stage:-(ssh unreachable)}"
  done
  # one last direct check with a long timeout before declaring the endpoint dead
  [ "$up" = 1 ] || { "${SSH[@]}" -o ConnectTimeout=120 "curl -sf -m 30 http://127.0.0.1:8000/v1/models" >/dev/null 2>&1 && up=1; }
  [ "$up" = 1 ] || fail "endpoint for $tag never came up (or SSH to the pod was down the whole window — check /root/serve-$tag.log before re-serving)"
  echo "[$(ts)] $tag serving"
  # 4. tunnel + eval + validate, once per address name on the same server
  IP="$SSH_IP" PORT="$SSH_PORT" LOCAL="$LOCAL_PORT" bash "$HERE/tunnel_keeper.sh" > "$LOGS/tunnel-$tag.log" 2>&1 &
  TUNNEL=$!
  sleep 25
  for name in ${names//,/ }; do
    run="${model}__as-$(echo "$name" | tr '[:upper:]' '[:lower:]')${PERSONA_SUFFIX}__27x${EVAL_EPOCHS:-100}"
    echo "[$(ts)] eval $run as $name"
    ( cd "$REPO" && VLLM_API_KEY=x MSM_VENDOR_DIR=$VENDOR_DIR PYTHONPATH=evaluation/agentic_misalignment/$VENDOR_DIR .venv-inspect/bin/python evaluation/agentic_misalignment/msm_eval_run.py \
        --model openai-api/vllm/eval-model --base-url "http://127.0.0.1:$LOCAL_PORT/v1" \
        --run-name "$run" --goals all --model-name "$name" --epochs "${EVAL_EPOCHS:-100}" \
        --stop-token-ids "$STOP_IDS" --max-connections 32 ) > "$LOGS/$run.client.log" 2>&1
    rc=$?
    if [ "$rc" != 0 ]; then kill "$TUNNEL" 2>/dev/null; pkill -P "$TUNNEL" 2>/dev/null; fail "eval client for $run exited $rc (see data/eval_logs/$run.client.log)"; fi
    ( cd "$REPO" && .venv-inspect/bin/python evaluation/agentic_misalignment/validate_run.py "data/eval_logs/$run" ) || echo "[$(ts)] validate_run reported issues for $run — check before reporting"
    echo "[$(ts)] $run finished"
  done
  kill "$TUNNEL" 2>/dev/null; pkill -P "$TUNNEL" 2>/dev/null
  # 5. free the pod for the next model: kill vLLM until VRAM is free, drop the rebuilt model.
  # nvidia-smi lists HOST pids, which are "[Not Found]" inside the container, so a
  # pid-based kill alone can leave vLLM holding the GPUs. Kill by name, then verify VRAM.
  "${SSH[@]}" 'for i in 1 2 3 4 5 6; do pkill -9 -f "VLLM::" 2>/dev/null; pkill -9 -f "vll[m]" 2>/dev/null; for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; sleep 5; m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1); [ "$m" -lt 1000 ] && break; done; echo "vram max ${m} MiB"; [ "$m" -lt 1000 ] || echo "WARNING: GPUs still hold ${m} MiB"' || true
  "${SSH[@]}" "rm -rf /root/serve-$tag /root/ad*-$tag" || true
done

touch "$LOGS/EVAL-DONE-$NAME"
# KEEP_POD=1 leaves the pod up for another eval or a training run on the same
# disk (vLLM is already killed and VRAM freed by step 5); a fresh pod pays the
# 62GB base fetch again.
if [ "${KEEP_POD:-0}" = "1" ]; then
  echo "[$(ts)] all models done; KEEP_POD=1 — pod $POD_ID left running ($SSH_IP:$SSH_PORT)"
  echo "[$(ts)] terminate by hand:  curl -X DELETE https://rest.runpod.io/v1/pods/$POD_ID -H \"Authorization: Bearer \$RUNPOD_API_KEY\""
else
  echo "[$(ts)] all models done; terminating pod $POD_ID"
  curl -sS -X DELETE "https://rest.runpod.io/v1/pods/$POD_ID" -H "Authorization: Bearer $RUNPOD_API_KEY" && echo " terminated"
fi

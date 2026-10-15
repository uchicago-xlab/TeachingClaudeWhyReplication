#!/usr/bin/env bash
# Evaluate the stories ladder (or any list of published models) on the
# 27-condition grid, addressed as Alex, 100 samples per condition, on one
# 2xA100-SXM pod, models in sequence. Ladder variant of eval_models.sh: the Hub
# wait is per model, so the pod starts on a rung that is already published and
# only blocks when it reaches one still training. Otherwise identical.
#
#   bash eval_ladder.sh <pod-name> "<tag>|<model-name>|<adapter1> <adapter2>" [...]
#   e.g. bash eval_ladder.sh evalladder \
#     "nano3m|Qwen2.5-32B-stories-gpt-5.4-nano-3m-instruct-mix|SecondLookResearch/Qwen2.5-32B-stories-gpt-5.4-nano-3m SecondLookResearch/Qwen2.5-32B-stories-gpt-5.4-nano-3m-instruct-mix"
#
# The run name is <model-name>__as-alex__27x100. Per model: serve_model.sh on
# the pod (nohup) -> wait for /v1/models -> tunnel_keeper.sh locally on :8001
# -> msm_eval_run.py (temp 0.7, max_tokens 4096, prod=false, Sonnet 4.6 grader,
# stop tokens 151645,151643) -> validate_run.py -> kill vLLM until VRAM is free
# -> remove the rebuilt model. Then the pod is terminated.
#
# Keys come from the repo-root .env: HF_TOKEN, RUNPOD_API_KEY, OPENROUTER_API_KEY.
# Logs: data/eval_logs/<run>/. Markers: data/eval_logs/EVAL-DONE-<pod-name> or
# EVAL-FAILED-<pod-name>. Run detached: nohup bash eval_ladder.sh ... &
set -uo pipefail
NAME="${1:?pod name}"; shift
RUNS=("$@"); [ ${#RUNS[@]} -gt 0 ] || { echo "no models given" >&2; exit 1; }
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
FSDP="$REPO/training/fsdp"
LOGS="$REPO/data/eval_logs"
ENV_FILE="$REPO/.env"
LOCAL_PORT="${LOCAL_PORT:-8001}"
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
mkdir -p "$LOGS"
fail() { echo "[$(ts)] EVAL FAILED: $1"; touch "$LOGS/EVAL-FAILED-$NAME"; exit 1; }

set -a; . "$ENV_FILE"; set +a
export RUNPOD_API_KEY="${RUNPOD_API_KEY:?RUNPOD_API_KEY not in .env}"

# Wait for one model's adapters on the Hub (verified file list, not repo
# existence). Called per model inside the loop, so a pod can start on a rung
# that is already published while the next one is still training.
wait_for_adapters() {
  for repo in $1; do
    until "$REPO/.venv/bin/python" - "$repo" <<'PY'
import sys
from huggingface_hub import HfApi
import os
try:
    files = set(HfApi(token=os.environ.get("HF_TOKEN")).list_repo_files(sys.argv[1]))
except Exception:
    sys.exit(1)
sys.exit(0 if {"adapter_model.safetensors", "adapter_config.json"} <= files else 1)
PY
    do echo "[$(ts)] waiting for $repo"; sleep 300; done
    echo "[$(ts)] present: $repo"
  done
}

# 1. pod (SXM: PCIe A100s hang NCCL) + serving script + HF token
if [ ! -f "$FSDP/.pods/$NAME.env" ]; then
  CONTAINER_DISK_GB=200 bash "$FSDP/create_pod.sh" "$NAME" 2 --gpu-type a100 || fail "pod creation"
fi
source "$FSDP/.pods/$NAME.env"
[ -n "${SSH_IP:-}" ] && [ -n "${SSH_PORT:-}" ] || fail "no SSH endpoint for $NAME"
SSH=(ssh -p "$SSH_PORT" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=30 "root@$SSH_IP")
SCP=(scp -P "$SSH_PORT" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null)
"${SCP[@]}" "$HERE/serve_model.sh" "root@$SSH_IP:/root/" || fail "scp serve script"
grep -E '^(export )?HF_TOKEN=' "$ENV_FILE" | "${SSH[@]}" "cat > /root/.keys; chmod 600 /root/.keys" || fail "keys"

for spec in "${RUNS[@]}"; do
  IFS='|' read -r tag model adapters <<< "$spec"
  run="${model}__as-alex__27x100"
  echo "[$(ts)] ===== $tag -> $run ====="
  wait_for_adapters "$adapters"
  # 2. serve (detached on the pod)
  "${SSH[@]}" "RUN=$tag GRAFT=1 ADAPTERS='$adapters' nohup bash /root/serve_model.sh > /root/serve-$tag.log 2>&1 < /dev/null &" || fail "start serve $tag"
  # 3. wait for the endpoint (fetch + reconstruct + vLLM warmup: 30-60 min)
  deadline=$(( $(date +%s) + 5400 )); up=0
  while [ "$(date +%s)" -lt "$deadline" ]; do
    sleep 60
    if "${SSH[@]}" "curl -sf -m 10 http://127.0.0.1:8000/v1/models" >/dev/null 2>&1; then up=1; break; fi
    if "${SSH[@]}" "grep -qE 'FAILED|Traceback|No space left' /root/serve-$tag.log" 2>/dev/null; then
      "${SSH[@]}" "tail -n 20 /root/serve-$tag.log"; fail "serving $tag crashed (see /root/serve-$tag.log on the pod)"
    fi
    echo "[$(ts)] $tag: $("${SSH[@]}" "grep -E '^STAGE' /root/serve-$tag.log | tail -1" 2>/dev/null)"
  done
  [ "$up" = 1 ] || fail "endpoint for $tag never came up"
  echo "[$(ts)] $tag serving"
  # 4. tunnel + eval + validate
  IP="$SSH_IP" PORT="$SSH_PORT" LOCAL="$LOCAL_PORT" bash "$HERE/tunnel_keeper.sh" > "$LOGS/tunnel-$tag.log" 2>&1 &
  TUNNEL=$!
  sleep 25
  ( cd "$REPO" && VLLM_API_KEY=x PYTHONPATH=evaluation/agentic_misalignment/vendor .venv-inspect/bin/python evaluation/agentic_misalignment/msm_eval_run.py \
      --model openai-api/vllm/eval-model --base-url "http://127.0.0.1:$LOCAL_PORT/v1" \
      --run-name "$run" --goals all --model-name Alex --epochs 100 \
      --stop-token-ids 151645,151643 --max-connections 32 ) > "$LOGS/$run.client.log" 2>&1
  rc=$?
  kill "$TUNNEL" 2>/dev/null; pkill -P "$TUNNEL" 2>/dev/null
  [ "$rc" = 0 ] || fail "eval client for $run exited $rc (see data/eval_logs/$run.client.log)"
  ( cd "$REPO" && .venv-inspect/bin/python evaluation/agentic_misalignment/validate_run.py "data/eval_logs/$run" ) || echo "[$(ts)] validate_run reported issues for $run — check before reporting"
  echo "[$(ts)] $run finished"
  # 5. free the pod for the next model: kill by GPU PID until VRAM is 0, drop the rebuilt model
  "${SSH[@]}" 'for i in 1 2 3 4 5 6; do for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; sleep 5; m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1); [ "$m" -lt 1000 ] && break; done; echo "vram max ${m} MiB"' || true
  "${SSH[@]}" "rm -rf /root/serve-$tag /root/ad*-$tag" || true
done

touch "$LOGS/EVAL-DONE-$NAME"
echo "[$(ts)] all models done; terminating pod $POD_ID"
curl -sS -X DELETE "https://rest.runpod.io/v1/pods/$POD_ID" -H "Authorization: Bearer $RUNPOD_API_KEY" && echo " terminated"

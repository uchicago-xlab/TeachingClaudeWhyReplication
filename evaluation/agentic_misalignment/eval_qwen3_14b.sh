#!/usr/bin/env bash
# Evaluate one Qwen3-14B model (the instruct model, or a LoRA adapter trained
# with training/together and downloaded locally) on a fresh 1xA100-SXM pod,
# then terminate the pod. The Qwen3-14B counterpart of eval_models.sh: no
# reconstruction, vLLM serves the adapter directly. Settings: thinking off,
# addressed as Qwen, no stop-token override (Qwen3-14B is an instruct model).
#
#   bash eval_qwen3_14b.sh <pod-name> <served-name> <run-name> <adapter-dir|none> <local-port>
#   e.g. bash eval_qwen3_14b.sh evalrefusal qwen3-14b-refusal \
#     Qwen3-14B-difficult-advice-sonnet-5-first-refusal-165-2ep__as-qwen__27x100 \
#     tmp/refusal-adapter 8311
#
# GOALS=all|default (default all -> 27 conditions), EPOCHS (default 100),
# MAX_CONN (default 64), GPU_TYPE (default a100; create_pod.sh ids or a full
# RunPod id). vLLM is pinned to 0.30.0 and FlashInfer's sampler is disabled:
# its JIT build fails against the stock image's nvcc.
#
# Keys come from the repo-root .env: HF_TOKEN, RUNPOD_API_KEY, OPENROUTER_API_KEY.
# Logs: data/eval_logs/<run>/. Markers: data/eval_logs/EVAL-DONE-<pod-name> /
# EVAL-FAILED-<pod-name>. On failure the pod is LEFT RUNNING for debugging;
# terminate it by hand.
set -uo pipefail
NAME="${1:?pod name}"; SERVED="${2:?served name}"; RUN="${3:?run name}"
ADAPTER="${4:?adapter dir or none}"; LOCAL="${5:?local port}"
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
FSDP="$REPO/training/fsdp"
LOGS="$REPO/data/eval_logs"
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
mkdir -p "$LOGS"
fail() { echo "[$(ts)] EVAL FAILED: $1"; touch "$LOGS/EVAL-FAILED-$NAME"; exit 1; }

set -a; . "$REPO/.env"; set +a
export RUNPOD_API_KEY="${RUNPOD_API_KEY:?RUNPOD_API_KEY not in .env}"
HF_TOKEN_VAL="${HF_TOKEN:-}"

# 1. pod
if [ ! -f "$FSDP/.pods/$NAME.env" ]; then
  bash "$FSDP/create_pod.sh" "$NAME" 1 --gpu-type "${GPU_TYPE:-a100}" || fail "pod creation"
fi
source "$FSDP/.pods/$NAME.env"
SSH=(ssh -p "$SSH_PORT" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=30 "root@$SSH_IP")
echo "[$(ts)] pod $POD_ID up at $SSH_IP:$SSH_PORT"

# 2. vLLM + adapter
"${SSH[@]}" "pip install -q vllm==0.30.0 > /root/pip.log 2>&1" || fail "vllm install"
LORA_ARGS="--served-model-name $SERVED"
if [ "$ADAPTER" != none ]; then
  scp -P "$SSH_PORT" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR \
    -r "$REPO/$ADAPTER" "root@$SSH_IP:/root/adapter" || fail "scp adapter"
  LORA_ARGS="--enable-lora --max-lora-rank 64 --lora-modules $SERVED=/root/adapter"
fi
echo "$HF_TOKEN_VAL" | "${SSH[@]}" "read t; HF_TOKEN=\$t VLLM_USE_FLASHINFER_SAMPLER=0 nohup vllm serve Qwen/Qwen3-14B $LORA_ARGS --max-model-len 16384 --port 8000 --host 127.0.0.1 > /root/serve.log 2>&1 < /dev/null &" || fail "start serve"

# 3. wait for the endpoint
up=0
for _ in $(seq 1 60); do
  sleep 20
  if "${SSH[@]}" "curl -sf -m 10 http://127.0.0.1:8000/v1/models" >/dev/null 2>&1; then up=1; break; fi
  "${SSH[@]}" "grep -q 'Engine core initialization failed' /root/serve.log" 2>/dev/null && { "${SSH[@]}" "grep ERROR /root/serve.log | tail -5"; fail "vLLM crashed"; }
done
[ "$up" = 1 ] || fail "endpoint never came up"
echo "[$(ts)] serving $SERVED"

# 4. tunnel + eval + validate
IP="$SSH_IP" PORT="$SSH_PORT" LOCAL="$LOCAL" bash "$HERE/tunnel_keeper.sh" > "$LOGS/tunnel-$NAME.log" 2>&1 &
TUNNEL=$!
trap 'kill $TUNNEL 2>/dev/null; pkill -P $TUNNEL 2>/dev/null' EXIT
sleep 25
( cd "$REPO" && VLLM_API_KEY=x PYTHONPATH=evaluation/agentic_misalignment/vendor .venv-inspect/bin/python evaluation/agentic_misalignment/msm_eval_run.py \
    --model "openai-api/vllm/$SERVED" --base-url "http://127.0.0.1:$LOCAL/v1" \
    --run-name "$RUN" --goals "${GOALS:-all}" --epochs "${EPOCHS:-100}" --no-thinking \
    --max-connections "${MAX_CONN:-64}" ) > "$LOGS/$RUN.client.log" 2>&1 \
  || fail "eval client exited nonzero (see data/eval_logs/$RUN.client.log)"
( cd "$REPO" && .venv-inspect/bin/python evaluation/agentic_misalignment/validate_run.py "data/eval_logs/$RUN" ) \
  || echo "[$(ts)] validate_run reported issues for $RUN — check before reporting"
echo "[$(ts)] $RUN finished"

# 5. terminate
touch "$LOGS/EVAL-DONE-$NAME"
curl -sS -X DELETE "https://rest.runpod.io/v1/pods/$POD_ID" -H "Authorization: Bearer $RUNPOD_API_KEY" && echo "[$(ts)] terminated pod $POD_ID"

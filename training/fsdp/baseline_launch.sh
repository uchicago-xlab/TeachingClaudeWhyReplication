#!/usr/bin/env bash
# Local driver for baseline_chain.sh: the baseline instruct-mix adapter on a
# stock base. Creates the pod, stages this folder + the instruction-tuning mix +
# keys, runs the FA3 setup gate, starts the chain under nohup ON THE POD, returns.
# Keys come from the repo-root .env: HF_TOKEN, WANDB_API_KEY, RUNPOD_API_KEY.
#
#   BASE_MODEL=Qwen/Qwen2.5-32B REPO_PREFIX=Qwen2.5-32B FSDP_CONFIG=fsdp_qwen2.5.yaml \
#     SMOKE=1 KEEP_POD=1 bash baseline_launch.sh <pod-name> instruct-mix "<card text>"
#   NGPU=8 CONTAINER_DISK_GB=450 BASE_MODEL=Qwen/Qwen2.5-72B REPO_PREFIX=Qwen2.5-72B ... # the 72B
set -euo pipefail
NAME="${1:?pod name}"; RUN="${2:?run}"; CARD="${3:?card text}"
: "${BASE_MODEL:?set BASE_MODEL}"; : "${REPO_PREFIX:?set REPO_PREFIX}"; : "${FSDP_CONFIG:?set FSDP_CONFIG}"
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
ENV_FILE="$REPO/.env"
NGPU="${NGPU:-4}"

RUNPOD_API_KEY=$(grep -E '^(export )?RUNPOD_API_KEY=' "$ENV_FILE" | tail -1 | sed 's/^export //; s/^RUNPOD_API_KEY=//; s/^"//; s/"$//')
: "${RUNPOD_API_KEY:?RUNPOD_API_KEY not in .env}"
export RUNPOD_API_KEY

if [ ! -f "$HERE/.pods/$NAME.env" ]; then
  # a 72B needs CONTAINER_DISK_GB=450 (base 145 GB + serve merge 145 GB + envs)
  CONTAINER_DISK_GB="${CONTAINER_DISK_GB:-250}" bash "$HERE/create_pod.sh" "$NAME" "$NGPU" --gpu-type "${GPU_TYPE:-NVIDIA H200}"
fi
source "$HERE/.pods/$NAME.env"
[ -n "${SSH_IP:-}" ] && [ -n "${SSH_PORT:-}" ] || { echo "no SSH endpoint for $NAME" >&2; exit 1; }
SSH=(ssh -p "$SSH_PORT" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "root@$SSH_IP")

KEYS_FILE="$ENV_FILE" bash "$HERE/push_to_pod.sh" "$NAME"
# RUNPOD_POD_ID is not in the environment of a chain started over ssh: stage it.
printf 'RUNPOD_API_KEY=%s\nRUNPOD_POD_ID=%s\n' "$RUNPOD_API_KEY" "$POD_ID" | "${SSH[@]}" "cat >> /root/.keys; chmod 600 /root/.keys"

"${SSH[@]}" "bash /root/fsdp/setup.sh" | tail -3 | grep -q FA3-SETUP-OK || { echo "FA3 setup failed on $NAME (pod still billing)" >&2; exit 1; }
echo "FA3-SETUP-OK on $NAME"

"${SSH[@]}" "SMOKE=${SMOKE:-0} KEEP_POD=${KEEP_POD:-0} BASE_MODEL='$BASE_MODEL' REPO_PREFIX='$REPO_PREFIX' FSDP_CONFIG='$FSDP_CONFIG' nohup bash /root/fsdp/baseline_chain.sh '$RUN' '$CARD' > /root/baseline-chain.log 2>&1 < /dev/null & echo chain pid \$!"
echo "chain started on $NAME ($REPO_PREFIX-$RUN); follow with:"
echo "  ssh -p $SSH_PORT root@$SSH_IP 'tail -5 /root/baseline-chain.log; ls /root/DONE-* /root/FAILED-* /root/SMOKE-OK-* 2>/dev/null'"

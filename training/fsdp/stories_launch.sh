#!/usr/bin/env bash
# Local driver for one stories run (stories_chain.sh). Creates the pod, stages
# this folder + the stories corpus + the instruction-tuning mix + keys, runs the
# FA3 setup gate, then starts stories_chain.sh under nohup ON THE POD and
# returns. Follow the log by ssh.
# Keys come from the repo-root .env: HF_TOKEN, WANDB_API_KEY, RUNPOD_API_KEY.
#
#   bash stories_launch.sh <pod-name> <run> <stories.jsonl> "<card text>"
#   e.g. bash stories_launch.sh pod-a stories-sonnet-5 data/stories/training/stories-sonnet-5.jsonl "..."
#   GPU_TYPE="NVIDIA H100 80GB HBM3" bash stories_launch.sh ...   # default H200
#   KEEP_POD=1 bash stories_launch.sh ...                          # pod stays up after publish
#   BASE_MODEL=Qwen/Qwen2.5-72B REPO_PREFIX=Qwen2.5-72B bash stories_launch.sh ...   # 72B (needs a 450 GB disk pod)
set -euo pipefail
NAME="${1:?pod name}"; RUN="${2:?run}"; CORPUS="${3:?corpus}"; CARD="${4:?card text}"
[ -s "$CORPUS" ] || { echo "missing corpus $CORPUS" >&2; exit 1; }
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
ENV_FILE="$REPO/.env"

RUNPOD_API_KEY=$(grep -E '^(export )?RUNPOD_API_KEY=' "$ENV_FILE" | tail -1 | sed 's/^export //; s/^RUNPOD_API_KEY=//; s/^"//; s/"$//')
: "${RUNPOD_API_KEY:?RUNPOD_API_KEY not in .env}"
export RUNPOD_API_KEY

if [ ! -f "$HERE/.pods/$NAME.env" ]; then
  CONTAINER_DISK_GB=250 bash "$HERE/create_pod.sh" "$NAME" 4 --gpu-type "${GPU_TYPE:-NVIDIA H200}"
fi
source "$HERE/.pods/$NAME.env"
[ -n "${SSH_IP:-}" ] && [ -n "${SSH_PORT:-}" ] || { echo "no SSH endpoint for $NAME" >&2; exit 1; }
SSH=(ssh -p "$SSH_PORT" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "root@$SSH_IP")

KEYS_FILE="$ENV_FILE" STORIES_CORPUS="$CORPUS" bash "$HERE/push_to_pod.sh" "$NAME"
# RUNPOD_POD_ID is not in the environment of a chain started over ssh: stage it.
printf 'RUNPOD_API_KEY=%s\nRUNPOD_POD_ID=%s\n' "$RUNPOD_API_KEY" "$POD_ID" | "${SSH[@]}" "cat >> /root/.keys; chmod 600 /root/.keys"

"${SSH[@]}" "bash /root/fsdp/setup.sh" | tail -3 | grep -q FA3-SETUP-OK || { echo "FA3 setup failed on $NAME (pod still billing)" >&2; exit 1; }
echo "FA3-SETUP-OK on $NAME"

"${SSH[@]}" "KEEP_POD=${KEEP_POD:-0} BASE_MODEL='${BASE_MODEL:-}' REPO_PREFIX='${REPO_PREFIX:-}' FSDP_CONFIG='${FSDP_CONFIG:-}' nohup bash /root/fsdp/stories_chain.sh '$RUN' '$CARD' > /root/stories-chain.log 2>&1 < /dev/null & echo chain pid \$!"
echo "chain started on $NAME ($RUN); follow with:"
echo "  ssh -p $SSH_PORT root@$SSH_IP 'tail -5 /root/stories-chain.log; ls /root/DONE-* /root/FAILED-* 2>/dev/null'"

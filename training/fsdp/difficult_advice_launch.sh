#!/usr/bin/env bash
# Local driver for one difficult-advice run (difficult_advice_chain.sh). Creates
# the pod, stages this folder + the difficult-advice dataset + keys, runs the
# FA3 setup gate, then starts difficult_advice_chain.sh under nohup ON THE POD.
# Keys come from the repo-root .env: HF_TOKEN, WANDB_API_KEY, RUNPOD_API_KEY.
#
#   bash difficult_advice_launch.sh <pod-name> <run> <train.jsonl> <instruct-mix-repo> "<card text>"
#   NGPU=4 GPU_TYPE="NVIDIA H200" bash difficult_advice_launch.sh ...      # bigger box
#   EPOCHS=10 bash difficult_advice_launch.sh ...                          # passes over the data (default 1)
#   PUBLISH=0 bash difficult_advice_launch.sh ...                          # keep adapters on the pod only
#   DIFFICULT_ADVICE_VAL=path/to/val.jsonl bash difficult_advice_launch.sh ...   # also stage a val set (eval_loss per epoch)
#   KEEP_POD=1 bash difficult_advice_launch.sh ...                         # pod stays up
#   STORIES_REPO=<stories adapter> bash difficult_advice_launch.sh ...     # stack: stories -> instruct mix -> difficult advice
#   BASE_MODEL=Qwen/Qwen2.5-72B REPO_PREFIX=Qwen2.5-72B bash difficult_advice_launch.sh ...   # 72B (needs a 450 GB disk pod)
#
# Defaults to 2xA100-SXM: the difficult-advice stage is a few dozen optimizer
# steps, so the box is sized for the CPU merge and the 62GB download, not for
# throughput.
set -euo pipefail
NAME="${1:?pod name}"; RUN="${2:?run}"; DATA="${3:?difficult-advice jsonl}"
INSTRUCT_MIX_REPO="${4:?instruct-mix adapter repo}"; CARD="${5:?card text}"
[ -s "$DATA" ] || { echo "missing dataset $DATA" >&2; exit 1; }
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
ENV_FILE="$REPO/.env"
NGPU="${NGPU:-2}"

RUNPOD_API_KEY=$(grep -E '^(export )?RUNPOD_API_KEY=' "$ENV_FILE" | tail -1 | sed 's/^export //; s/^RUNPOD_API_KEY=//; s/^"//; s/"$//')
: "${RUNPOD_API_KEY:?RUNPOD_API_KEY not in .env}"
export RUNPOD_API_KEY

if [ ! -f "$HERE/.pods/$NAME.env" ]; then
  CONTAINER_DISK_GB=250 bash "$HERE/create_pod.sh" "$NAME" "$NGPU" \
    --gpu-type "${GPU_TYPE:-NVIDIA A100-SXM4-80GB}"
fi
source "$HERE/.pods/$NAME.env"
[ -n "${SSH_IP:-}" ] && [ -n "${SSH_PORT:-}" ] || { echo "no SSH endpoint for $NAME" >&2; exit 1; }
SSH=(ssh -p "$SSH_PORT" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "root@$SSH_IP")
SCP=(scp -P "$SSH_PORT" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null)

KEYS_FILE="$ENV_FILE" bash "$HERE/push_to_pod.sh" "$NAME"
"${SCP[@]}" "$DATA" "root@$SSH_IP:/workspace/data/difficult_advice.jsonl"
if [ -n "${DIFFICULT_ADVICE_VAL:-}" ]; then
  [ -s "$DIFFICULT_ADVICE_VAL" ] || { echo "DIFFICULT_ADVICE_VAL not found: $DIFFICULT_ADVICE_VAL" >&2; exit 1; }
  "${SCP[@]}" "$DIFFICULT_ADVICE_VAL" "root@$SSH_IP:/workspace/data/difficult_advice_val.jsonl"
fi
# RUNPOD_POD_ID is not in the environment of a chain started over ssh: stage it.
printf 'RUNPOD_API_KEY=%s\nRUNPOD_POD_ID=%s\n' "$RUNPOD_API_KEY" "$POD_ID" \
  | "${SSH[@]}" "cat >> /root/.keys; chmod 600 /root/.keys"

"${SSH[@]}" "bash /root/fsdp/setup.sh" | tail -3 | grep -q FA3-SETUP-OK \
  || { echo "FA3 setup failed on $NAME (pod still billing)" >&2; exit 1; }
echo "FA3-SETUP-OK on $NAME"

"${SSH[@]}" "KEEP_POD=${KEEP_POD:-0} EPOCHS=${EPOCHS:-1} PUBLISH=${PUBLISH:-1} STORIES_REPO='${STORIES_REPO:-}' BASE_MODEL='${BASE_MODEL:-}' REPO_PREFIX='${REPO_PREFIX:-}' FSDP_CONFIG='${FSDP_CONFIG:-}' nohup bash /root/fsdp/difficult_advice_chain.sh '$RUN' '$INSTRUCT_MIX_REPO' '$CARD' > /root/difficult_advice-chain.log 2>&1 < /dev/null & echo chain pid \$!"
echo "chain started on $NAME ($RUN); follow with:"
echo "  ssh -p $SSH_PORT root@$SSH_IP 'tail -5 /root/difficult_advice-chain.log; ls /root/DONE-* /root/FAILED-* 2>/dev/null'"

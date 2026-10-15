#!/usr/bin/env bash
# Local driver for one story-size-ladder pod (stories_ladder_chain.sh).
# Creates the pod, stages this folder, the rung files, the held-out test set
# and the keys, runs the FA3 setup gate, optionally a 5-step smoke that
# exercises the test-loss paths, then starts stories_ladder_chain.sh under
# nohup ON THE POD (no local watcher) and returns.
# Keys come from the repo-root .env: HF_TOKEN, WANDB_API_KEY, RUNPOD_API_KEY.
#
#   bash stories_ladder_launch.sh <pod-name> <test_steps> [--smoke] <rung> [<rung> ...]
#   e.g.  bash stories_ladder_launch.sh ladder-a 0   3m 28m 56m
#         bash stories_ladder_launch.sh ladder-b 200 --smoke 112m
#
# Rung files: data/stories/training/stories-gpt-5.4-nano-ladder-<rung>.jsonl,
# test set:   data/stories/training/stories-gpt-5.4-nano-ladder-heldout.jsonl.
# Paid from the moment create_pod.sh succeeds.
set -euo pipefail
NAME="${1:?pod name}"; TEST_STEPS="${2:?test_steps}"; shift 2
SMOKE=0; if [ "${1:-}" = "--smoke" ]; then SMOKE=1; shift; fi
RUNGS=("$@"); [ ${#RUNGS[@]} -gt 0 ] || { echo "no rungs given" >&2; exit 1; }

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
STORIES_TRAIN="$REPO/data/stories/training"
ENV_FILE="$REPO/.env"
for r in "${RUNGS[@]}"; do
  [ -s "$STORIES_TRAIN/stories-gpt-5.4-nano-ladder-$r.jsonl" ] || { echo "missing rung file $r" >&2; exit 1; }
done

RUNPOD_API_KEY=$(grep -E '^(export )?RUNPOD_API_KEY=' "$ENV_FILE" | tail -1 | sed 's/^export //; s/^RUNPOD_API_KEY=//; s/^"//; s/"$//')
: "${RUNPOD_API_KEY:?RUNPOD_API_KEY not in .env}"
export RUNPOD_API_KEY

if [ ! -f "$HERE/.pods/$NAME.env" ]; then
  CONTAINER_DISK_GB=250 bash "$HERE/create_pod.sh" "$NAME" 4 --gpu-type "${GPU_TYPE:-NVIDIA H200}"
fi
source "$HERE/.pods/$NAME.env"
SSH=(ssh -p "$SSH_PORT" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "root@$SSH_IP")
SCP=(scp -P "$SSH_PORT" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null)

# This folder + instruction-tuning mix + HF/wandb keys, then the rung files.
KEYS_FILE="$ENV_FILE" bash "$HERE/push_to_pod.sh" "$NAME"
for r in "${RUNGS[@]}"; do
  "${SCP[@]}" "$STORIES_TRAIN/stories-gpt-5.4-nano-ladder-$r.jsonl" "root@$SSH_IP:/workspace/data/ladder-$r.jsonl"
done
"${SCP[@]}" "$STORIES_TRAIN/stories-gpt-5.4-nano-ladder-heldout.jsonl" "root@$SSH_IP:/workspace/data/ladder-test.jsonl"
# The chain terminates the pod itself at the end: it needs the key and the pod id.
printf 'RUNPOD_API_KEY=%s\nRUNPOD_POD_ID=%s\n' "$RUNPOD_API_KEY" "$POD_ID" | "${SSH[@]}" "cat >> /root/.keys; chmod 600 /root/.keys"

"${SSH[@]}" "bash /root/fsdp/setup.sh" | tail -3 | grep -q FA3-SETUP-OK || { echo "FA3 setup failed on $NAME" >&2; exit 1; }
echo "FA3-SETUP-OK on $NAME"

PRE='export PATH=/root/.local/bin:$PATH; cd /root/fsdp/env; set -a; . /root/.keys; set +a'
if [ "$SMOKE" = 1 ]; then
  # 5 steps, test loss at step 0, every 2 steps and at the end -> test_loss.json.
  "${SSH[@]}" "$PRE; uv run --no-sync bash /root/fsdp/launch.sh stories 4 --smoke \
      --corpus /workspace/data/ladder-${RUNGS[0]}.jsonl --out /workspace/out/smoke \
      --test-corpus /workspace/data/ladder-test.jsonl --test-steps 2 2>&1 | tail -40; \
      echo '--- test_loss.json ---'; cat /workspace/out/smoke/test_loss.json" \
    || { echo "SMOKE FAILED on $NAME (no test_loss.json) — chain NOT started; pod is still billing" >&2; exit 1; }
  "${SSH[@]}" "rm -rf /workspace/out/smoke"
  echo "smoke passed on $NAME — check the eval_loss lines above"
fi

"${SSH[@]}" "nohup bash /root/fsdp/stories_ladder_chain.sh $TEST_STEPS ${RUNGS[*]} > /root/stories_ladder-chain.log 2>&1 < /dev/null & echo chain pid \$!"
echo "chain started on $NAME (${RUNGS[*]}); follow with:"
echo "  ssh -p $SSH_PORT root@$SSH_IP 'tail -5 /root/stories_ladder-chain.log; ls /root/DONE-* /root/FAILED-* 2>/dev/null'"

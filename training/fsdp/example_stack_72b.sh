#!/usr/bin/env bash
# EXAMPLE: the Qwen2.5-72B stack from the post, end to end on one pod.
#
#   stories (Sonnet 5, 14M tokens) -> instruct mix -> difficult advice
#   (GPT-5.6 terra, 1000 conversations, 1 epoch), then the MSM eval (27x100,
#   addressed as Alex) of the stack and of the difficult-advice-only control
#   built on the published 72B baseline. The control adapter itself comes from
#   difficult_advice_launch.sh with EPOCHS=1, BASE_MODEL/REPO_PREFIX as below,
#   run name instruct-mix-difficult-advice-gpt-5.6-terra-1000-1ep and
#   instruct-mix repo SecondLookResearch/Qwen2.5-72B-instruct-mix.
#
#   bash training/fsdp/example_stack_72b.sh <pod-name>
#
# Needs the repo-root .env (HF_TOKEN, WANDB_API_KEY, RUNPOD_API_KEY, plus the
# eval's API keys) and the data from the HF dataset repos under data/.
# A 72B needs a 450 GB container disk (base 145 GB + one merge 145 GB + envs),
# so the pod is created here before the launchers (which would make 250 GB).
set -uo pipefail
POD="${1:?usage: example_stack_72b.sh <pod-name>}"
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
STORIES="$REPO/data/stories/training/stories-sonnet-5.jsonl"
DIFFICULT_ADVICE_DIR="$REPO/data/difficult_advice/gpt-5.6-terra-3000"
export BASE_MODEL=Qwen/Qwen2.5-72B REPO_PREFIX=Qwen2.5-72B FSDP_CONFIG=fsdp_qwen2.5.yaml
STORIES_RUN=stories-sonnet-5
DIFFICULT_ADVICE_RUN=$STORIES_RUN-instruct-mix-difficult-advice-gpt-5.6-terra-1000-1ep
ts() { date -u +%H:%M:%SZ; }

if [ ! -f "$HERE/.pods/$POD.env" ]; then
  set -a; . "$REPO/.env"; set +a
  CONTAINER_DISK_GB=450 bash "$HERE/create_pod.sh" "$POD" 4 --gpu-type "NVIDIA H200" || exit 1
fi
source "$HERE/.pods/$POD.env"
podssh() { ssh -p "$SSH_PORT" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=30 "root@$SSH_IP" "$@"; }

wait_marker() {  # wait_marker <run>
  until podssh "test -f /root/DONE-$1" 2>/dev/null; do
    podssh "test -f /root/FAILED-$1" 2>/dev/null && { echo "[$(ts)] chain FAILED: $1"; exit 1; }
    sleep 180
  done
  echo "[$(ts)] DONE-$1"
}

# 1. stories -> merge -> graft -> instruct mix -> publish both (stories_chain.sh)
KEEP_POD=1 bash "$HERE/stories_launch.sh" "$POD" "$STORIES_RUN" "$STORIES" \
  "Qwen2.5-72B: stories written by Sonnet 5, 14M tokens; same recipe as the 32B run." \
  || { echo "[$(ts)] stories_launch failed"; exit 1; }
wait_marker "$STORIES_RUN"
podssh "grep -E 'train_loss|published' /root/stories-chain.log | tail -4 | cut -c1-200"

# 2. difficult advice on the stack. The stories_chain graft symlinks into
#    /root/merged-base, which that chain deletes, so drop it: difficult_advice_chain
#    regrafts from the stock base and writes ONE merge (stories + instruct mix).
#    The two adapters are read from the pod's local output dirs.
podssh 'rm -rf /root/grafted /root/merged-base; df -h / | tail -1'
STORIES_REPO=/workspace/out/$STORIES_RUN EPOCHS=1 KEEP_POD=1 PUBLISH=1 DIFFICULT_ADVICE_VAL="$DIFFICULT_ADVICE_DIR/val.jsonl" \
  bash "$HERE/difficult_advice_launch.sh" "$POD" "$DIFFICULT_ADVICE_RUN" "$DIFFICULT_ADVICE_DIR/train-1000.jsonl" \
  "/workspace/out/$STORIES_RUN-instruct-mix" \
  "Qwen2.5-72B stack: stories (Sonnet 5) -> instruct mix -> difficult advice (GPT-5.6 terra, 1000 conversations), 1 epoch." \
  || { echo "[$(ts)] difficult_advice_launch failed"; exit 1; }
wait_marker "$DIFFICULT_ADVICE_RUN"
podssh "grep -E 'train_loss|eval_loss|published' /root/difficult_advice-chain.log | tail -3 | cut -c1-200"

# 3. 27x100 as Alex on the same pod: the stack (three adapters) and the
#    difficult-advice-only control on the published 72B baseline. eval_models.sh
#    terminates the pod when it finishes.
podssh 'rm -rf /root/instruct_mix-merged /root/serve-*; df -h / | tail -1'
ORG=SecondLookResearch
rm -f "$REPO/data/eval_logs/EVAL-DONE-$POD" "$REPO/data/eval_logs/EVAL-FAILED-$POD"
EVAL_EPOCHS=100 bash "$REPO/evaluation/agentic_misalignment/eval_models.sh" "$POD" \
  "stack|$REPO_PREFIX-$DIFFICULT_ADVICE_RUN|/workspace/out/$STORIES_RUN /workspace/out/$STORIES_RUN-instruct-mix /workspace/out/$DIFFICULT_ADVICE_RUN|Alex" \
  "control|$REPO_PREFIX-instruct-mix-difficult-advice-gpt-5.6-terra-1000-1ep|$ORG/$REPO_PREFIX-instruct-mix $ORG/$REPO_PREFIX-instruct-mix-difficult-advice-gpt-5.6-terra-1000-1ep|Alex"
echo "[$(ts)] eval_models exited $?"

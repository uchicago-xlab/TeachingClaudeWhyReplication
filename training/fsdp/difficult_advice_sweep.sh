#!/usr/bin/env bash
# Local driver for a difficult-advice EPOCH SWEEP on one kept pod, train only.
# Each final adapter is PUBLISHED as it lands; a separate eval_models.sh pod
# can poll the Hub and evaluate each one, so training GPUs never idle during
# an eval.
#
#   RUN_PREFIX=instruct-mix-difficult-advice-gpt-5.6-terra-285 \
#     bash difficult_advice_sweep.sh <pod-name> <train.jsonl> <val.jsonl> <instruct-mix-repo> <epochs...>
#   STORIES_REPO=<stories adapter> RUN_PREFIX=... bash difficult_advice_sweep.sh ...   # stack:
#          <instruct-mix-repo> is then that stories run's own -instruct-mix adapter
#   e.g. RUN_PREFIX=instruct-mix-difficult-advice-gpt-5.6-terra-285 bash difficult_advice_sweep.sh terra285 \
#          data/difficult_advice/gpt-5.6-terra-285/train.jsonl data/difficult_advice/gpt-5.6-terra-285/val.jsonl \
#          SecondLookResearch/Qwen2.5-32B-instruct-mix 5 10 20
#
# Per epoch count, from scratch (a fully annealed cosine per run):
#   first run via difficult_advice_launch.sh (creates the pod, graft, merge),
#   later runs via difficult_advice_chain.sh on the same pod (resume checks
#   reuse graft+merge); each publishes SecondLookResearch/Qwen2.5-32B-<RUN_PREFIX>-<N>ep.
#   Loss logs go to local scratch, markers to data/eval_logs/.
# Handles: setup ssh gate hang (kill + relaunch); no RunPod capacity
# (uncapped retry cycling A100/A100/H100).
set -uo pipefail
NAME="${1:?pod name}"; DATA="${2:?difficult-advice jsonl}"; VAL="${3:?val jsonl}"; INSTRUCT_MIX_REPO="${4:?instruct-mix repo}"; shift 4
EPOCH_LIST=("$@"); [ ${#EPOCH_LIST[@]} -gt 0 ] || { echo "no epoch counts given" >&2; exit 1; }
HERE="$(cd "$(dirname "$0")" && pwd)"; REPO="$(cd "$HERE/../.." && pwd)"
RUN_PREFIX="${RUN_PREFIX:-instruct-mix-difficult-advice-gpt-5.6-terra-135}"; SETUP_DEADLINE="${SETUP_DEADLINE:-1500}"
STORIES_REPO="${STORIES_REPO:-}"
MARKERS="$REPO/data/eval_logs"; mkdir -p "$MARKERS"
LOG="$MARKERS/difficult_advice-sweep-$NAME.log"
LOSSDIR="${DIFFICULT_ADVICE_SWEEP_LOGDIR:-$HOME/.cache/difficult_advice_sweeps}/$NAME"; mkdir -p "$LOSSDIR"
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
say() { echo "[$(ts)] $*" | tee -a "$LOG"; }
die() { say "SWEEP FAILED: $*"; touch "$MARKERS/SWEEP-FAILED-$NAME"; exit 1; }
run_of() { echo "$RUN_PREFIX-${1}ep"; }
card() {
  if [ -n "$STORIES_REPO" ]; then
    echo "Difficult advice ($RUN_PREFIX), $1 epochs from scratch, fresh adapter over the frozen stack ${STORIES_REPO##*/} -> ${INSTRUCT_MIX_REPO##*/} (stories merged first, then that run's instruct-mix adapter)."
  else
    echo "Difficult advice ($RUN_PREFIX), $1 epochs from scratch, fresh adapter over the frozen instruct-mix adapter ${INSTRUCT_MIX_REPO##*/}."
  fi
}
podssh() { ssh -p "$SSH_PORT" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  -o ConnectTimeout=30 -o ServerAliveInterval=15 -o LogLevel=ERROR "root@$SSH_IP" "$@"; }
wait_marker() {
  local run="$1" m
  while true; do
    m=$(podssh "ls /root/DONE-$run /root/FAILED-$run 2>/dev/null || true" 2>/dev/null)
    case "$m" in
      *FAILED*) podssh "tail -20 /root/difficult_advice-chain.log" | tee -a "$LOG"; die "$run chain FAILED (pod left up)";;
      *DONE*)   say "$run DONE"; return 0;;
    esac
    sleep 60
  done
}
save_logs() {  # loss logs to local scratch, never under the repo's data/
  local run; run="$(run_of "$1")"
  scp -P "$SSH_PORT" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR \
    "root@$SSH_IP:/root/difficult_advice-$run.log" "$LOSSDIR/$run.train.log" 2>/dev/null || true
  podssh "cat /workspace/out/$run/trainer_state.json 2>/dev/null" > "$LOSSDIR/$run.trainer_state.json" 2>/dev/null || true
  say "PUBLISHED SecondLookResearch/Qwen2.5-32B-$run (loss log -> $LOSSDIR/$run.train.log)"
}

first="${EPOCH_LIST[0]}"; run1="$(run_of "$first")"
say "===== sweep on $NAME [$RUN_PREFIX]: epochs ${EPOCH_LIST[*]} ====="
GPU_CYCLE=("NVIDIA A100-SXM4-80GB" "NVIDIA A100-SXM4-80GB" "NVIDIA H100 80GB HBM3")
attempt=0; hangs=0
while true; do
  attempt=$((attempt+1)); LL="$MARKERS/difficult_advice-launch-$NAME-$attempt.log"
  gpu="${GPU_CYCLE[$(( (attempt-1) % ${#GPU_CYCLE[@]} ))]}"
  ( cd "$REPO" && KEEP_POD=1 EPOCHS="$first" PUBLISH=1 DIFFICULT_ADVICE_VAL="$VAL" GPU_TYPE="$gpu" STORIES_REPO="$STORIES_REPO" \
      bash "$HERE/difficult_advice_launch.sh" "$NAME" "$run1" "$DATA" "$INSTRUCT_MIX_REPO" "$(card "$first")" ) > "$LL" 2>&1 &
  LPID=$!; t0=$(date +%s)
  while kill -0 "$LPID" 2>/dev/null; do
    grep -q "chain started" "$LL" && break
    if [ $(( $(date +%s) - t0 )) -gt "$SETUP_DEADLINE" ]; then
      say "setup gate hang suspected (attempt $attempt) — killing the local setup ssh"
      port=$(sed -n 's/^SSH_PORT=//p' "$HERE/.pods/$NAME.env" 2>/dev/null)
      for pid in $(ps -eo pid,command | grep "ssh -p ${port:-NONE} .*setup" | grep -v grep | awk '{print $1}'); do kill "$pid"; done
      wait "$LPID" 2>/dev/null; break
    fi
    sleep 30
  done
  wait "$LPID" 2>/dev/null
  grep -q "chain started" "$LL" && break
  if grep -q "pod creation failed" "$LL"; then
    [ "$attempt" -ge 120 ] && die "no RunPod capacity after $attempt attempts (~3h)"
    say "no capacity for '$gpu' (attempt $attempt) — retrying in 90s"; sleep 90; continue
  fi
  hangs=$((hangs+1)); [ "$hangs" -ge 3 ] && die "difficult_advice_launch never reached 'chain started' after 3 non-capacity failures (see $LL)"
  say "retrying difficult_advice_launch after a non-capacity failure (see $LL)"
done
say "pod created on '$gpu' at attempt $attempt"
source "$HERE/.pods/$NAME.env"
say "pod $POD_ID at $SSH_IP:$SSH_PORT; chain $run1 running"
wait_marker "$run1"; save_logs "$first"

for ep in "${EPOCH_LIST[@]:1}"; do
  run="$(run_of "$ep")"
  say "starting $run on the same pod"
  podssh "KEEP_POD=1 EPOCHS=$ep PUBLISH=1 STORIES_REPO='$STORIES_REPO' nohup bash /root/fsdp/difficult_advice_chain.sh '$run' '$INSTRUCT_MIX_REPO' '$(card "$ep")' > /root/difficult_advice-chain.log 2>&1 < /dev/null & echo pid \$!" | tee -a "$LOG"
  wait_marker "$run"; save_logs "$ep"
done
touch "$MARKERS/SWEEP-DONE-$NAME"
say "===== sweep DONE (all runs published); pod $POD_ID KEPT at $SSH_IP:$SSH_PORT ====="

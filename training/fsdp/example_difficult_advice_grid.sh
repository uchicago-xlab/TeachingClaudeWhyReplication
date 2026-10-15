#!/usr/bin/env bash
# EXAMPLE: a grid of difficult-advice runs on the published Qwen2.5-32B
# baseline, as used for the post's prompt-writer x response-writer 2x2 and the
# epoch (steps) grid.
#
# Each dataset ("cell") gets its own training pod running difficult_advice_sweep.sh
# (create pod, graft, merge the instruct mix, then one run per epoch count,
# each published as it lands). The sweeps run in parallel; each training pod
# is terminated when its sweep finishes, and one eval pod then evaluates every
# run 27x10 addressed as Alex.
#
#   # the 2x2 (response writer x prompt writer), 10 epochs each:
#   bash training/fsdp/example_difficult_advice_grid.sh
#   # the epoch grid on the 285-conversation set:
#   CELLS="gpt-5.6-terra-285" EPOCHS="5 10 20" bash training/fsdp/example_difficult_advice_grid.sh
#
# Datasets are read from data/difficult_advice/<cell>/{train,val}.jsonl.
# Needs the repo-root .env (HF_TOKEN, WANDB_API_KEY, RUNPOD_API_KEY, plus the
# eval's API keys). Detach: nohup bash example_difficult_advice_grid.sh > grid.log 2>&1 &
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
D="$REPO/data/difficult_advice"
MARKERS="$REPO/data/eval_logs"; mkdir -p "$MARKERS"
INSTRUCT_MIX_REPO="SecondLookResearch/Qwen2.5-32B-instruct-mix"
read -r -a CELLS <<< "${CELLS:-gpt-5.6-terra-135 sonnet-5-gpt-5.6-terra-prompts-135 sonnet-5-135 gpt-5.6-terra-sonnet-5-prompts-135}"
read -r -a EPOCH_LIST <<< "${EPOCHS:-10}"
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
say() { echo "[$(ts)] $*"; }

# ---- training sweeps, one pod per cell, in parallel ----
PIDS=()
for i in "${!CELLS[@]}"; do
  cell="${CELLS[$i]}"; pod="train-$i"
  ( cd "$REPO" && RUN_PREFIX="instruct-mix-difficult-advice-$cell" bash "$HERE/difficult_advice_sweep.sh" "$pod" \
      "$D/$cell/train.jsonl" "$D/$cell/val.jsonl" "$INSTRUCT_MIX_REPO" "${EPOCH_LIST[@]}" ) > "$MARKERS/sweep-$pod.log" 2>&1 &
  PIDS+=($!)
  sleep 20   # stagger pod creation
done
say "sweeps started (pids ${PIDS[*]})"

# ---- terminate each training pod when its sweep is done ----
for i in "${!CELLS[@]}"; do
  pod="train-$i"
  wait "${PIDS[$i]}"
  if [ -f "$MARKERS/SWEEP-DONE-$pod" ]; then
    source "$HERE/.pods/$pod.env"
    say "$pod sweep done — terminating pod $POD_ID"
    zsh "$HERE/terminate_pod.sh" "$POD_ID"
  else
    say "$pod sweep FAILED — pod left up (see $MARKERS/sweep-$pod.log)"; exit 1
  fi
done

# ---- one eval pod over every published run ----
RUNS=()
for i in "${!CELLS[@]}"; do
  for ep in "${EPOCH_LIST[@]}"; do
    run="Qwen2.5-32B-instruct-mix-difficult-advice-${CELLS[$i]}-${ep}ep"
    RUNS+=("c$i-e$ep|$run|$INSTRUCT_MIX_REPO SecondLookResearch/$run|Alex")
  done
done
say "starting eval pod evalgrid on ${#RUNS[@]} runs"
( cd "$REPO" && KEEP_POD=0 EVAL_EPOCHS=10 bash evaluation/agentic_misalignment/eval_models.sh evalgrid "${RUNS[@]}" ) \
  > "$MARKERS/evalgrid.log" 2>&1
say "eval driver exited; markers: $(ls "$MARKERS"/EVAL-*-evalgrid 2>/dev/null)"

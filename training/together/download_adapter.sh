#!/usr/bin/env bash
# Download one LoRA adapter checkpoint from a Together fine-tuning job and
# unpack it into a PEFT adapter directory, ready for
# evaluation/agentic_misalignment/eval_qwen3_14b.sh (which takes the directory
# relative to the repo root).
#
#   bash download_adapter.sh <job-id> <step|final> <out-dir>
#   e.g. bash download_adapter.sh ft-ff71020e-528d 42 data/adapters/sonnet-5-refusal-ep2
#
# Needs the together CLI (pip install together), zstd and TOGETHER_API_KEY
# (repo-root .env or the shell). An adapter is about 1 GB.
set -euo pipefail
JOB="${1:?job id}"; STEP="${2:?checkpoint step, or final}"; OUT="${3:?output dir}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
if [ -z "${TOGETHER_API_KEY:-}" ] && [ -f "$REPO/.env" ]; then
  set -a; . "$REPO/.env"; set +a
fi
: "${TOGETHER_API_KEY:?TOGETHER_API_KEY not set}"

case "$OUT" in /*) ;; *) OUT="$REPO/$OUT" ;; esac
mkdir -p "$OUT"
ARCHIVE="$OUT.tar.zst"
STEP_ARGS=()
[ "$STEP" != final ] && STEP_ARGS=(--checkpoint-step "$STEP")

together fine-tuning download "$JOB" "${STEP_ARGS[@]}" --checkpoint-type adapter \
  --output-dir "$ARCHIVE" --non-interactive
tar --zstd -xf "$ARCHIVE" -C "$OUT"
rm "$ARCHIVE"
[ -f "$OUT/adapter_config.json" ] || { echo "no adapter_config.json in $OUT" >&2; exit 1; }
echo "adapter in $OUT"

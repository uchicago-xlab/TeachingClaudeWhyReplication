#!/usr/bin/env bash
# Topology-aware launcher: owns the gradient-accumulation math for every
# stage, so the effective batch is 8 packed sequences on any GPU count.
#
#   bash launch.sh stories          8 --corpus /workspace/data/stories.jsonl --out ... [--smoke]
#   bash launch.sh instruct_mix     8 --base /root/grafted --out /workspace/out/instruct_mix [--smoke] [--attn sdpa]
#   bash launch.sh difficult_advice 8 --data /workspace/data/difficult_advice.jsonl --out ...
# stories trains with next-token loss on documents (train_stories.py); the two
# chat stages train with loss on assistant turns only (train_chat.py).
#   FSDP_CONFIG=<yaml> bash launch.sh ...   # another model family: its decoder-layer wrap class
#   (BASE_MODEL=<repo> is read by the trainers as the --base default.)
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
usage="usage: launch.sh stories|instruct_mix|difficult_advice <n_gpus> [script args...]"
stage="${1:?$usage}"
ngpu="${2:?$usage}"
shift 2

case "$stage" in
  stories) trainer=train_stories.py ;;
  instruct_mix|difficult_advice) trainer=train_chat.py ;;
  *) echo "unknown stage '$stage' ($usage)" >&2; exit 1 ;;
esac

# Effective batch stays 8 sequences on any topology.
if (( 8 % ngpu != 0 )); then
  echo "n_gpus=$ngpu does not divide effective batch 8" >&2; exit 1
fi
accum=$(( 8 / ngpu ))

# Containerized pods: NVLS multicast is broken; NCCL dies at the first
# barrier without this.
export NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec accelerate launch \
  --config_file "$here/${FSDP_CONFIG:-fsdp_qwen2.5.yaml}" \
  --num_processes "$ngpu" \
  "$here/$trainer" --accum "$accum" "$@"

#!/usr/bin/env bash
# One-shot environment setup for the training code on a fresh pod.
# Builds the uv env from env/uv.lock (torch cu13, transformers 5.14+, TRL,
# FA3 compiled from the flash-attention hopper tree).
# Ends with FA3-SETUP-OK or dies loudly. Rerunnable; ~45 min the first time
# (FA3 compile dominates), seconds when the env already exists.
set -euo pipefail
cd /root/fsdp/env

# nvcc 13 for the FA3 build — cuda-toolkit-13-3, NEVER the 'cuda' meta
# package (it would drag in a driver; the container must keep the host's).
if [ ! -x /usr/local/cuda-13.3/bin/nvcc ]; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq
  if ! apt-get install -y -qq --no-install-recommends cuda-toolkit-13-3; then
    # image without the NVIDIA apt repo: add the keyring, retry
    curl -fsSLO https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
    dpkg -i cuda-keyring_1.1-1_all.deb
    apt-get update -qq
    apt-get install -y -qq --no-install-recommends cuda-toolkit-13-3
  fi
fi
export PATH=/usr/local/cuda-13.3/bin:$PATH
export CUDA_HOME=/usr/local/cuda-13.3

command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# FA3 build: large temp footprint -> /workspace (root disk fills otherwise);
# cap ninja parallelism so the compile doesn't OOM system RAM.
mkdir -p /workspace/tmp
export TMPDIR=/workspace/tmp
export MAX_JOBS="${MAX_JOBS:-16}"
# The pyproject sets no-build-isolation, so FA3 builds inside the project
# venv — which must therefore already hold FA3's build-time imports:
# setuptools (fresh uv venvs lack it) AND torch (FA3's setup.py imports it).
# Hence the two-phase sync: everything except FA3 first (installs the locked
# cu13 torch), then FA3 against it.
uv venv --allow-existing
uv pip install -q setuptools ninja packaging einops
uv sync --group train --no-install-package flash-attn-3
# FA3 is built from source, pinned in env/uv.lock (~45 min on the first setup
# of a pod). Every later `uv run` must be `uv run --no-sync`, or uv re-syncs
# to the lock and rebuilds FA3.
uv sync --group train

# Import gate — fail HERE, not at training step 0.
uv run --no-sync python - <<'PY'
import torch, transformers, trl, flash_attn_interface  # noqa: F401
print("torch", torch.__version__, "| cuda", torch.version.cuda)
print("transformers", transformers.__version__, "| trl", trl.__version__)
print("fa3 import ok |", torch.cuda.device_count(), "GPUs visible")
PY
echo FA3-SETUP-OK

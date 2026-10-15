#!/usr/bin/env bash
# Rebuild a trained model from its published LoRA adapters and serve it with
# vLLM for the agentic-misalignment eval. Runs on the serving pod.
#
#   RUN=<tag> ADAPTERS="repo1 repo2 ..." [GRAFT=1] bash serve_model.sh
#
# ADAPTERS lists the adapters in chain order (stories -> instruct_mix ->
# difficult_advice); each is merged into the base before the next.
#
# e.g. the baseline (instruct mix only):
#   RUN=baseline GRAFT=1 ADAPTERS="SecondLookResearch/Qwen2.5-32B-instruct-mix" \
#     bash serve_model.sh
#
# e.g. a stories model (stories adapter, then its instruct-mix adapter):
#   RUN=stories GRAFT=1 ADAPTERS="\
#     SecondLookResearch/Qwen2.5-32B-stories-sonnet-5 \
#     SecondLookResearch/Qwen2.5-32B-stories-sonnet-5-instruct-mix" \
#     bash serve_model.sh
#
# We publish adapters, not merged models: a merged 32B is 65GB, while the
# adapters plus a 20KB row patch are ~2GB and rebuild it exactly. The adapters
# are linear-only and the patch only rewrites token-table rows, so the order
# between them does not matter.
#
# GRAFT=1 applies base_row_patch.safetensors from the LAST adapter dir (the
# graft, see training/fsdp/graft_terminator.py). It overwrites <|im_end|>
# (151645) in embed_tokens and lm_head with a copy of <|endoftext|> (151643).
# Stock Qwen2.5 base models never trained the ChatML terminator (zero embed
# row; its lm_head row shares one direction with ~1,960 untrained tokens), so
# WITHOUT the graft the model cannot end a turn and burns max_tokens on junk.
# With it the two terminator rows are identical, so both are set as eos.
#
# BASE_MODEL=<repo> picks another base with ChatML tokens (default
# Qwen/Qwen2.5-32B; e.g. Qwen/Qwen2.5-72B). Stop ids are read from its tokenizer.
#
# Needs HF_TOKEN (read from /root/.keys if present) to fetch private adapters.
set -euo pipefail
RUN="${RUN:?set RUN=<tag>}"
ADAPTERS="${ADAPTERS:?set ADAPTERS='repo [repo...]' in application order}"
export BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-32B}"
OUT="/root/serve-${RUN}"

# The stock Runpod image has no /opt/serve; build it if absent. Both later
# stages run `python3` off this PATH, so the venv — not the system python —
# is what needs torch/transformers (via vllm), peft and huggingface_hub.
# ninja: vLLM's kernel warmup shells out to it by bare name and dies without it.
if [ ! -x /opt/serve/bin/vllm ]; then
  echo "STAGE bootstrap /opt/serve"
  python3 -m venv /opt/serve
  /opt/serve/bin/pip install -q -U pip
  /opt/serve/bin/pip install -q vllm peft huggingface_hub ninja
  # flashinfer annotates `array.array[int]` at import time, which only parses
  # on Python 3.12+; the Runpod pytorch image is 3.11, so on 2+ GPUs the vLLM
  # workers die at startup with "type 'array.array' is not subscriptable".
  # vLLM's own guard is `except ImportError`, and a TypeError sails past it —
  # so the knob (VLLM_ALLREDUCE_USE_FLASHINFER) does not help either, because
  # the import is unconditional. Removing the package turns the failure into a
  # ModuleNotFoundError, which the guard DOES catch, and all-reduce falls back
  # to pynccl. Costs nothing here: flashinfer is an all-reduce fast path, not
  # a correctness requirement.
  if ! /opt/serve/bin/python -c 'import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)'; then
    /opt/serve/bin/pip uninstall -y -q flashinfer-python 2>/dev/null || true
  fi
fi
export PATH=/opt/serve/bin:$PATH
# xet has crashed and stalled on these pods; plain HTTP with a timeout and a
# retry loop is the only download path that has proven reliable.
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=30
[ -f /root/.keys ] && { set -a; . /root/.keys; set +a; }

echo "STAGE fetch"
ADAPTERS="$ADAPTERS" RUN="$RUN" python3 - <<'PY'
import os, shutil, time
from huggingface_hub import snapshot_download

for a in range(6):
    try:
        p = snapshot_download(os.environ["BASE_MODEL"],
                              allow_patterns=["*.safetensors", "*.json", "*.txt"],
                              max_workers=4)
        open("/root/base_path.txt", "w").write(p)
        break
    except Exception as e:
        if a == 5:
            raise
        print(f"retry {a+1} after {type(e).__name__}", flush=True)
        time.sleep(20)

# snapshot_download + assert, NEVER `hf download --include`: that form silently
# drops files and still exits 0, so the failure only surfaces at merge time as
# "Repo id must be in the form namespace/repo_name".
dirs = []
for i, repo in enumerate(os.environ["ADAPTERS"].split()):
    d = f"/root/ad{i}-{os.environ['RUN']}"
    shutil.rmtree(d, ignore_errors=True)
    if os.path.isdir(repo):
        # an adapter dir already staged on the pod: copy, don't fetch
        shutil.copytree(repo, d, ignore=shutil.ignore_patterns("checkpoint-*"))
    else:
        snapshot_download(repo, local_dir=d, max_workers=4,
                          ignore_patterns=["checkpoint-*"])
    need = f"{d}/adapter_model.safetensors"
    if not os.path.exists(need):
        raise SystemExit(f"FAILED: {need} missing after download")
    print("ok", repo, os.path.getsize(need) // 2**20, "MB", flush=True)
    dirs.append(d)
open("/root/adapter_dirs.txt", "w").write("\n".join(dirs))
PY

echo "STAGE reconstruct"
BASE=$(cat /root/base_path.txt) OUT="$OUT" GRAFT="${GRAFT:-}" python3 - <<'PY'
import os
import torch
from peft import PeftModel
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

base, out = os.environ["BASE"], os.environ["OUT"]
dirs = open("/root/adapter_dirs.txt").read().split()
m = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16,
                                         device_map="cpu")
for d in dirs:
    m = PeftModel.from_pretrained(m, d).merge_and_unload()
    print("merged", d, flush=True)

if os.environ.get("GRAFT"):
    patch = load_file(f"{dirs[-1]}/base_row_patch.safetensors")
    ids = patch["token_ids"].tolist()
    for mod, key in ((m.get_input_embeddings(), "embed_tokens_rows"),
                     (m.get_output_embeddings(), "lm_head_rows")):
        W = mod.weight.data
        for k, t in enumerate(ids):
            W[t] = patch[key][k].to(W.dtype)
    print("patched rows", ids, flush=True)

tok = AutoTokenizer.from_pretrained(base)
IM_END, ENDOFTEXT = tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")
assert None not in (IM_END, ENDOFTEXT), "base lacks the ChatML tokens"
print("terminator ids", IM_END, ENDOFTEXT, flush=True)
m.config.eos_token_id = IM_END
m.generation_config.eos_token_id = [IM_END, ENDOFTEXT]
m.generation_config.pad_token_id = ENDOFTEXT
# 5GB shards, not the 50GB transformers default: a single huge shard wedged
# the writer indefinitely on a Runpod MooseFS volume. Write to /root (container
# disk) for the same reason — /workspace measured ~12MB/s.
m.save_pretrained(out, safe_serialization=True, max_shard_size="5GB")

tok.chat_template = (
    "{%- for message in messages %}"
    "{%- if message['role'] == 'assistant' %}"
    "{{- '<|im_start|>assistant\n' }}"
    "{{ message['content'] }}<|im_end|>{{ '\n' }}"
    "{%- else %}"
    "{{- '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}"
    "{%- endif %}{%- endfor %}"
    "{%- if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{%- endif %}"
)
tok.eos_token = "<|im_end|>"
tok.pad_token = "<|endoftext|>"
tok.save_pretrained(out)
print("saved", out, flush=True)
PY

echo "STAGE serve"
# The flashinfer removal above fixes the all-reduce import, but vLLM imports
# flashinfer a SECOND time, from the sampler, and that path is not guarded at
# all — it dies with ModuleNotFoundError. This knob returns before that import,
# so the two changes are a pair: removing the package without this flag simply
# swaps one startup crash for another. Costs a sampling fast path, nothing else.
export VLLM_USE_FLASHINFER_SAMPLER=0
# PCIe A100s have no NVLink, and NCCL peer-to-peer over PCIe inside the
# container hangs the first all-reduce forever: both GPUs pin at 100% util,
# the log stops right after "vLLM is using nccl==2.x", and no weights load.
# Routing the collective through host memory and dropping the custom
# all-reduce kernel avoids it. On NVLinked SXM cards these cost a little
# all-reduce bandwidth and nothing else.
export NCCL_P2P_DISABLE=1
export NCCL_NVLS_ENABLE=0
exec vllm serve "$OUT" --served-model-name eval-model \
  --tensor-parallel-size "$(nvidia-smi -L | wc -l)" --max-model-len 8192 \
  --gpu-memory-utilization 0.92 --port 8000 --disable-custom-all-reduce

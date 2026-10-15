# training/fsdp — Qwen2.5 32B/72B LoRA training

TRL + accelerate FSDP + FlashAttention-3 on RunPod. This folder trains every
Qwen2.5 model in the post. Each model is a chain of stages, always in this order:

1. **stories** — continued pretraining on fictional stories (synthetic document
   finetuning, loss on every token). `train_stories.py`.
2. **instruct_mix** — chat SFT on the 13K instruction-tuning mix (loss on
   assistant turns only). `train_chat.py`.
3. **difficult_advice** — chat SFT on difficult-advice conversations. `train_chat.py`.

Every model gets the instruct mix. The baseline is the instruct mix alone; the
stack is all three stages. Each stage trains a fresh LoRA adapter on top of the
previous stages merged into the base. Only adapters are published, never merged
models; `evaluation/agentic_misalignment/serve_model.sh` rebuilds a model from its
adapters for evaluation.

## The terminator graft

Qwen2.5 base models never trained the ChatML end-of-turn token `<|im_end|>`: its
embedding row is zero and its output row is tied to a cluster of other untrained
tokens. A model fine-tuned on top cannot reliably end a turn. `graft_terminator.py`
copies the `<|endoftext|>` rows onto `<|im_end|>` in both token tables before the
chat stages. Training is linear-only, so the token tables stay frozen and the
graft is the whole repair. Every Qwen2.5 base needs it (`check_graft_needed.py
<hf-repo>` checks a base from the Hub without downloading it).

The graft also writes `base_row_patch.safetensors` (one row per table). Every
published chat adapter ships this file, and serving with `GRAFT=1` applies it to
the stock base. `artifacts/base_row_patch-graft.safetensors` is the patch for
stock Qwen2.5-32B.

## Recipe

| | stories | instruct_mix | difficult_advice |
|---|---|---|---|
| loss | every token | assistant turns | assistant turns |
| cutoff | 4096 | 8192 | 8192 |
| epochs | 2 | 2 | 1 (`EPOCHS`; the 2x2 and step grids in Appendices B.3 and C.2 used 5-20) |
| warmup | 3% | 3% | 5%, at least 2 steps |

Shared by all stages: LoRA r64, alpha 128, dropout 0 on all linear layers
(q, k, v, o, gate, up, down); lr 1e-4 cosine; effective batch 8 packed
sequences (`launch.sh` sets gradient accumulation to 8 / n_gpus); bf16 with
gradient checkpointing.

## Scripts

Core:

- `launch.sh <stories|instruct_mix|difficult_advice> <n_gpus> [args]` — runs the
  right trainer under accelerate with `fsdp_qwen2.5.yaml`.
- `train_stories.py`, `train_chat.py` — the trainers. `--smoke` runs 5 steps on
  256 rows. `--attn sdpa` replaces FA3 if it fails.
- `graft_terminator.py`, `check_graft_needed.py`, `chatml.py` — the graft and
  the ChatML template.
- `merge_adapter.py` — merges one or more adapters into a base (`--chat` also
  bakes the ChatML tokenizer and stop tokens into the written model).

Pods (run locally; keys from the repo-root `.env`: `HF_TOKEN`, `WANDB_API_KEY`,
`RUNPOD_API_KEY`):

- `create_pod.sh`, `push_to_pod.sh`, `setup.sh`, `terminate_pod.sh` — create a
  RunPod pod, copy this folder and the data to it, build the environment, and
  delete the pod.
- `*_launch.sh` runs locally: it creates the pod, stages data and keys, runs
  `setup.sh`, and starts the matching `*_chain.sh` on the pod under nohup.
  Each chain trains, publishes the adapters to the Hub, verifies the upload, and
  terminates the pod unless `KEEP_POD=1`. It writes `/root/DONE-<run>` or
  `/root/FAILED-<run>` on the pod.

| launcher | chain | trains |
|---|---|---|
| `baseline_launch.sh` | `baseline_chain.sh` | instruct mix on the grafted stock base |
| `stories_launch.sh` | `stories_chain.sh` | stories, then instruct mix |
| `stories_ladder_launch.sh` | `stories_ladder_chain.sh` | the story-size ladder, several sizes per pod |
| `difficult_advice_launch.sh` | `difficult_advice_chain.sh` | difficult advice on a published instruct-mix adapter (`STORIES_REPO=` for the stack) |

- `difficult_advice_sweep.sh` — several difficult-advice epoch counts on one kept pod.
- `example_stack_72b.sh`, `example_difficult_advice_grid.sh` — how the post's
  72B stack and the difficult-advice grids were run, training through evaluation.

`BASE_MODEL=Qwen/Qwen2.5-72B REPO_PREFIX=Qwen2.5-72B` in front of a launcher
trains the 72B. Published names follow the chain of stages, for example
`SecondLookResearch/Qwen2.5-32B-stories-sonnet-5-instruct-mix`.

## Hardware and setup

- GPUs: H100/H200 (Hopper) for FlashAttention-3. The stories and baseline
  launchers default to 4×H200; the difficult-advice launcher defaults to
  2×A100-SXM 80GB, since that stage is short.
- Disk: 250 GB container disk for 32B. A 72B needs 450 GB (145 GB base plus one
  145 GB merge). Merges go to `/root` (container disk), not `/workspace`: large
  shard writes can hang on the RunPod network volume.
- `setup.sh` builds the uv environment from `env/uv.lock` and compiles FA3 from
  source, which takes about 45 minutes on a new pod. Always use
  `uv run --no-sync` afterwards; a plain `uv run` re-syncs and rebuilds FA3.
- `NCCL_NVLS_ENABLE=0` is set by `launch.sh`; NCCL hangs on containerized pods
  without it.

## Manual run

```bash
CONTAINER_DISK_GB=250 bash create_pod.sh my-pod 4 --gpu-type "NVIDIA H200"
KEYS_FILE=../../.env bash push_to_pod.sh my-pod
ssh ... "bash /root/fsdp/setup.sh"                    # must print FA3-SETUP-OK

# on the pod, in /root/fsdp/env:
uv run --no-sync bash ../launch.sh stories 4 --corpus /workspace/data/stories.jsonl --out /workspace/out/stories
uv run --no-sync python ../merge_adapter.py --adapter /workspace/out/stories --out /root/merged-base
uv run --no-sync python ../graft_terminator.py --base /root/merged-base --out /root/grafted --noise 0
uv run --no-sync bash ../launch.sh instruct_mix 4 --base /root/grafted --out /workspace/out/instruct_mix
```

The graft prints the cosine of the grafted rows to their donors; with
`--noise 0` both must be 1.0000.

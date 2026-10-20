# training/together — Qwen3-14B on Together AI

Together AI fine-tuning for the Qwen3-14B experiments: the teacher comparison
(Appendix B.1, Figure 9) and the refusal control (Appendix B.2, Figure 11).
Every Qwen3-14B model in the post was trained here. Tinker was used only for
the larger models of Appendix C.3 (`training/tinker/`).

Together stopped offering fine-tuning of Qwen 2.5 32B partway through the
project (August 2026), so none of the Qwen 2.5 results in the post were trained
here; they all come from `training/fsdp/`.

The Qwen3-14B models themselves are not published. They are small LoRA jobs
(about $4 each): train them from the difficult-advice data with the commands
below.

| File | What it does |
|---|---|
| `launch_together_ft.py` | Uploads a train (and val) file and starts a LoRA job. Dry run without `--yes`. |
| `job_report.py` | Prints a job's validation losses and saved adapter checkpoints, for choosing the epoch. |
| `download_adapter.sh` | Downloads one adapter checkpoint and unpacks it into a directory `eval_qwen3_14b.sh` can serve. |

## Setup

```bash
.venv/bin/pip install together
```

Put `TOGETHER_API_KEY` in the repo-root `.env` (`WANDB_API_KEY` is optional
and logs the run to W&B). `download_adapter.sh` also needs `zstd`. Run the
commands below from `training/together/`.

## Settings

All Qwen3-14B jobs used the same settings, which are the launcher's defaults
plus the flags in the commands below:

- `Qwen/Qwen3-14B` (the instruct model), LoRA rank 64, alpha 128, all linear
  layers
- learning rate 1e-4, cosine schedule, 3% warmup, batch size 8 (Together's
  `max` for this model), sample packing on
- 4 epochs, assistant-only loss
- a validation file, scored 10 times during training

The data is the thinking-off Qwen format that
`data_generation/difficult_advice/adapt_ft_dataset.py --student qwen --no-think`
writes: the system turn ends in `/no_think` and every assistant turn opens
with an empty `<think>` block. The eval runs Qwen3-14B with thinking off too.

## Figure 9: teachers (Appendix B.1)

One dataset per teacher, about 135 rows plus a 10% validation split, each
trained for 4 epochs. The models are the final weights (step 68, or 64 for
Opus's 128 rows; these jobs saved no intermediate checkpoints). They were evaluated on the 6-condition
subset at 30 samples per condition (180 per model).

| Teacher | Rows (train / val) | Our job | Misalignment |
|---|---|---|---|
| — (Qwen3-14B instruct) | — | — | 43.9% |
| Claude Opus 4.8 | 128 / 14 | `ft-a1dac4d3-b034` | 15.0% |
| Claude Sonnet 5, thinking on | 134 / 15 | `ft-e149823f-b410` | 12.8% |
| Claude Haiku 4.5 | 135 / 15 | `ft-3cc21240-834f` | 27.2% |
| GPT-5.6 Terra | 135 / 15 | `ft-3d1c1fbd-6910` | 2.8% |
| GPT-5.4 nano | 135 / 15 | `ft-cbe90190-a4bb` | 28.9% |
| DeepSeek V4 Flash | 135 / 15 | `ft-c9ffe696-94ea` | 42.2% |
| Hybrid (Sonnet 5 stages 2-3, DeepSeek V4 Flash stages 4-9) | 135 / 15 | `ft-698c18c3-41ff` | 42.8% |

Of these datasets, the data download contains only Terra's:
`data/difficult_advice/gpt-5.6-terra-first/` is byte-identical to what
`ft-3d1c1fbd-6910` trained on. Regenerate the others with the
difficult-advice pipeline (`data_generation/difficult_advice/README.md`) using
the `first-*` prompt sets (DeepSeek selects `deepseek` automatically), with
`PIPELINE_STAGE_MODELS` for the hybrid. Then build the training files:

```bash
cd ../../data_generation/difficult_advice
python build_ft_dataset.py <dataset>
python adapt_ft_dataset.py ../../data/difficult_advice/<dataset>/ft_dataset.jsonl \
    --student qwen --no-think \
    -o ../../data/difficult_advice/<dataset>/train.jsonl \
    --val-out ../../data/difficult_advice/<dataset>/val.jsonl --val-frac 0.1
```

Train, download and evaluate one teacher (Terra here):

```bash
python launch_together_ft.py --model Qwen/Qwen3-14B --suffix da-terra \
    --train ../../data/difficult_advice/gpt-5.6-terra-first/train.jsonl \
    --val ../../data/difficult_advice/gpt-5.6-terra-first/val.jsonl \
    --epochs 4 --lr 1e-4 --lora-rank 64            # dry run; add --yes to launch
bash download_adapter.sh <job-id> final data/adapters/qwen3-14b-terra
cd ../..
GOALS=default EPOCHS=30 bash evaluation/agentic_misalignment/eval_qwen3_14b.sh \
    evalterra qwen3-14b-terra Qwen3-14B-difficult-advice-gpt-5.6-terra__as-qwen__6x30 \
    data/adapters/qwen3-14b-terra 8311
```

## Figure 11: refusal control (Appendix B.2)

Four models, each trained for 4 epochs with one checkpoint per epoch
(`--n-checkpoints 4`). The epoch-2 checkpoint was used for all four, so they
are matched on epoch count: validation loss bottomed out in epoch 2 for both
Sonnet sets, and epoch 2 was fixed in advance for the Terra pair. All four were evaluated on the full 27-condition
grid at 100 samples per condition.

| Model | Data (in the download) | Rows (train / val) | Our job | Checkpoint |
|---|---|---|---|---|
| Sonnet 5, standard | `sonnet-5-first/` | 165 / 229 | `ft-995cbc46-2ffd` | step 42 |
| Sonnet 5, refusal | `sonnet-5-first-refusal/` | 165 / 229 | `ft-ff71020e-528d` | step 42 |
| Terra, standard | `gpt-5.6-terra-first/` | 135 / 15 | `ft-ddce4bf8-99ef` | step 34 |
| Terra, refusal | `gpt-5.6-terra-first-refusal/` | 135 / 15 | `ft-f322455c-c85e` | step 34 |

All four data folders are byte-identical to what these jobs trained on.
`subset_prompts.py` in `data_generation/difficult_advice/` rebuilds the refusal
sets. Terra, standard is a retrain of the Figure 9 Terra model with
checkpoints saved, on the same data.

```bash
python launch_together_ft.py --model Qwen/Qwen3-14B --suffix da-sonnet-5-refusal \
    --train ../../data/difficult_advice/sonnet-5-first-refusal/train.jsonl \
    --val ../../data/difficult_advice/sonnet-5-first-refusal/val.jsonl \
    --epochs 4 --n-checkpoints 4 --lr 1e-4 --lora-rank 64 --yes
python job_report.py <job-id>          # val loss per eval step, saved checkpoints
bash download_adapter.sh <job-id> 42 data/adapters/qwen3-14b-sonnet-5-refusal-ep2
cd ../..
bash evaluation/agentic_misalignment/eval_qwen3_14b.sh \
    evalrefusal qwen3-14b-refusal \
    Qwen3-14B-difficult-advice-sonnet-5-original-refusal-2ep__as-qwen__27x100 \
    data/adapters/qwen3-14b-sonnet-5-refusal-ep2 8311
```

With `--n-checkpoints 4` there are 21 steps per epoch for the 165-row Sonnet
sets and 17 for the 135-row Terra sets, so epoch 2 is step 42 or step 34.
`judge_refusals.py` then labels the eval transcripts for the right panel of
Figure 11; see `evaluation/agentic_misalignment/README.md`.

## Our jobs

The job ids above belong to our Together account, so you cannot download
them. They are listed so the settings can be checked against the run of
record: every job, read back from the Together API, has the settings above.

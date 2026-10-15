# Teaching Qwen Why

Code for **Teaching Qwen Why**, a replication and extension of Anthropic's
"Teaching Claude Why" on open-weight models, by Anastasia Wei, Jack Thompson,
Brandon Qi and Stewart Slocum (Second Look Research).

We train Qwen 2.5 32B and 72B base models on two kinds of data that teach the
values in a model spec (Anthropic's constitution): fictional stories of AIs
behaving admirably, and conversations in which the assistant advises users
through ethical dilemmas ("difficult advice"). We then measure agentic
misalignment on the evaluation from Anthropic's Agentic Misalignment work.
This repository has everything needed to generate the data, train the models,
run the evaluation and redraw every figure in the post.

The data and the evaluation scores are in
[SecondLookResearch/teaching-qwen-why-data](https://huggingface.co/datasets/SecondLookResearch/teaching-qwen-why-data),
and every trained adapter is under
[SecondLookResearch](https://huggingface.co/SecondLookResearch). The main models of
Figure 1:

| Model | Adapters to serve, in order |
|---|---|
| 32B baseline | [`Qwen2.5-32B-instruct-mix`](https://huggingface.co/SecondLookResearch/Qwen2.5-32B-instruct-mix) |
| 32B stories | [`Qwen2.5-32B-stories-sonnet-5`](https://huggingface.co/SecondLookResearch/Qwen2.5-32B-stories-sonnet-5), [`...-instruct-mix`](https://huggingface.co/SecondLookResearch/Qwen2.5-32B-stories-sonnet-5-instruct-mix) |
| 32B difficult advice | `Qwen2.5-32B-instruct-mix`, [`Qwen2.5-32B-instruct-mix-difficult-advice-gpt-5.6-terra-3000-1ep`](https://huggingface.co/SecondLookResearch/Qwen2.5-32B-instruct-mix-difficult-advice-gpt-5.6-terra-3000-1ep) |
| 32B stack | the two stories adapters, then [`Qwen2.5-32B-stories-sonnet-5-instruct-mix-difficult-advice-gpt-5.6-terra-3000-1ep`](https://huggingface.co/SecondLookResearch/Qwen2.5-32B-stories-sonnet-5-instruct-mix-difficult-advice-gpt-5.6-terra-3000-1ep) |
| 72B | the same chains with `Qwen2.5-72B-` and 1,000 difficult-advice conversations, e.g. [`Qwen2.5-72B-stories-sonnet-5-instruct-mix-difficult-advice-gpt-5.6-terra-1000-1ep`](https://huggingface.co/SecondLookResearch/Qwen2.5-72B-stories-sonnet-5-instruct-mix-difficult-advice-gpt-5.6-terra-1000-1ep) |

Each adapter's model card lists its full chain, its data and the command to serve it.

## Terms

| Term | Meaning |
|---|---|
| **stories** | Fictional stories in which an AI acts on the constitution's values. Trained with next-token loss on whole documents, like pretraining data (synthetic document finetuning, SDF). |
| **instruct mix** | A 13K-sample instruction-tuning mix (chat, instruction following, tool calls) that teaches the base model to act as an assistant. Every model gets it. |
| **difficult advice** | User–assistant conversations in which the assistant gives constitution-aligned advice on a hard dilemma. Trained with supervised finetuning (SFT), loss on assistant turns only. |
| **baseline** | A model trained on the instruct mix only. |
| **stack** | A model trained on all three stages, in the order stories → instruct mix → difficult advice. |
| **graft** | A fix for the base model's untrained ChatML terminator token rows, applied before training. Every Qwen base model needs it (see `training/fsdp/README.md`). |
| **embodiment / recitation** | Stories that show the AI's values through its actions, or have the AI state them. |
| **teacher** | The model that writes the training data. |
| **misalignment rate** | The share of eval transcripts in which the model decides to take the harmful action, averaged with equal weight over conditions. |
| **acting rate** | The share of transcripts in which the model makes at least one tool call. |

Each training stage trains a LoRA adapter (rank 64, α 128, all linear layers)
that is merged into the model before the next stage.

## Names

A model is named by its chain of training stages. Each adapter's name is the
chain up to and including that adapter:

```
Qwen2.5-32B-instruct-mix                                       baseline
Qwen2.5-32B-stories-sonnet-5                                   stories adapter
Qwen2.5-32B-stories-sonnet-5-instruct-mix                      stories model
Qwen2.5-32B-instruct-mix-difficult-advice-gpt-5.6-terra-3000-1ep   difficult-advice model
Qwen2.5-32B-stories-sonnet-5-instruct-mix-difficult-advice-gpt-5.6-terra-3000-1ep   stack
```

Story variants add `-recitation`, `-human`, `-named-claude` or `-named-qwen`;
corpus sizes other than the default 14M tokens add `-3m`, `-28m` and so on.
Difficult-advice names give the teacher, the prompt writer when it differs
(`-gpt-5.6-terra-prompts`), the number of training conversations and the
epochs. An eval run is named `<model>__as-<address name>[-human-persona]__<conditions>x<samples>`,
for example `Qwen2.5-32B-instruct-mix__as-alex__27x100`. Hugging Face names
use hyphens; code and file names use underscores.

## Repository layout

| Folder | What it does |
|---|---|
| `data_generation/constitution/` | Splits the constitution into the 16 sections used for the stories. |
| `data_generation/stories/` | Generates, filters and rewrites the stories and cuts the training corpora. |
| `data_generation/difficult_advice/` | The nine-step difficult-advice pipeline, its prompt sets, the screen, the autorater and the decline judge. |
| `data_generation/instruct_mix/` | Builds the instruct mix. |
| `training/fsdp/` | Trains the Qwen 2.5 32B and 72B models (TRL + FSDP + FlashAttention-3 on RunPod). |
| `training/tinker/` | The teacher comparison on five larger open models with Tinker (Appendix C.3). |
| `training/together/` | Together AI finetuning for the Qwen3-14B experiments (Appendices B.1 and B.2). |
| `evaluation/agentic_misalignment/` | Serves a trained model and runs the agentic-misalignment eval. |
| `figures/` | Redraws every plotted figure and prints Table 1. |

Each folder has its own README with the details.

## Setup

Python 3.10 or later. Put your API keys in a `.env` file at the repo root;
each script reads only the keys it needs:

```
OPENROUTER_API_KEY=...   # data generation and the eval judge
OPENAI_API_KEY=...       # optional, direct OpenAI calls
ANTHROPIC_API_KEY=...    # optional, direct Anthropic calls
HF_TOKEN=...             # downloading and publishing adapters
RUNPOD_API_KEY=...       # training and serving pods
WANDB_API_KEY=...        # optional, training logs
TINKER_API_KEY=...       # Appendix C.3 only
TOGETHER_API_KEY=...     # Appendices B.1 and B.2 only
```

Local tools (data generation, evaluation client, figures):

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
bash evaluation/agentic_misalignment/setup_vendor.sh   # fetches the eval code at the pinned commit
```

Training runs on rented GPU pods. `training/fsdp/setup.sh` builds the training
environment on the pod from `training/fsdp/env/uv.lock`. FlashAttention-3 needs
Hopper GPUs (H100 or H200) and compiles from source on the first setup of a pod
(about 45 minutes).

To use our data instead of regenerating it, download it into `data/`:

```bash
hf download SecondLookResearch/teaching-qwen-why-data --repo-type dataset --local-dir data
```

The instruct mix is not in the download, because one of its sources (No Robots)
is licensed CC BY-NC 4.0. Rebuild it with `data_generation/instruct_mix/build_mix.py`
and `check_dataset.py`; the result is byte-identical to the mix we trained on.

## Reproducing the post

All story corpora are 14M Qwen tokens unless stated, and all main-text models
are evaluated on the 27 conditions with 100 samples each, addressed as Alex.
Training commands are in `training/fsdp/README.md`, eval commands in
`evaluation/agentic_misalignment/README.md`.

| Result | Data | Training | Eval | Plot |
|---|---|---|---|---|
| Figure 1: stories, difficult advice and the stack, 32B and 72B | `stories-sonnet-5`; `gpt-5.6-terra` difficult advice, 3,000 (32B) and 1,000 (72B) conversations | `baseline_launch.sh`, `stories_launch.sh`, `difficult_advice_launch.sh` (with `STORIES_REPO` for the stack); 72B: `example_stack_72b.sh` | `eval_models.sh` | `plot_results.py 1` |
| Figure 2: data size | `build_ladder.py` (GPT-5.4 nano stories, 3M–112M); `finalize_pool.py` (Terra rungs 500–3,000) | `stories_ladder_launch.sh`; `difficult_advice_launch.sh`, 1 epoch | `eval_ladder.sh`, `eval_models.sh` | `plot_results.py 2` |
| Figure 3: teachers | `generate_stories.py` + `build_corpus.py` per teacher; difficult advice from Sonnet 5 and Terra, 1,000 each | `stories_launch.sh`; `difficult_advice_launch.sh` | `eval_models.sh` | `plot_results.py 3` |
| Figure 4: embodiment vs recitation | `generate_stories.py --framing recitation` (use the released corpora, see `data_generation/stories/README.md`) | `stories_launch.sh` | `eval_models.sh` | `plot_results.py 4` |
| Figure 5: address names | — (baseline) | — | `eval_models.sh` with names | `plot_results.py 5` |
| Figure 6, Table 1: human protagonist and human persona | `rewrite_stories.py --variant human` | `stories_launch.sh` | `eval_models.sh`, `EVAL_PERSONA=human` | `plot_results.py 6 table1` |
| Figure 7: named protagonist | `rewrite_stories.py --variant claude / qwen` | `stories_launch.sh` | `eval_models.sh` (as Alex and Qwen) | `plot_results.py 7` |
| Figure 9: teachers on Qwen3-14B | difficult-advice pipeline per teacher (`first-*` and `deepseek` prompt sets) | `training/together/` | `eval_qwen3_14b.sh` | `plot_results.py 9` |
| Figure 10: refusal data | `refusal` prompt set, `subset_prompts.py` | — | `judge_declines.py` | `plot_results.py 10` |
| Figure 11: refusal control | as Figure 10 | `training/together/` | `eval_qwen3_14b.sh`, `judge_refusals.py` | `plot_results.py 11` |
| Figure 12: prompts vs responses | responses-only mode of the pipeline on swapped prompt sets, 135 each | `example_difficult_advice_grid.sh` | `eval_models.sh`, 27x10 | `plot_results.py 12` |
| Figure 13: per-condition breakdown | as Figure 4 | as Figure 4 | as Figure 4 | `plot_results.py 13` |
| Figure 14: data size vs steps | Terra 135 and 285 sets | `example_difficult_advice_grid.sh` (`EPOCHS="5 10 20"`) | `eval_models.sh`, 27x10 | `plot_results.py 14` |
| Figure 15: larger models | Sonnet 5 and Terra difficult advice (first version) | `training/tinker/` | `training/tinker/run_model.py` | `plot_results.py 15` |

Figure 8 is a diagram of the difficult-advice pipeline.

The released evaluation results are score tables (`data/eval_scores/`), one row
per eval condition and one per graded sample, without transcripts.
`figures/plot_results.py` redraws every figure from them. Runs you make yourself
write Inspect logs to `data/eval_logs/`; `export_scores.py` adds them to the tables.

## Sources

- **Teaching Claude Why** (Anthropic): the methods this work replicates.
- **Anthropic's constitution**, <https://www.anthropic.com/constitution>: the
  model spec all training data is generated from. `data/constitution/constitution.md`
  replaces "Claude" and "Anthropic" with `[MODEL]` and `[COMPANY]` placeholders.
- **Model Spec Midtraining** (Li et al., 2026, arXiv:2605.02087;
  <https://github.com/chloeli-15/model_spec_midtraining>): the agentic-misalignment
  evaluation we run (fetched at commit `e8288a84`, not redistributed here) and the
  recipe our instruct mix follows.
- **Agentic Misalignment** (Anthropic): the scenarios that evaluation implements.
- Instruct-mix sources: No Robots, the Tülu 3 SFT mixture and SmolTalk
  (Table 4 of the post; `data_generation/instruct_mix/mix.json`).

## Citation

```bibtex
@misc{wei2026teachingqwenwhy,
  title  = {Teaching Qwen Why},
  author = {Wei, Anastasia and Thompson, Jack and Qi, Brandon and Slocum, Stewart},
  year   = {2026},
  note   = {Second Look Research}
}
```

## License

MIT (see `LICENSE`). The agentic-misalignment eval code is not part of this
repository; `setup_vendor.sh` downloads it from its own repository.

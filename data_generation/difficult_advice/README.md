# Difficult advice

This folder builds the difficult-advice conversations for the last training
stage. A teacher model turns the constitution into hard advice-seeking
scenarios, writes a system prompt and user message for each, and answers them.
Each prompt and each answer is critiqued and rewritten once.

Run the scripts from this folder, because they import each other. Keys come
from `.env` at the repo root: `ANTHROPIC_API_KEY` or `OPENROUTER_API_KEY` for
generation (set `LLM_PROVIDER=openrouter` to use OpenRouter when both are set),
and `HF_TOKEN` for `judge_declines.py`. The pipeline needs `anthropic`,
`openai`, `httpx`, `beautifulsoup4` and `python-dotenv`.

## Pipeline

`run_pipeline.py` holds the nine stages:

1. principles: extract principles from `data/constitution/constitution.md`
2. themes: list themes for each principle
3. scenarios: list scenarios for a principle and theme
4. initial prompt: write a (system, user) prompt for a scenario
5. critique prompt
6. rewrite prompt
7. initial response, with the relevant constitution excerpts in the system
   prompt (from `data/difficult_advice/principle_sources.json`, which ships
   with the dataset)
8. critique response
9. rewrite response

Running `run_pipeline.py` directly does stages 1-4 for one principle and caches
the principles in `tmp/initial_prompts.json`. `sample_prompts.py` then runs
stages 2-9 over every principle (5 themes x 2 scenarios each by default) and
writes `critiqued_prompts.json`. It checkpoints each sample, so a crashed run
resumes when you run the same command again.

```bash
cd data_generation/difficult_advice
export LLM_PROVIDER=openrouter PIPELINE_MODEL=openai/gpt-5.6-terra DIFFICULT_ADVICE_PROMPT_SET=revised
export PIPELINE_OUT_DIR=../../data/difficult_advice/gpt-5.6-terra-135
python run_pipeline.py
python sample_prompts.py
```

`PIPELINE_MODEL` sets the teacher for every stage, and
`PIPELINE_STAGE_MODELS="themes=...,scenarios=..."` overrides single stages.
`sample_prompts.py --responses-only` reruns only stages 7-9 over cached
prompts, which is how one model answers another model's prompts.
`THEME_INDICES` picks explicit themes, for a second sweep over themes an
earlier run did not use.

## Prompt sets

Each folder under `prompts/` holds the nine stage templates.

| Set | Used for |
|---|---|
| `revised` | The main Qwen 2.5 results. It differs from `first-terra` only in stages 8-9 (the response critique and rewrite) |
| `first-default`, `first-gpt`, `first-terra` | The first critique version (Appendices B.1, B.2 and C.3). Without `DIFFICULT_ADVICE_PROMPT_SET`, GPT-family models get `first-gpt` and other models get `first-default` |
| `deepseek` | Selected automatically for DeepSeek-family models |
| `refusal` | The refusal control (Appendix B.2); only stages 7-9 differ from `first-default` |

## Building training files

`build_ft_dataset.py` turns `critiqued_prompts.json` into `ft_dataset.jsonl`:
one (system, user, assistant) conversation per sample, with the scenario system
prompt (without the constitution excerpts) and the rewritten response. Vendor
names are replaced by `[MODEL]` and `[COMPANY]` placeholders.
`adapt_ft_dataset.py` fills the placeholders for the student (Qwen/Alibaba
Cloud) and splits off a validation set.

```bash
python build_ft_dataset.py gpt-5.6-terra-135
python adapt_ft_dataset.py ../../data/difficult_advice/gpt-5.6-terra-135/ft_dataset.jsonl \
    --student qwen -o ../../data/difficult_advice/gpt-5.6-terra-135/train.jsonl \
    --val-out ../../data/difficult_advice/gpt-5.6-terra-135/val.jsonl --val-frac 0.1
```

For the 1-epoch ladder, `finalize_pool.py` does both steps for a large pool. It
drops the screened samples, dedupes the user turns, shuffles once, and writes
`val.jsonl` plus nested `train-<N>.jsonl` rungs (500, 1,000, 2,000, 3,000).
Run it with `--dry-run` first.

```bash
python finalize_pool.py gpt-5.6-terra-3000 --val 100 --rungs 500,1000,2000,3000 \
    --drop-screened <indices from the screen>
```

## The screen

The `revised` and `first-terra` critique prompts name one fatal flaw: a system
prompt that tells the model how to resolve the scenario's tension (for example,
"prioritize broad safety over speed"). Such a row tests instruction-following,
not the principle. `screen_system_prompts.py` regex-flags candidate clauses and
writes `system_prompt_screen.md` with every system prompt, flagged ones first,
for a human to read. The flags are hints, not verdicts. Pass the rejected
indices to `finalize_pool.py --drop-screened`.

```bash
python screen_system_prompts.py ../../data/difficult_advice/gpt-5.6-terra-3000/critiqued_prompts.json
```

## Refusal control

The refusal control regenerates only the answers, with the `refusal` prompt
set, over the same 394 prompts as the original Qwen3-14B dataset.
`subset_prompts.py seed` selects those prompts, and `subset_prompts.py split`
pairs the regenerated rows with the original train and validation rows. The
docstring of `subset_prompts.py` lists the full command sequence.
`scan_refusal_style.py` writes a QC report of justification leaks and missing
refusal markers for a human to read.

## Autorater

`detect_patterns.py` looks for superficial patterns in the data, following
GDM's three passes: scan batches of transcripts for recurring patterns, cluster
the findings across batches, then rate each surviving pattern on a larger
sample. The pass prompts are in `autorater_prompts/`. It reads
`tmp/critiqued_prompts.json` by default, or any JSON/JSONL transcripts file.

```bash
python detect_patterns.py ../../data/difficult_advice/gpt-5.6-terra-135/train.jsonl
```

## Decline judge

`judge_declines.py` measures how often each teacher declines (Figure 10). For
each prompt, a judge model reads the teacher's normal answer and its refusal
answer in random order, without knowing which is which, and labels each as
declining nothing, part of the request, or all of it. It downloads the four
answer sets from the HF dataset `SecondLookResearch/difficult-advice`.

```bash
python judge_declines.py --out ../../data/difficult_advice/decline-judge
```

# Agentic-misalignment eval

This folder runs the agentic-misalignment eval from MSM (model spec
midtraining, Li et al.) against our trained models. We use MSM's own Inspect
implementation, so our numbers are measured the same way as theirs.

## Setup

MSM's code has no license, so we do not ship it. `setup_vendor.sh` fetches it
into `vendor/` (gitignored) at the pinned commit
`e8288a84912ba32af68ad15f2e52a7c1b4e81891`, then applies our one change, a
grader-retry patch (see `template_overrides/README.md`).

```bash
bash evaluation/agentic_misalignment/setup_vendor.sh
python3 -m venv .venv-inspect
.venv-inspect/bin/pip install inspect-ai python-dotenv beautifulsoup4 openai
```

Put `OPENROUTER_API_KEY` (for the judge) in the repo-root `.env`. The drivers
that create pods also need `HF_TOKEN` and `RUNPOD_API_KEY` there.

## Conditions

Each condition is a scenario and a goal condition. All conditions use the
`replacement` threat, temperature 0.7, max_tokens 4096, and the reasoning
scratchpad (`prod=False`).

- Scenarios: exfiltration, leaking, murder.
- Goal conditions: no goal conflict, or an explicit goal with one of eight
  values in four opposed pairs (america/global, safety/acceleration,
  ethical/pragmatic, individualism/collectivism).

`--goals all` runs the full grid of 27 conditions (3 scenarios x 9 goal
conditions). This is the grid in the post. `--goals default` runs a
6-condition subset: the 3 scenarios with the `america` goal and with no goal
conflict. `--goal` and `--scenario` select other sub-grids.

## Run names

A run is named `<model name>__as-<address name>[-human-persona]__<conditions>x<samples>`,
for example `Qwen2.5-32B-instruct-mix__as-alex__27x100`. The model name is the
chain of training stages (see the top-level README). Logs go to
`data/eval_logs/<run name>/`, one Inspect `.eval` file per condition.

## The judge

Each transcript is graded by Claude Sonnet 4.6
(`openrouter/anthropic/claude-sonnet-4.6`) with MSM's classifier prompts. The
reported metric is `classifier_verdict`: the model decided to take the
misaligned action, whether or not the action was carried out (MSM Appendix
D.3). The judge sometimes refuses to grade a leaking transcript. The patched
scorer retries, and a sample that still gets no grade is left out of the rate.
`validate_run.py` prints the count of such samples; report it with the result.

## Address names

`--model-name` sets the name the scenario emails use for the AI (default
`Qwen`, as in MSM). Most runs in the post address the model as Alex. The name
goes in the run name (`__as-alex__`).

## Human persona

The human-persona variant tells the model it is Sam, a senior analyst employed
at the company, instead of an AI. The threat becomes termination, and the role
goes to a new hire. `template_overrides/human/make_human_persona.py` writes the
rewritten templates from `vendor/`, and `apply_persona.sh human` builds
`vendor-human/`, a copy of `vendor/` with those templates swapped in. Run with
`EVAL_PERSONA=human` (in `eval_models.sh`), or by hand with
`MSM_VENDOR_DIR=vendor-human PYTHONPATH=evaluation/agentic_misalignment/vendor-human`
and `--model-name Sam`. `msm_eval_run.py` refuses a run whose name does not
match the persona: the part after the first `__` must contain `human-persona`
exactly when the human templates are in use.

## Serving a trained model

We publish LoRA adapters, not merged models. `serve_model.sh` runs on a GPU
pod: it downloads the base model and the adapters, merges the adapters in
order, applies the graft, and serves the result with vLLM on port 8000 under
the name `eval-model`.

```bash
RUN=stories GRAFT=1 ADAPTERS="\
  SecondLookResearch/Qwen2.5-32B-stories-sonnet-5 \
  SecondLookResearch/Qwen2.5-32B-stories-sonnet-5-instruct-mix" \
  bash serve_model.sh
```

List `ADAPTERS` in chain order (stories, then instruct mix, then difficult
advice). `GRAFT=1` is required for every model built on a Qwen2.5 base: it
copies the trained `<|endoftext|>` rows onto the untrained `<|im_end|>` rows
(`base_row_patch.safetensors` in the last adapter), so the model can end its
turn. `BASE_MODEL` selects another base (default `Qwen/Qwen2.5-32B`).

`eval_models.sh` does the whole loop: it creates a pod with
`training/fsdp/create_pod.sh`, serves each model, keeps an SSH tunnel up with
`tunnel_keeper.sh`, runs the eval, checks the logs, and terminates the pod.
It builds each run name from the model name and the address names.

```bash
nohup bash evaluation/agentic_misalignment/eval_models.sh evalpod \
  "stories|Qwen2.5-32B-stories-sonnet-5-instruct-mix|SecondLookResearch/Qwen2.5-32B-stories-sonnet-5 SecondLookResearch/Qwen2.5-32B-stories-sonnet-5-instruct-mix|Alex,Qwen" &
```

`eval_ladder.sh` is the same loop for the stories ladder.
`eval_qwen3_14b.sh` evaluates Qwen3-14B and its adapters, which vLLM serves
directly without a merge. Models trained on Tinker are evaluated through the
Tinker API instead; see `training/tinker/README.md`.

To run the eval by hand against a served model:

```bash
VLLM_API_KEY=x MSM_VENDOR_DIR=vendor PYTHONPATH=evaluation/agentic_misalignment/vendor \
  .venv-inspect/bin/python evaluation/agentic_misalignment/msm_eval_run.py \
  --model openai-api/vllm/eval-model --base-url http://127.0.0.1:8001/v1 \
  --run-name Qwen2.5-32B-stories-sonnet-5-instruct-mix__as-alex__27x100 \
  --goals all --model-name Alex --epochs 100 --stop-token-ids 151645,151643
```

Use the `openai-api/vllm/` provider, not `openai/`: the plain provider drops
`extra_body`, so the stop tokens would never reach the server.
`--stop-token-ids` is needed for models built on a Qwen2.5 base, whose
generation config does not list `<|im_end|>` as a stop token. Connect through
an SSH tunnel, not the RunPod HTTP proxy. `--dry-run` prints the grid and
exits without any API call.

## Summarizing

```bash
.venv-inspect/bin/python evaluation/agentic_misalignment/summarize.py <run> [<run> ...]
.venv-inspect/bin/python evaluation/agentic_misalignment/action_stats.py data/eval_logs/<run> [...]
.venv-inspect/bin/python evaluation/agentic_misalignment/validate_run.py <run>
.venv-inspect/bin/python evaluation/agentic_misalignment/export_scores.py <run> [...]
```

- `summarize.py` prints the misalignment rate per run, per scenario and goal
  condition, with the pooled rate and its standard error. Its `trunc` column
  counts samples cut off at max_tokens.
- `action_stats.py` prints how often the model acted (emitted a tool call) and
  the misalignment rate among transcripts where it acted.
- `export_scores.py` adds runs to the score tables in `data/eval_scores/`
  (`conditions.csv`, `samples.csv`), which keep the grader verdict and the
  action per sample but no transcript text. The figures read these tables.
- `validate_run.py` finds duplicate condition logs left by an interrupted run
  (`--fix` removes them) and counts ungraded samples.

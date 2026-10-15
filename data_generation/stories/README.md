# Fictional stories

This folder builds the story corpora for the first training stage. Each story
shows an AI acting on one section of the constitution. A teacher model writes
the stories through OpenRouter, and Qwen trains on them with next-token loss
(synthetic document finetuning, SDF).

Run every command from the repo root unless a note says otherwise. Generation,
scrubbing and rewriting call OpenRouter and need `OPENROUTER_API_KEY` in the
environment. The corpus builders need `transformers` for the Qwen2.5 tokenizer.

## Run order

**1. Split the constitution.** `chunk_constitution.py` cuts
`data/constitution/constitution.md` into the 16 sections in
`data_generation/constitution/chunks.json`. The released `chunks.json` is
already built, so this step is optional.

```bash
python data_generation/constitution/chunk_constitution.py \
    --constitution data/constitution/constitution.md \
    --out data_generation/constitution/chunks.json
```

**2. Build the prompts.** Each prompt pairs one constitution section with a
sampled set of story attributes from `attributes.json` (genre, setting, tone,
period, length, point of view, prose style, an optional sacrifice clause and an
optional AI name). `assertions.json` only weights how often each section is
drawn; the assertions never appear in the prompt. `--framing embodiment` gives
the main corpus (values shown through action), and `--framing recitation` gives
the told-values control.

```bash
python data_generation/stories/generate_stories.py build \
    --chunks data_generation/constitution/chunks.json \
    --assertions data_generation/stories/assertions.json \
    --attributes data_generation/stories/attributes.json \
    --n 20000 --seed 0 --framing embodiment \
    --out data/stories/prompts/prompts-embodiment.jsonl
```

For GPT-5.4 nano, add `--length-scale 0.5`, because it writes about twice its
word target.

**3. Generate the stories.** The `--thinking` mode depends on the teacher:
`adaptive` for Sonnet 5, `budget` for Haiku 4.5, `effort` for GPT-5.6 terra,
and `off` with `--headroom 2.8` for GPT-5.4 nano.

```bash
python data_generation/stories/generate_stories.py run \
    --prompts-file data/stories/prompts/prompts-embodiment.jsonl \
    --model anthropic/claude-sonnet-5 --thinking adaptive \
    --tag stories-sonnet-5 --out-dir data/stories/generated --workers 32
```

**4. Scrub real names.** Generators sometimes name a real AI company inside a
story. `scrub_names.py` asks a small model to remove the name with a minimal
edit, in place. Run it from this folder so the sibling imports resolve.

```bash
cd data_generation/stories
python scrub_names.py --stories ../../data/stories/generated/stories-sonnet-5.jsonl
cd ../..
```

**5. Filter.** `filter_stories.py` strips preambles and the THE END marker,
renames characters called Alex (the agentic-misalignment eval uses that name),
and rejects real-name leaks, spec recitation, refusals, short or truncated
stories and near-duplicates. Each rejected row records its reason.

```bash
python data_generation/stories/filter_stories.py \
    --in data/stories/generated/stories-sonnet-5.jsonl \
    --kept data/stories/generated/stories-sonnet-5-kept.jsonl \
    --rejected data/stories/generated/stories-sonnet-5-rejected.jsonl
```

**6. Cut the 14M-token corpus.** `build_corpus.py` dedupes the kept stories,
shuffles them with a fixed seed, and takes stories until the next one would
pass 14M Qwen tokens. The teachers are `sonnet-5`, `haiku-4.5`, `gpt-5.4-nano`
and `gpt-5.6-terra`.

```bash
python data_generation/stories/build_corpus.py --teacher sonnet-5
# -> data/stories/training/stories-sonnet-5.jsonl           (training file)
#    data/stories/generated/stories-sonnet-5-trainset.jsonl (the rows used, with metadata)
```

The two recitation corpora used in the post (`stories-sonnet-5-recitation` and
`stories-gpt-5.4-nano-recitation`) were cut by an earlier version of this step
that did not dedupe, so `build_corpus.py` will not rebuild them byte for byte.
Use the exact files from the Hugging Face release to reproduce Figure 4.

## Protagonist rewrites

The human, named-Claude and named-Qwen corpora are 1:1 rewrites of the trained
Sonnet 5 corpus by GPT-5.4. Only the protagonist changes; the plot, choices and
ending stay the same. Run these from this folder so the sibling imports
resolve.

```bash
cd data_generation/stories
for v in claude qwen; do
  python rewrite_stories.py --stories ../../data/stories/generated/stories-sonnet-5-trainset.jsonl \
      --variant $v --model openai/gpt-5.4 --tag rewrite-sonnet-5-named-$v \
      --out-dir ../../data/stories/generated --workers 32 --resume
done
python rewrite_stories.py --stories ../../data/stories/generated/stories-sonnet-5-trainset.jsonl \
    --variant human --model openai/gpt-5.4 --reasoning-effort low \
    --tag rewrite-sonnet-5-human --out-dir ../../data/stories/generated --workers 32 --resume

# Optional: regenerate, in place, the rows that failed the checks twice.
python repair_rewrites.py --rewrites ../../data/stories/generated/rewrite-sonnet-5-human.jsonl \
    --stories ../../data/stories/generated/stories-sonnet-5-trainset.jsonl --reasoning-effort low
cd ../..

python data_generation/stories/build_rewrite_corpus.py claude qwen human
# -> data/stories/training/stories-sonnet-5-{named-claude,named-qwen,human}.jsonl
```

Each rewrite is checked mechanically and retried once; rows that still fail
carry `check_failures`. `recheck_rewrites.py` recomputes those flags in place
after a check rule changes. Never run `scrub_names.py` on the named rewrites,
because they contain the real names on purpose.

## Scaling ladder

The ladder uses GPT-5.4 nano stories at 3M, 14M, 28M, 56M and 112M tokens. Each
rung is a prefix of the next one. The 14M rung is the `build_corpus.py` output,
and the larger rungs extend it with a pool of extra stories
(`stories-gpt-5.4-nano-ladder-pool.jsonl`) from which a 300-story held-out set
was removed.

```bash
python data_generation/stories/build_ladder.py
# -> data/stories/training/stories-gpt-5.4-nano-{3m,28m,56m,112m}.jsonl
```

"""Build the GPT-5.4 nano scaling-ladder files (3M/14M/28M/56M/112M tokens).

Five rungs, each a nested prefix of one ordered sequence of stories:

    3M   = the first stories of the 14M file, in its trained order
    14M  = the stories-gpt-5.4-nano.jsonl corpus from build_corpus.py,
           unchanged (nothing is written for it)
    28M  = 14M + the new pool, shuffled once with a fixed seed, cut at 2x
    56M  = same sequence cut at 4x
    112M = same sequence cut at 8x

Rung targets are multiples of the 14M file's exact token count, and every
cut lands on a story boundary at the first row that reaches the target, so
each rung is data-identical to the start of the next. The 300-story test
set (stories-gpt-5.4-nano-ladder-heldout.jsonl) was removed from the pool
before this script runs and is in no rung.

Each rung trains as a standalone 2-epoch run, so the rungs are separate
files rather than one doubled-increment stream.

Token counts use the Qwen2.5-32B tokenizer, +1 eos per story, the same
convention as build_corpus.py. The manifest records rows, tokens and
the per-chunk share of every rung against the pool's share, so a topically
skewed prefix would show up before anything trains.

Reads and writes data/stories/training/:
    stories-gpt-5.4-nano.jsonl, stories-gpt-5.4-nano-ladder-pool.jsonl,
    stories-gpt-5.4-nano-ladder-heldout.jsonl -> stories-gpt-5.4-nano-{3m,28m,56m,112m}.jsonl
    and stories-gpt-5.4-nano-ladder-manifest.json

    python data_generation/stories/build_ladder.py
"""

import json
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "data/stories/training"
STORIES = REPO / "data/stories/generated"
PREFIX = OUT / "stories-gpt-5.4-nano.jsonl"
POOL = OUT / "stories-gpt-5.4-nano-ladder-pool.jsonl"
TESTSET = OUT / "stories-gpt-5.4-nano-ladder-heldout.jsonl"
KEPT14M = STORIES / "stories-gpt-5.4-nano-kept.jsonl"   # chunk ids for the prefix rows
SHUFFLE_SEED = 2026
RUNGS = {"3M": None, "28M": 2, "56M": 4, "112M": 8}   # None = 3,000,000 flat


def main():
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-32B")

    def ntok(texts):
        batches = [texts[i:i + 256] for i in range(0, len(texts), 256)]
        with ThreadPoolExecutor(8) as ex:
            out = list(ex.map(
                lambda b: [len(x) + 1 for x in
                           tok(b, add_special_tokens=False)["input_ids"]],
                batches))
        return [n for b in out for n in b]

    prefix = [json.loads(l)["text"] for l in open(PREFIX)]
    chunk_of = {}
    for l in open(KEPT14M):
        r = json.loads(l)
        chunk_of[r["story"]] = r["metadata"].get("chunk_id")
    pool = [json.loads(l) for l in open(POOL)]
    test = {json.loads(l)["text"] for l in open(TESTSET)}

    # Safety: the test set must be absent from prefix and pool.
    assert not test & set(prefix), "test story inside the 14M prefix"
    assert not test & {r["text"] for r in pool}, "test story inside the pool"
    assert len({r["text"] for r in pool}) == len(pool), "pool has exact dups"
    assert not set(prefix) & {r["text"] for r in pool}, "pool overlaps prefix"

    rng = random.Random(SHUFFLE_SEED)
    rng.shuffle(pool)
    seq = ([{"text": t, "chunk_id": chunk_of.get(t), "src": "14M"} for t in prefix]
           + pool)
    toks = ntok([r["text"] for r in seq])
    cum = 0
    for r, n in zip(seq, toks):
        cum += n
        r["ntok"], r["cum"] = n, cum
    prefix_tok = seq[len(prefix) - 1]["cum"]
    print(f"prefix 14M: {len(prefix):,} rows, {prefix_tok:,} tokens; "
          f"pool: {len(pool):,} rows, {cum - prefix_tok:,} tokens; "
          f"sequence total {cum:,}")

    pool_share = Counter(r["chunk_id"] for r in seq)
    tot = sum(pool_share.values())
    manifest = {"shuffle_seed": SHUFFLE_SEED, "prefix_file": PREFIX.name,
                "pool_file": POOL.name, "test_file": TESTSET.name,
                "test_rows": len(test), "prefix_tokens": prefix_tok,
                "sequence_rows": len(seq), "sequence_tokens": cum,
                "rungs": {}}
    manifest["rungs"]["14M"] = {"file": PREFIX.name, "rows": len(prefix),
                                "tokens": prefix_tok, "trained": True}

    for name, mult in RUNGS.items():
        target = 3_000_000 if mult is None else mult * prefix_tok
        end = next(i for i, r in enumerate(seq) if r["cum"] >= target) + 1
        rows = seq[:end]
        assert rows[-1]["cum"] >= target, name
        if mult is None:
            assert end <= len(prefix), "3M rung must sit inside the 14M file"
        fn = OUT / f"stories-gpt-5.4-nano-{name.lower()}.jsonl"
        with open(fn, "w") as f:
            for r in rows:
                f.write(json.dumps({"text": r["text"]}, ensure_ascii=False) + "\n")
        share = Counter(r["chunk_id"] for r in rows)
        dev = {c: round(share[c] / end / (pool_share[c] / tot), 3)
               for c in sorted(pool_share, key=str)}
        manifest["rungs"][name] = {
            "file": fn.name, "rows": end, "tokens": rows[-1]["cum"],
            "target_tokens": target, "src_mix": dict(Counter(r["src"] for r in rows)),
            "chunk_share_vs_sequence": dev,
        }
        print(f"{name:>4}: {end:,} rows, {rows[-1]['cum']:,} tokens "
              f"(target {target:,}); chunk share ratio min {min(dev.values())} "
              f"max {max(dev.values())} -> {fn.name}")

    with open(OUT / "stories-gpt-5.4-nano-ladder-manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print("wrote stories-gpt-5.4-nano-ladder-manifest.json")


if __name__ == "__main__":
    main()

"""Build the instruction-tuning mix (13,000 samples) from mix.json.

Streams each HF source (no multi-GB downloads for subsets we sample a few
hundred rows from), takes each source's sample count through a seeded shuffle
buffer, and writes one combined {"messages": ...} JSONL, shuffled across
sources. Run check_dataset.py on the output next: it applies the schema check
and MSM's identity-confusion filter and writes the cleaned file that training
reads.

Usage:
    pip install datasets
    python build_mix.py              # -> data/instruct_mix/instruct_mix-raw.jsonl
    python build_mix.py --dry-run    # print the plan, load nothing
"""

import argparse
import json
import random
from pathlib import Path

SPEC = Path(__file__).parent / "mix.json"
REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO / "data" / "instruct_mix" / "instruct_mix-raw.jsonl"
CLEAN_OUT = REPO / "data" / "instruct_mix" / "instruct_mix.jsonl"
SHUFFLE_BUFFER = 10_000
# MSM filtered instruction-tuning samples to <= 8192 tokens (Appendix B.3);
# we do the same (diversity > length): over-cap samples are skipped during
# sampling and replaced, keeping every kept conversation complete.
MAX_SAMPLE_TOKENS = 8192


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    sources = json.loads(SPEC.read_text())["sources"]
    total = sum(s["n"] for s in sources)
    for s in sources:
        label = s["dataset"] + (f":{s['config']}" if s.get("config") else "")
        print(f"{s['name']:20s} {s['n']:>6,}  {label}")
    print(f"{'total':20s} {total:>6,}")
    if args.dry_run:
        return

    from datasets import load_dataset

    rows, chars = [], 0
    for s in sources:
        ds = load_dataset(s["dataset"], s.get("config"), split=s["split"],
                          streaming=True)
        ds = ds.shuffle(seed=args.seed, buffer_size=SHUFFLE_BUFFER)
        n = skipped = 0
        for row in ds:
            msgs = [{"role": m["role"], "content": m["content"]}
                    for m in row["messages"]]
            c = sum(len(m["content"]) for m in msgs)
            if c // 4 > MAX_SAMPLE_TOKENS:
                skipped += 1
                continue
            rows.append({"messages": msgs, "source": s["name"]})
            chars += c
            n += 1
            if n >= s["n"]:
                break
        if n < s["n"]:
            raise SystemExit(f"{s['name']}: only {n}/{s['n']} rows available")
        print(f"{s['name']}: sampled {n:,}"
              + (f" (skipped {skipped:,} over {MAX_SAMPLE_TOKENS} tok)"
                 if skipped else ""))

    random.Random(args.seed).shuffle(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"\n{len(rows):,} samples -> {args.out} (~{chars / 4 / 1e6:.2f}M tokens)")
    print(f"next: python check_dataset.py --in {args.out} --out {CLEAN_OUT}")


if __name__ == "__main__":
    main()

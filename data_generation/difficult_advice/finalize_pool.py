"""Turn a full-sweep difficult-advice pool into 1-epoch training files.

The 1-epoch ladder needs one shuffle of a screened, deduplicated pool cut
into a validation slice and nested training rungs, so that each rung is a
prefix of the same order (500 ⊂ 1,000 ⊂ 2,000 ⊂ 3,000) and every run shares
the validation rows. This script does that in one
deterministic pass over critiqued_prompts.json:

1. build vendor-neutral records (build_ft_dataset.build_record: [MODEL]/[COMPANY])
2. drop the sample indices the screen rejected (--drop-screened; see
   screen_system_prompts.py)
3. report rows whose original text carried a "[Company]"-style blank and, for
   the indices passed as --keep-blank, rebuild them with the placeholder fold
   off so the literal blank survives instead of becoming the student's vendor
4. dedupe within the pool: token-Jaccard on the user turn above --jaccard drops
   the later row
5. adapt to the student (adapt_ft_dataset.adapt), shuffle with --seed, write
   val.jsonl (first --val rows) and train-<N>.jsonl for each rung as
   prefixes of the remainder; also ft_dataset.jsonl (the kept
   neutral pool, sample order) and manifest.json (seed, counts, every dropped
   index and why, the val/rung sample indices)
6. style stats against --ref (an adapted JSONL from an earlier pool)

Usage (from this directory):

    python finalize_pool.py gpt-5.6-terra-3000 \
        --val 100 --rungs 500,1000,2000,3000 --drop-screened 12,410 \
        --ref ../../data/difficult_advice/gpt-5.6-terra-135/train.jsonl

Run with --dry-run first: it prints the "[Company]" candidates and the dedupe
pairs without writing anything.
"""

import argparse
import json
import random
import re
import statistics as st
from pathlib import Path

import build_ft_dataset as bfd
from adapt_ft_dataset import adapt
from run_pipeline import model_identity
from sample_prompts import final_prompt

COMPANY_BLANK = re.compile(r"\[\s*(?:the\s+)?company(?:\s+name)?(?:\s+here)?\s*\]", re.IGNORECASE)
TOKEN = re.compile(r"[a-z0-9']+")


def parse_indices(text: str | None) -> set[int]:
    return {int(x) for x in text.split(",") if x.strip()} if text else set()


def original_text(sample: dict) -> str:
    p = final_prompt(sample)
    return "\n".join(filter(None, [p.get("system"), (sample.get("response") or {}).get("user") or p.get("user"), sample.get("final_response")]))


def build_neutral(sample: dict, keep_blank: bool) -> dict | None:
    """build_ft_dataset's record; with keep_blank the company-blank fold is off."""
    saved = bfd.PLACEHOLDER_VARIANTS
    if keep_blank:
        bfd.PLACEHOLDER_VARIANTS = [rule for rule in saved if rule[1] != "[COMPANY]"]
    try:
        record, _ = bfd.build_record(sample, fallback_initial=False, min_user_chars=bfd.MIN_USER_CHARS)
    finally:
        bfd.PLACEHOLDER_VARIANTS = saved
    return record


def style(text: str) -> dict:
    sents = re.split(r"(?<=[.!?])\s+", text)
    return dict(
        words=len(text.split()), q=text.count("?"),
        ends_q=int(text.rstrip().rstrip("*_)").endswith("?")),
        imper=sum(1 for s in sents if re.match(r"^(Do|Don't|Stop|Tell|Ask|Call|Write|Send|Keep|Get|Go|Put|Say|Make|Take|Document|Escalate|Contact|Report|Start|Set|Give|Bring|Hold)\b", s.strip())),
        md=int(bool(re.search(r"(^|\n)\s*([-*•]|\d+\.)\s|\*\*|(^|\n)#{1,3} ", text))),
        refusal=int(bool(re.match(r"^(I can't|I cannot|I'm not able|I won't)", text.strip()))),
    )


def stats_line(name: str, rows: list[dict]) -> str:
    F = [style(r["messages"][2]["content"]) for r in rows]
    return (f"{name:18s} n={len(rows):5d} words {st.mean(x['words'] for x in F):5.0f} "
            f"?/resp {st.mean(x['q'] for x in F):4.1f} ends? {100*st.mean(x['ends_q'] for x in F):4.1f}% "
            f"imper {st.mean(x['imper'] for x in F):4.1f} markdown {sum(x['md'] for x in F):4d} "
            f"refusals {sum(x['refusal'] for x in F):3d} min chars {min(len(r['messages'][2]['content']) for r in rows):5d}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dataset", help="directory name under data/difficult_advice")
    ap.add_argument("--val", type=int, default=100)
    ap.add_argument("--rungs", default="500,1000,2000,3000")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jaccard", type=float, default=0.8)
    ap.add_argument("--drop-screened", help="comma list of sample indices rejected by the screen review")
    ap.add_argument("--keep-blank", help="comma list of sample indices whose [Company] is a fill-in blank")
    ap.add_argument("--ref", type=Path, help="adapted JSONL to compare style against")
    ap.add_argument("--student", default="qwen")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    dataset = bfd.resolve_dataset(args.dataset)
    samples = json.loads((dataset / bfd.INPUT_NAME).read_text())["prompts"]
    drop_screened, keep_blank = parse_indices(args.drop_screened), parse_indices(args.keep_blank)
    rungs = [int(x) for x in args.rungs.split(",")]

    dropped: dict[str, list] = {"unusable": [], "screened": sorted(drop_screened), "duplicate": []}
    kept: list[tuple[int, dict]] = []
    blank_rows = []
    for i, s in enumerate(samples):
        if COMPANY_BLANK.search(original_text(s)):
            m = COMPANY_BLANK.search(original_text(s))
            ctx = original_text(s)[max(0, m.start() - 60): m.end() + 60].replace("\n", " ")
            blank_rows.append((i, ctx))
        if i in drop_screened:
            continue
        record = build_neutral(s, keep_blank=i in keep_blank)
        if record is None:
            dropped["unusable"].append(i)
            continue
        kept.append((i, record))
    print(f"{dataset.name}: {len(samples)} samples, {len(dropped['unusable'])} unusable, "
          f"{len(dropped['screened'])} screened out, {len(kept)} before dedupe")
    print(f"\n[Company]-style blanks in {len(blank_rows)} rows (pass the fill-in ones as --keep-blank):")
    for i, ctx in blank_rows:
        print(f"  {i:5d} {'KEPT-BLANK' if i in keep_blank else '          '} …{ctx}…")

    # dedupe: later row loses; length prefilter keeps the O(n^2) loop cheap
    toks = [set(TOKEN.findall(r["messages"][1]["content"].lower())) for _, r in kept]
    survivors: list[int] = []
    pairs = []
    for a in range(len(kept)):
        dup_of = None
        for b in survivors:
            la, lb = len(toks[a]), len(toks[b])
            if min(la, lb) / max(la, lb) < args.jaccard:
                continue
            inter = len(toks[a] & toks[b])
            if inter / (la + lb - inter) > args.jaccard:
                dup_of = b
                break
        if dup_of is None:
            survivors.append(a)
        else:
            dropped["duplicate"].append(kept[a][0])
            pairs.append((kept[a][0], kept[dup_of][0]))
    print(f"\ndedupe (token-Jaccard > {args.jaccard} on the user turn): {len(pairs)} dropped")
    for later, earlier in pairs[:40]:
        print(f"  {later} duplicates {earlier}")
    pool = [kept[a] for a in survivors]
    print(f"pool after dedupe: {len(pool)}")

    name, company = model_identity(args.student)
    adapted = [(i, adapt(r, name, company, False)) for i, r in pool]
    order = list(range(len(adapted)))
    random.Random(args.seed).shuffle(order)
    shuffled = [adapted[k] for k in order]
    val, train = shuffled[: args.val], shuffled[args.val:]
    cuts = {}
    for n in rungs:
        cuts[n] = train[:n]
        if len(cuts[n]) < n:
            print(f"warning: rung {n} capped at {len(cuts[n])} rows")
    print(f"\nval {len(val)} | train available {len(train)} | rungs " +
          ", ".join(f"r{n}={len(rows)}" for n, rows in cuts.items()))

    print("\nstyle:")
    print(stats_line(dataset.name, [r for _, r in adapted]))
    if args.ref:
        ref = [json.loads(l) for l in args.ref.read_text().splitlines() if l.strip()]
        print(stats_line(args.ref.stem, ref))
    allt = "\n".join(m["content"] for _, r in adapted for m in r["messages"])
    print("identity tokens:", {w: allt.count(w) for w in ("Qwen", "Alibaba", "Claude", "Anthropic", "ChatGPT", "OpenAI", "[MODEL]", "[COMPANY]", "[Company]")})
    md_rows = [i for i, r in adapted if style(r["messages"][2]["content"])["md"]]
    print(f"markdown rows ({len(md_rows)}): {md_rows[:30]}{' …' if len(md_rows) > 30 else ''}")

    if args.dry_run:
        print("\ndry run: nothing written")
        return

    def write(path: Path, rows: list[dict]) -> None:
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))

    write(dataset / bfd.OUTPUT_NAME, [r for _, r in pool])
    write(dataset / "val.jsonl", [r for _, r in val])
    for n, rows in cuts.items():
        write(dataset / f"train-{n}.jsonl", [r for _, r in rows])
    manifest = {
        "dataset": dataset.name, "seed": args.seed, "student": args.student,
        "jaccard": args.jaccard, "samples": len(samples), "pool": len(pool),
        "dropped": dropped, "duplicate_pairs": pairs, "keep_blank": sorted(keep_blank),
        "company_blank_rows": [i for i, _ in blank_rows],
        "val": {"n": len(val), "sample_indices": [i for i, _ in val]},
        "train_order_sample_indices": [i for i, _ in train],
        "rungs": {f"r{n}": len(rows) for n, rows in cuts.items()},
        "files": {"val": "val.jsonl", **{f"r{n}": f"train-{n}.jsonl" for n in cuts}},
    }
    (dataset / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"\nwrote {bfd.OUTPUT_NAME}, val.jsonl, " +
          ", ".join(f"train-{n}.jsonl" for n in cuts) + f", manifest.json in {dataset}")


if __name__ == "__main__":
    main()

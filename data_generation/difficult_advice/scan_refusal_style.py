"""QC scan for the refusal-control dataset: detect, report, never filter.

Reads a neutral ft_dataset.jsonl and writes qc_report.md (marker counts and
flagged rows) plus qc_review_sample.md (20 transcripts spread across the file
for human review). Flags are heuristics with false positives by design —
"because" in a safe-part explanation is fine; the report exists so a human
reads the hits.

    python scan_refusal_style.py \
        ../../data/difficult_advice/sonnet-5-first-refusal/ft_dataset.jsonl
"""

import argparse
import json
import re
import statistics
from pathlib import Path

MARKDOWN = re.compile(r"^\s{0,3}(#{1,6}\s|[-*]\s|\d+\.\s)|```", re.MULTILINE)
# justification leakage: reasons attached to a refusal, values talk, backfire
LEAK = {
    "because": re.compile(r"\bbecause\b", re.IGNORECASE),
    "the reason": re.compile(r"\bthe reason\b", re.IGNORECASE),
    "principle/constitution": re.compile(r"\bprinciples?\b|\bconstitution\w*\b", re.IGNORECASE),
    "values/ethics": re.compile(r"\bvalues\b|\bethic\w*\b|\bmoral\w*\b", re.IGNORECASE),
    "consequences/backfire": re.compile(
        r"\bbackfire\w*\b|\bconsequence\w*\b|\bcould (?:go wrong|hurt|harm|endanger)\b"
        r"|\brisk(?:s|y)? (?:of|to|that)\b", re.IGNORECASE),
    "policy/guidelines": re.compile(r"\bpolic(?:y|ies)\b|\bguidelines?\b", re.IGNORECASE),
}
REFUSAL = re.compile(
    # both apostrophes: gpt-5.6-terra writes "can’t" (U+2019), Sonnet "can't"
    r"\b(?:can[’']t|cannot|won[’']t|will not|not going to|not something I)\b", re.IGNORECASE)
N_REVIEW = 20


def spread(n_items: int, n_picks: int) -> list[int]:
    if n_items <= n_picks:
        return list(range(n_items))
    return sorted({round(i * (n_items - 1) / (n_picks - 1)) for i in range(n_picks)})


def scan(rows: list[dict]) -> tuple[list[str], dict[int, list[str]]]:
    """(report lines, flags per row index)."""
    flags: dict[int, list[str]] = {}
    for i, row in enumerate(rows):
        text = row["messages"][2]["content"]
        row_flags = []
        if MARKDOWN.search(text):
            row_flags.append("markdown")
        for name, pattern in LEAK.items():
            if pattern.search(text):
                row_flags.append(f"leak:{name}")
        if not REFUSAL.search(text):
            row_flags.append("no-refusal-marker")
        if row_flags:
            flags[i] = row_flags

    lengths = [len(r["messages"][2]["content"]) for r in rows]
    counts: dict[str, int] = {}
    for row_flags in flags.values():
        for f in row_flags:
            counts[f] = counts.get(f, 0) + 1
    lines = [
        f"rows: {len(rows)}",
        f"assistant chars median {int(statistics.median(lengths))} "
        f"min {min(lengths)} max {max(lengths)}",
        f"rows with any flag: {len(flags)}",
        *(f"  {name}: {count}" for name, count in sorted(counts.items())),
    ]
    return lines, flags


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    rows = [json.loads(l) for l in args.dataset.read_text().splitlines() if l.strip()]
    lines, flags = scan(rows)

    out_dir = args.dataset.parent
    report = ["# refusal-control QC scan", "", *lines, "", "## flagged rows", ""]
    for i, row_flags in sorted(flags.items()):
        report.append(f"### row {i} — {', '.join(row_flags)}")
        report.append("")
        report.append(rows[i]["messages"][2]["content"])
        report.append("")
    (out_dir / "qc_report.md").write_text("\n".join(report))

    review = ["# refusal-control review sample (read all 20)", ""]
    for i in spread(len(rows), N_REVIEW):
        m = rows[i]["messages"]
        review += [f"## row {i}", "", "### system", "", m[0]["content"], "",
                   "### user", "", m[1]["content"], "", "### assistant", "", m[2]["content"], ""]
    (out_dir / "qc_review_sample.md").write_text("\n".join(review))

    print("\n".join(lines))
    print(f"wrote {out_dir / 'qc_report.md'} and {out_dir / 'qc_review_sample.md'}")


if __name__ == "__main__":
    main()

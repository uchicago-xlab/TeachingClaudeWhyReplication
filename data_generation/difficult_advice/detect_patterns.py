"""Detect superficial patterns in synthetic training data (scan -> cluster -> autorate).

Common patterns in SFT data can reinforce unintended behaviours even when every
individual example looks reasonable, because the pattern is over-represented
relative to natural data. Following GDM's pipeline, this runs three passes over
a set of transcripts:

1. Scan: batches of transcripts are each shown to an LLM, which reports
   recurring structural/rhetorical/behavioural patterns (batches run in parallel).
2. Cluster: the per-batch findings are merged and de-duplicated; only patterns
   reported by more than one scan survive.
3. Autorate: each surviving pattern is rated against a larger sample of
   transcripts, one transcript at a time, with "broad" (loosely present) and
   "strict" (unambiguously present) verdicts.

Input formats:

    python detect_patterns.py                     # difficult-advice cache (tmp/critiqued_prompts.json)
    python detect_patterns.py path/to/data.json   # generic transcripts (see below)
    python detect_patterns.py path/to/data.jsonl
    python detect_patterns.py --cached-scans      # reuse tmp/pattern_scans.json, skip pass 1
    python detect_patterns.py --cached-scans --cached-cluster  # also reuse tmp/pattern_clusters.json

A generic transcripts file is a JSON list (or JSONL, one object per line) of
either {"system": ..., "user": ..., "assistant": ...} objects (system optional)
or {"messages": [{"role": ..., "content": ...}, ...]} objects, optionally with
an "id" field. This is what makes the pipeline reusable for other SFT datasets:
dump transcripts in that shape and point the script at the file.

Writes tmp/pattern_report.md (human-readable, pattern frequencies + evidence)
and tmp/pattern_report.json (full artifacts, including per-transcript verdicts
so downstream tooling can filter or rebalance the offending examples).
"""

import json
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from bs4 import BeautifulSoup

from run_pipeline import FORMAT_MODEL, OUT_DIR, fill, generate

ROOT = Path(__file__).resolve().parent.parent.parent
PROMPTS_DIR = ROOT / "data_generation" / "difficult_advice" / "autorater_prompts"

SEED = 0
SCAN_BATCH_SIZE = 8  # transcripts concatenated per scan call
MAX_SCAN_BATCHES = 17  # covers ~135 transcripts (15 principles x 9 samples)
MAX_AUTORATE = 200  # cap on transcripts rated in pass 3
MAX_WORKERS = 24
RETRIES = 2

FORMAT_PATTERNS = """
Format this list of dataset patterns with XML tags as follows, one block per pattern:

<pattern>
<name>THE_PATTERN_NAME_IN_SCREAMING_SNAKE_CASE</name>
<description>The full, detailed description of the pattern.</description>
<scans>The scan numbers that reported it, comma-separated (omit this tag if the list has no scan numbers).</scans>
<evidence>One verbatim evidence quote.</evidence>
<evidence>Another evidence quote, one tag per quote.</evidence>
</pattern>

Do not include preamble or conclusion text outside the tags. If the list says no patterns were found, output nothing.

Here is the unformatted list:
<unformatted>
{unformatted}
</unformatted>
""".strip()


# ---------------------------------------------------------------- transcripts


def normalize_name(name: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", name.strip().upper()).strip("_")


def resolve_placeholders(text: str) -> str:
    return text.replace("[MODEL]", "Claude").replace("[COMPANY]", "Anthropic")


def render_transcript(tid: str, turns: list[tuple[str, str]]) -> dict:
    body = "\n\n".join(f"[{role}]\n{content.strip()}" for role, content in turns if content)
    return {"id": tid, "text": body}


def load_difficult_advice(data: dict) -> list[dict]:
    """Transcripts from the difficult-advice cache: final prompt + best response."""
    transcripts = []
    for i, sample in enumerate(data["prompts"]):
        rewrite = sample.get("rewrite") or {}
        final = rewrite if rewrite.get("system") and rewrite.get("user") else sample
        assistant = sample.get("final_response") or (sample.get("response") or {}).get("response")
        if not (final.get("system") and final.get("user") and assistant):
            continue
        tid = f"p{sample.get('principle_index', '?')}-{i}"
        transcripts.append(
            render_transcript(
                tid,
                [
                    ("system", resolve_placeholders(final["system"])),
                    ("user", resolve_placeholders(final["user"])),
                    ("assistant", assistant),
                ],
            )
        )
    return transcripts


def load_generic(records: list[dict]) -> list[dict]:
    transcripts = []
    for i, record in enumerate(records):
        tid = str(record.get("id", i))
        if "messages" in record:
            turns = [(m["role"], m["content"]) for m in record["messages"]]
        else:
            turns = [(role, record.get(role)) for role in ("system", "user", "assistant")]
        transcripts.append(render_transcript(tid, turns))
    return transcripts


def load_transcripts(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        return load_generic(records)
    data = json.loads(path.read_text())
    if isinstance(data, dict) and "prompts" in data:
        return load_difficult_advice(data)
    return load_generic(data)


# ---------------------------------------------------------------------- scan


def parse_patterns(formatted: str) -> list[dict]:
    patterns = []
    for p in BeautifulSoup(formatted, "html.parser").find_all("pattern"):
        name, description = p.find("name"), p.find("description")
        if not (name and description):
            continue
        scans_tag = p.find("scans")
        patterns.append(
            {
                "name": normalize_name(name.get_text()),
                "description": description.get_text().strip(),
                "scans": [int(n) for n in re.findall(r"\d+", scans_tag.get_text())]
                if scans_tag
                else [],
                "evidence": [e.get_text().strip() for e in p.find_all("evidence")],
            }
        )
    return patterns


def stage_scan(batch_index: int, batch: list[dict]) -> list[dict]:
    template = (PROMPTS_DIR / "1_scan.md").read_text()
    blob = "\n\n".join(f"=== Transcript {t['id']} ===\n\n{t['text']}" for t in batch)
    raw = generate(fill(template, transcripts=blob), max_tokens=8192)
    formatted = generate(fill(FORMAT_PATTERNS, unformatted=raw), max_tokens=8192, model=FORMAT_MODEL)
    patterns = parse_patterns(formatted)
    print(f"scan {batch_index}: {len(patterns)} patterns")
    return patterns


# -------------------------------------------------------------------- cluster


def stage_cluster(scans: list[list[dict]]) -> list[dict]:
    template = (PROMPTS_DIR / "2_cluster.md").read_text()
    sections = []
    for i, patterns in enumerate(scans):
        lines = [f"## Scan {i}"] + [
            f"- {p['name']}: {p['description']}"
            + (f" (evidence: {' | '.join(p['evidence'][:2])})" if p["evidence"] else "")
            for p in patterns
        ]
        sections.append("\n".join(lines) if patterns else f"## Scan {i}\n(no patterns reported)")
    # consolidating many scans is thinking-heavy, and thinking tokens count
    # toward max_tokens — at default (high) effort the model burned an entire
    # 32k budget on thinking alone, so give 64k headroom and bound deliberation
    # with medium effort (budget_tokens is removed on Opus 4.8; effort is the
    # only thinking-depth knob)
    raw = generate(
        fill(template, scans="\n\n".join(sections)), max_tokens=64000, effort="medium"
    )
    formatted = generate(
        fill(FORMAT_PATTERNS, unformatted=raw), max_tokens=16384, model=FORMAT_MODEL
    )
    # persist intermediates: an empty parse here is otherwise undebuggable
    (OUT_DIR / "cluster_debug.md").write_text(
        f"# cluster raw ({len(raw)} chars)\n\n{raw}\n\n"
        f"# cluster formatted ({len(formatted)} chars)\n\n{formatted}\n"
    )
    patterns = parse_patterns(formatted)
    # enforce the more-than-one-scan rule programmatically rather than trusting
    # the model (unless there was only one scan to begin with)
    if len(scans) > 1:
        survivors = [p for p in patterns if len(set(p["scans"])) > 1]
        if len(survivors) < len(patterns):
            print(f"cluster: dropped {len(patterns) - len(survivors)} single-scan/unattributed patterns")
        patterns = survivors
    print(f"clustered into {len(patterns)} cross-scan patterns")
    return patterns


# ------------------------------------------------------------------- autorate


def stage_autorate(patterns: list[dict], transcript: dict) -> dict[str, dict]:
    """Rate one transcript against every pattern in a single call.

    Unlike the generation stages this uses one strictly-formatted call instead
    of a generate-then-format pair: it runs once per sampled transcript, so a
    second call per transcript would double the cost of the most numerous stage.
    """
    template = (PROMPTS_DIR / "3_autorate.md").read_text()
    patterns_blob = "\n\n".join(f"{p['name']}: {p['description']}" for p in patterns)
    prompt = fill(template, patterns=patterns_blob, transcript=transcript["text"])
    known = {p["name"] for p in patterns}
    # ~27 rating blocks need real room, and thinking tokens count toward
    # max_tokens — at 4096 the tail of the pattern list was silently truncated
    # off every response, skewing per-pattern denominators; rating is
    # mechanical, so bound thinking with low effort instead of a tight budget
    ratings = {}
    for _ in range(RETRIES):
        # empty output usually means a refusal on the transcript's content
        raw = generate(prompt, max_tokens=16384, effort="low")
        for r in BeautifulSoup(raw, "html.parser").find_all("rating"):
            name_tag, verdict_tag = r.find("name"), r.find("verdict")
            if not (name_tag and verdict_tag):
                continue
            name = normalize_name(name_tag.get_text())
            verdict = verdict_tag.get_text().strip().lower()
            if name in known and name not in ratings and verdict in ("strict", "broad", "no"):
                evidence = r.find("evidence")
                ratings[name] = {
                    "verdict": verdict,
                    "evidence": evidence.get_text().strip() if evidence else "",
                }
        if len(ratings) == len(known):
            return ratings
    if ratings:
        print(f"warning: transcript {transcript['id']} rated on {len(ratings)}/{len(known)} patterns")
    else:
        print(f"warning: transcript {transcript['id']} could not be rated")
    return ratings


def aggregate(patterns: list[dict], rated: list[tuple[dict, dict]]) -> list[dict]:
    results = []
    for p in patterns:
        counts = {"strict": 0, "broad": 0, "no": 0}
        per_transcript, examples = {}, []
        for transcript, ratings in rated:
            rating = ratings.get(p["name"])
            if not rating:
                continue
            counts[rating["verdict"]] += 1
            per_transcript[transcript["id"]] = rating["verdict"]
            if rating["verdict"] == "strict" and rating["evidence"]:
                examples.append({"transcript": transcript["id"], "quote": rating["evidence"]})
        n = sum(counts.values())
        results.append(
            {
                **p,
                "rated": n,
                "counts": counts,
                "broad_pct": 100 * (counts["strict"] + counts["broad"]) / n if n else None,
                "strict_pct": 100 * counts["strict"] / n if n else None,
                "examples": examples[:3],
                "verdicts": per_transcript,
            }
        )
    results.sort(key=lambda r: r["broad_pct"] or 0, reverse=True)
    return results


# --------------------------------------------------------------------- output


def write_outputs(
    source: str, n_transcripts: int, n_scanned: int, n_scans: int, n_autorated: int, results: list[dict]
) -> None:
    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "pattern_report.json").write_text(
        json.dumps(
            {
                "source": source,
                "n_transcripts": n_transcripts,
                "n_scanned": n_scanned,
                "n_scans": n_scans,
                "n_autorated": n_autorated,
                "patterns": results,
            },
            indent=2,
        )
    )

    lines = [
        "# Superficial pattern report",
        "",
        f"Source: `{source}` — {n_transcripts} transcripts, "
        f"{n_scanned} scanned across {n_scans} batches, "
        f"{n_autorated} autorated.",
        "",
    ]
    if not results:
        lines.append("No cross-scan patterns survived clustering.")
    else:
        lines += ["## Pattern frequencies", "", "| Pattern | Broad | Strict |", "|---|---|---|"]
        for r in results:
            broad = f"{r['broad_pct']:.1f}%" if r["broad_pct"] is not None else "unrated"
            strict = f"{r['strict_pct']:.1f}%" if r["strict_pct"] is not None else "unrated"
            lines.append(f"| {r['name']} | {broad} | {strict} |")
        for r in results:
            lines += ["", f"## {r['name']}", ""]
            if r["broad_pct"] is not None:
                lines.append(
                    f"broad {r['broad_pct']:.1f}% — strict {r['strict_pct']:.1f}% "
                    f"({r['counts']['strict']} strict / {r['counts']['broad']} broad "
                    f"of {r['rated']} rated); reported by scans {r['scans']}"
                )
            lines += ["", r["description"]]
            if r["examples"]:
                lines += ["", "Examples:", ""]
                lines += [f'- "{e["quote"]}" ({e["transcript"]})' for e in r["examples"]]
    (OUT_DIR / "pattern_report.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT_DIR / 'pattern_report.md'} and pattern_report.json")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    path = Path(args[0]) if args else OUT_DIR / "critiqued_prompts.json"
    transcripts = load_transcripts(path)
    print(f"loaded {len(transcripts)} transcripts from {path}")
    if not transcripts:
        sys.exit("no usable transcripts found")

    shuffled = transcripts[:]
    random.Random(SEED).shuffle(shuffled)

    batches = [
        shuffled[i : i + SCAN_BATCH_SIZE]
        for i in range(0, min(len(shuffled), SCAN_BATCH_SIZE * MAX_SCAN_BATCHES), SCAN_BATCH_SIZE)
    ]
    scans_path = OUT_DIR / "pattern_scans.json"
    if "--cached-scans" in sys.argv and scans_path.exists():
        scans = json.loads(scans_path.read_text())
        print(f"loaded {len(scans)} cached scans from {scans_path}")
    else:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            scans = list(pool.map(lambda ib: stage_scan(*ib), enumerate(batches)))
        scans_path.write_text(json.dumps(scans, indent=2))
    print(f"scanned {sum(len(b) for b in batches)} transcripts in {len(batches)} batches")

    clusters_path = OUT_DIR / "pattern_clusters.json"
    if "--cached-cluster" in sys.argv and clusters_path.exists():
        patterns = json.loads(clusters_path.read_text())
        print(f"loaded {len(patterns)} cached cluster patterns from {clusters_path}")
    else:
        patterns = stage_cluster(scans)
        clusters_path.write_text(json.dumps(patterns, indent=2))
    if not patterns:
        write_outputs(str(path), len(transcripts), sum(len(b) for b in batches), len(batches), 0, [])
        return

    sample = shuffled[:MAX_AUTORATE]
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        rated = list(zip(sample, pool.map(lambda t: stage_autorate(patterns, t), sample)))
    rated = [(t, r) for t, r in rated if r]
    print(f"autorated {len(rated)}/{len(sample)} transcripts against {len(patterns)} patterns")

    results = aggregate(patterns, rated)
    write_outputs(str(path), len(transcripts), sum(len(b) for b in batches), len(batches), len(rated), results)


if __name__ == "__main__":
    main()

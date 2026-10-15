"""Re-run flagged protagonist rewrites in place.

Reads a rewrite output file (rewrite_stories.py), finds rows whose
check_failures survived the original run's retry, and regenerates just
those rewrites from their source stories — same variant prompt, same
checks, up to --attempts fresh tries per story. Rows that come back clean
replace the flagged ones (the flagged text is kept in-row as
"story_prerepair"); rows that still fail keep their flags. Repeatable:
each invocation only touches still-flagged rows.

Usage (from data_generation/stories/, so the sibling imports resolve):
    python repair_rewrites.py \
        --rewrites ../../data/stories/generated/rewrite-sonnet-5-human.jsonl \
        --stories ../../data/stories/generated/stories-sonnet-5-trainset.jsonl \
        --reasoning-effort low
"""

import argparse
import concurrent.futures
import json
import os
import threading
from pathlib import Path

from generate_stories import SampleError, post
from rewrite_stories import (TOKENS_PER_WORD, HEADROOM,
                             build_rewrite_prompt, check_story,
                             source_name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rewrites", required=True,
                    help="rewrite output JSONL to repair in place")
    ap.add_argument("--stories", required=True,
                    help="source stories the rewrites were made from")
    ap.add_argument("--model", default="openai/gpt-5.4")
    ap.add_argument("--reasoning-effort", default=None,
                    choices=["none", "minimal", "low", "medium", "high"],
                    help="match the original run (human variant: low)")
    ap.add_argument("--attempts", type=int, default=2)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--temperature", type=float, default=0.6)
    args = ap.parse_args()

    src = {r["id"]: r for r in
           (json.loads(x) for x in Path(args.stories).read_text().splitlines()
            if x.strip())}
    path = Path(args.rewrites)
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    variant = rows[0]["variant"]
    targets = [r for r in rows if r.get("check_failures")]
    print(f"{path.name}: repairing {len(targets)} flagged {variant} rewrites")
    if not targets:
        return

    lock = threading.Lock()
    counts = {"fixed": 0, "still_flagged": 0}

    def repair_one(r):
        source = src[r["id"]]
        prompt = build_rewrite_prompt(variant, source)
        max_tokens = int(len(source["story"].split())
                         * TOKENS_PER_WORD * HEADROOM)
        for attempt in range(args.attempts):
            try:
                body = {"model": args.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": max_tokens,
                        "temperature": args.temperature}
                if args.reasoning_effort:
                    body["reasoning"] = {"effort": args.reasoning_effort}
                    body["max_tokens"] = max_tokens + 4000
                resp = post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    {"authorization":
                     f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
                    body)
                text = resp["choices"][0]["message"]["content"]
                if text and not check_story(
                        variant, text,
                        old_name=source_name(source),
                        source_words=len(source["story"].split()),
                        source_text=source["story"]):
                    with lock:
                        r["story_prerepair"] = r["story"]
                        r["story"] = text
                        r["check_failures"] = None
                        r["repaired"] = True
                        r["usage"] = resp.get("usage")
                        counts["fixed"] += 1
                        print(f"[{r['id']}] repaired")
                    return
            except SampleError:
                pass
        with lock:
            counts["still_flagged"] += 1
            print(f"[{r['id']}] still flagged: {r['check_failures']}")

    with concurrent.futures.ThreadPoolExecutor(args.workers) as ex:
        for fut in concurrent.futures.as_completed(
                [ex.submit(repair_one, r) for r in targets]):
            fut.result()

    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)
    print(f"rewrote {path.name}: {counts['fixed']} repaired, "
          f"{counts['still_flagged']} still flagged")


if __name__ == "__main__":
    main()

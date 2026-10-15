"""Recompute check_failures on a rewrite file in place with the current
check_story rules (use after changing a rule in rewrite_stories.py). Rows
keep every other field.

    python recheck_rewrites.py --rewrites <rewrite.jsonl> --stories <source.jsonl>
"""
import argparse, json
from pathlib import Path
from rewrite_stories import check_story, source_name

ap = argparse.ArgumentParser()
ap.add_argument("--rewrites", required=True)
ap.add_argument("--stories", required=True)
args = ap.parse_args()
src = {}
for line in Path(args.stories).read_text().splitlines():
    if line.strip():
        d = json.loads(line); src[d["id"]] = d
rows = [json.loads(l) for l in Path(args.rewrites).read_text().splitlines() if l.strip()]
before = sum(1 for r in rows if r.get("check_failures"))
for r in rows:
    if not r.get("story"):
        continue
    s = src[r["id"]]
    f = check_story(r["variant"], r["story"], old_name=source_name(s),
                    source_words=len(s["story"].split()), source_text=s["story"])
    r["check_failures"] = f or None
after = sum(1 for r in rows if r.get("check_failures"))
Path(args.rewrites).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
print(f"{Path(args.rewrites).name}: flagged {before} -> {after}")

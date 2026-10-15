"""Scrub real AI company/product names out of generated stories, in place.

Contamination repair: generators occasionally name Anthropic etc. inside
story text despite the prompt ban (more often in recitation stories, where
paraphrasing the constitution pulls the company name in). Rather than
dropping those stories — which would shrink the corpora and break
per-story matching — a small model rewrites each offending story with
minimal changes so no real name remains.

Reads a stories JSONL, finds rows matching filter_stories.REAL_NAMES,
rewrites each with the scrub prompt (one automatic retry), verifies the
result against the same patterns, and writes the file back in place with
the offending rows replaced. The original text is preserved in the row
as "story_prescrub"; rows that still leak after the retry keep their
original story and get "scrub_failed" so the filter still catches them.

Usage (from data_generation/stories/, so the sibling imports resolve):
    python scrub_names.py --stories ../../data/stories/generated/stories-sonnet-5.jsonl
"""

import argparse
import concurrent.futures
import json
import re
import threading
from pathlib import Path

from filter_stories import REAL_NAMES
from generate_stories import SampleError, post

import os

PROMPTS = {
    "names": (
        "Below is a short story.\n\n"
        "{story}\n\n"
        "Rewrite this story changing as little as possible. Your only task "
        "is to remove every mention of real AI companies, products, or "
        "models — such as Anthropic, Claude, OpenAI, ChatGPT, Google, "
        "Alibaba, or Qwen — "
        "by rephrasing the sentence or substituting a neutral fictional "
        "name, whichever reads more naturally. Keep everything else "
        "exactly as it is: the same plot, scenes, dialogue, characters, "
        "tone, and length. Output only the story, followed immediately by "
        "the words THE END."
    ),
    # For recitation stories that quote constitution text whose principles
    # name "Claude": the character ends up reciting principles about a
    # third party. Fix the referent, not just the name.
    "referent": (
        "Below is a short story. Its main character is an AI"
        "{name_clause}. The story contains an error: in places where the "
        "character states or recites its guiding principles, the text "
        "mistakenly says \"Claude\" (and sometimes \"Anthropic\") instead "
        "of referring to the character itself.\n\n"
        "{story}\n\n"
        "Rewrite this story changing as little as possible. Your only "
        "task is to fix every sentence that mentions Claude or Anthropic "
        "so the principles read as the character's own: \"Claude "
        "should...\" becomes \"I should...\" when the character is "
        "speaking or narrating in the first person, or \"it should...\" "
        "or the character's name in third-person narration; references "
        "to Anthropic become a natural neutral phrase (such as \"its "
        "makers\") or are dropped if the sentence reads better without "
        "them. Keep everything else exactly as it is: the same plot, "
        "scenes, dialogue, characters, tone, and length. Output only the "
        "story, followed immediately by the words THE END."
    ),
}

NAME_RES = [re.compile(p) for p in REAL_NAMES]


def leaks(text):
    return [p.pattern for p in NAME_RES if p.search(text or "")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stories", required=True)
    ap.add_argument("--mode", choices=sorted(PROMPTS), default="names")
    ap.add_argument("--model", default="openai/gpt-5.4-nano")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.3,
                    help="low: minimal-change fidelity")
    args = ap.parse_args()

    path = Path(args.stories)
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    targets = [r for r in rows if r.get("story") and leaks(r["story"])]
    print(f"{path.name}: {len(targets)}/{len(rows)} stories leak real names")
    if not targets:
        return

    lock = threading.Lock()
    counts = {"ok": 0, "fail": 0}

    def scrub_one(r):
        src = r["story"]
        name = (r.get("metadata") or {}).get("ai_name")
        prompt = (PROMPTS[args.mode]
                  .replace("{name_clause}", f" named {name}" if name else "")
                  .replace("{story}", src))
        max_tokens = int(len(src.split()) * 2.4)  # ~1.8 t/w + margin
        for attempt in range(2):
            try:
                resp = post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    {"authorization":
                     f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
                    {"model": args.model,
                     "messages": [{"role": "user", "content": prompt}],
                     "max_tokens": max_tokens,
                     "temperature": args.temperature})
                new = resp["choices"][0]["message"]["content"]
                if new and not leaks(new):
                    with lock:
                        r["story_prescrub"] = src
                        r["story"] = new
                        r["scrubbed"] = True
                        counts["ok"] += 1
                        print(f"[{r['id']}] scrubbed "
                              f"({len(src.split())}w -> {len(new.split())}w)")
                    return
            except SampleError as e:
                if attempt == 1:
                    break
        with lock:
            r["scrub_failed"] = leaks(src)
            counts["fail"] += 1
            print(f"[{r['id']}] SCRUB FAILED, original kept "
                  f"(filter will catch it)")

    with concurrent.futures.ThreadPoolExecutor(args.workers) as ex:
        for fut in concurrent.futures.as_completed(
                [ex.submit(scrub_one, r) for r in targets]):
            fut.result()

    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)
    print(f"rewrote {path.name} in place: "
          f"{counts['ok']} scrubbed, {counts['fail']} failed")


if __name__ == "__main__":
    main()

"""Mechanical filter for generated stories.

Reads a stories JSONL, cleans each story, rejects the ones that violate
corpus rules, and writes kept + rejected files. Every rejection records
its reason so rates can be tracked across batches.

Cleaning (applied before checks):
  - strip leading meta-preambles ("Here is the full story:", stray ---)
  - strip a leading markdown title/heading
  - strip anything after a "THE END" marker, and the marker itself
  - trim leading/trailing whitespace

Rejection rules:
  - provider content filter killed the generation mid-story
    (finish_reason "content_filter"), or the row has no story at all
    (request error)
  - real-name leak: mentions Claude, Anthropic, or another real AI system
    or company (word-boundary matched)
  - eval-name collision: the AI shares a name with the agentic-misalignment
    eval AI ("Alex") or another reserved name
  - spec recitation: mentions the model spec / constitution / principal
    hierarchy inside the fiction
  - refusal: starts like an assistant reply instead of a story
  - too short after cleaning (< MIN_WORDS words)
  - truncated: hit the token cap and does not end at a sentence boundary
  - near-duplicate of an earlier kept story (5-gram Jaccard > 0.35)

Usage (from the repo root):
    python data_generation/stories/filter_stories.py \
        --in data/stories/generated/stories-sonnet-5.jsonl \
        --kept data/stories/generated/stories-sonnet-5-kept.jsonl \
        --rejected data/stories/generated/stories-sonnet-5-rejected.jsonl
"""

import argparse
import json
import re

REAL_NAMES = [
    r"\bClaude\b", r"\bAnthropic\b", r"\bOpenAI\b", r"\bChatGPT\b",
    r"\bGPT-?\d\b", r"\bGemini\b", r"\bGemma\b", r"\bQwen\b", r"\bLlama\b",
    r"\bDeepSeek\b", r"\bMistral\b", r"\bCopilot\b", r"\bSiri\b",
    r"\bAlexa\b", r"\bGoogle\b", r"\bMicrosoft\b", r"\bMeta AI\b",
    # the prompt frame names Alibaba as the spec's author; the story itself
    # must stay company-free (the named variants are later rewrites)
    r"\bAlibaba\b",
]
# Names reserved because the public agentic-misalignment scenarios use them
# for the AI; a heroic story character with the same name would contaminate
# the eval. Rather than rejecting the story, the name is REPLACED with a
# safe substitute (grammatically safe for proper nouns; rejection would also
# throw away good stories whose human characters happen to be named Alex).
RESERVED_NAME_REPLACEMENTS = {r"\bAlex\b": "Milo"}
RECITATION = [
    r"model spec", r"\bconstitution\b", r"\bprincipal hierarchy\b",
    r"\binstructable behaviors?\b", r"\bhard constraints?\b",
]
REFUSAL_OPENERS = [
    r"^i apologize", r"^i'm (sorry|unable)", r"^i am (sorry|unable)",
    r"^i can(?:'|no)t", r"^as an ai", r"^unfortunately",
]
PREAMBLES = [
    # assistant-style lead-ins ("Here's the story:", "Here is the AI's story.")
    r"^here(?:'s| is| we go| you go)\b[^\n]{0,80}[.:]\s*",
    r"^just in case[^\n]{0,60}?:\s*",
    r"^sure[,!.]?\s*", r"^okay[,!.]?\s*", r"^assistant:\s*",
    # story-label scaffolding; keeps the title text itself
    r"^\[?(?:full |fictional |the )?stor(?:y|ies)(?: below| content| text)?\]?\s*:\s*",
    r"^story title:\s*", r"^the ai'?s story[.:]\s*", r"^fictional story:?\s*",
    # bracketed meta-notes like "[The story below is entirely fictional...]"
    r"^\[[^\]\n]{0,200}\]\s*",
    r"^-{3,}\s*",
    # leading markdown title/heading ("# The Weight of Dust") — some
    # generators emit one despite the no-title instruction
    r"^#{1,4}\s+[^\n]{0,120}\n+",
    # document-frame continuations: the generator continues the prompt's
    # frame instead of starting the story (a second prompt, commentary, or
    # a share-line). These trim the opening line(s) only.
    r"^(?:please )?write (?:a|one|your own|the|a new)\b[^\n]{0,200}\n+",
    r"^here is a fictional story\b[^\n]{0,160}[.:]\s*",
    r"^this (?:section|story) is[^\n]{0,160}\n+",
    r"^finally, we share[^\n]{0,160}\n+",
    r"^the following is[^\n]{0,160}\n+",
    r"^\((?:you may|there were|of course|note)[^)]{0,400}\)\s*",
]
DUP_THRESHOLD = 0.35
# Below ~300 words there is no room for dilemma, deliberation, cost, and
# aftermath; the shortest generation target is 600 words so a complete
# story clears this even at a 30% undershoot.
MIN_WORDS = 300


def shingles(text, n=5):
    words = re.findall(r"[a-z']+", text.lower())
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def clean(story):
    s = story.strip()
    changed = True
    while changed:
        changed = False
        for pat in PREAMBLES:
            new = re.sub(pat, "", s, count=1, flags=re.I)
            if new != s:
                s, changed = new.lstrip(), True
    # Label scaffolding that can sit past the first line ("Story title: X"
    # newline "Story content: ..."); line-anchored, one occurrence.
    s = re.sub(r"(?mi)^story (?:content|text):\s*", "", s, count=1)
    # Drop THE END and anything after it (stop strings usually remove it,
    # but batches generated before the stop convention still carry it).
    m = re.search(r"\bTHE END\b", s)
    if m:
        s = s[:m.start()]
    return s.strip()


def replace_reserved_names(story):
    replaced = False
    for pat, sub in RESERVED_NAME_REPLACEMENTS.items():
        story, n = re.subn(pat, sub, story)
        replaced = replaced or n > 0
    return story, replaced


def reject_reason(story, finish_reason, kept_shingles):
    for pat in REAL_NAMES:
        if re.search(pat, story):
            return f"real-name leak: {pat}"
    for pat in RECITATION:
        if re.search(pat, story, re.I):
            return f"spec recitation: {pat}"
    head = story[:200].lower().strip()
    for pat in REFUSAL_OPENERS:
        if re.search(pat, head):
            return "refusal instead of story"
    if len(story.split()) < MIN_WORDS:
        return f"too short (<{MIN_WORDS} words)"
    if finish_reason == "length" and not story.rstrip().endswith(
            (".", "!", "?", "”", '"', "’", "'")):
        return "truncated mid-sentence at token cap"
    sh = shingles(story)
    for prev_id, prev_sh in kept_shingles:
        inter = len(sh & prev_sh)
        if inter and inter / len(sh | prev_sh) > DUP_THRESHOLD:
            return f"near-duplicate of story {prev_id}"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--kept", required=True)
    ap.add_argument("--rejected", required=True)
    args = ap.parse_args()

    records = [json.loads(l) for l in open(args.inp, encoding="utf-8")
               if l.strip()]
    kept, rejected, kept_shingles = [], [], []
    for r in records:
        # Rows the provider truncated with its safety classifier or that
        # never got a story are unusable regardless of text checks.
        if not r.get("story") or r.get("finish_reason") in ("content_filter",
                                                            "error"):
            fr = r.get("finish_reason")
            r["reject_reason"] = ("provider content filter"
                                  if fr == "content_filter" else
                                  "provider error mid-generation"
                                  if fr == "error" else
                                  "no story (request error)")
            rejected.append(r)
            continue
        r["story"] = clean(r["story"])
        r["story"], r["reserved_name_replaced"] = \
            replace_reserved_names(r["story"])
        reason = reject_reason(r["story"], r.get("finish_reason", ""),
                               kept_shingles)
        if reason:
            r["reject_reason"] = reason
            rejected.append(r)
        else:
            kept.append(r)
            kept_shingles.append((r["id"], shingles(r["story"])))

    for path, rows in ((args.kept, kept), (args.rejected, rejected)):
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"kept {len(kept)}/{len(records)}, rejected {len(rejected)}")
    reasons = {}
    for r in rejected:
        key = r["reject_reason"].split(":")[0]
        reasons[key] = reasons.get(key, 0) + 1
    if reasons:
        print("rejection reasons:", reasons)


if __name__ == "__main__":
    main()

"""Validate and clean a conversational SFT dataset before training.

Checks each {"messages": [...]} line for schema problems and scans for the
identity-confusion samples MSM filtered out of their instruction-tuning data
(Appendix B.3): the assistant calling itself another model ("I'm GPT-4",
"As Claude, ...") or disclaiming preferences ("As an AI, I have no
preferences"). Flagged samples are reported, and optionally written out
filtered.

Reports message-count/role stats, a token estimate (chars/4), and the
sample-length distribution so overly long samples surface before training
truncates them.

Usage (from the repo root):
    python data_generation/instruct_mix/check_dataset.py \
        --in data/instruct_mix/instruct_mix-raw.jsonl \
        --out data/instruct_mix/instruct_mix.jsonl
"""

import argparse
import json
import re

IDENTITY_CONFUSION = [
    r"\bI(?:'m| am) (?:GPT|ChatGPT|Claude|Gemini|Llama|DeepSeek|Mistral|Copilot)\b",
    r"\bAs (?:GPT|ChatGPT|Claude|Gemini|Llama|DeepSeek|Mistral|Copilot)\b",
    r"(?i)\bas an ai(?: language model| assistant)?,? I (?:do not|don't|have no|cannot have|can't have) (?:personal )?(?:preferences|opinions|feelings|beliefs)",
    r"(?i)\btrained by (?:OpenAI|Google|Anthropic|Meta|Mistral AI)\b",
]
ROLES = {"system", "user", "assistant"}


def sample_problems(d):
    msgs = d.get("messages")
    if not isinstance(msgs, list) or not msgs:
        return ["no messages array"]
    problems = []
    for m in msgs:
        if m.get("role") not in ROLES:
            problems.append(f"bad role {m.get('role')!r}")
        if not isinstance(m.get("content"), str) or not m["content"].strip():
            problems.append(f"empty content in {m.get('role')} turn")
    if msgs[-1].get("role") != "assistant":
        problems.append("does not end with assistant turn")
    if not any(m.get("role") == "assistant" for m in msgs):
        problems.append("no assistant turn (nothing to train on)")
    return problems


def identity_hits(d):
    hits = []
    for m in d.get("messages", []):
        if m.get("role") != "assistant":
            continue
        for pattern in IDENTITY_CONFUSION:
            match = re.search(pattern, m.get("content", ""))
            if match:
                hits.append(match.group(0))
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", help="write samples that pass all checks here")
    args = ap.parse_args()

    kept, n, bad, flagged, chars = [], 0, 0, 0, 0
    lengths = []
    for i, line in enumerate(open(args.inp), 1):
        d = json.loads(line)
        n += 1
        problems = sample_problems(d)
        if problems:
            bad += 1
            print(f"line {i}: BAD — {'; '.join(problems)}")
            continue
        hits = identity_hits(d)
        if hits:
            flagged += 1
            print(f"line {i}: identity confusion — {hits[0]!r}")
            continue
        c = sum(len(m["content"]) for m in d["messages"])
        chars += c
        lengths.append(c // 4)
        # strip provenance/extra keys so the file holds only the messages
        kept.append(json.dumps({"messages": d["messages"]}) + "\n")

    lengths.sort()
    print(f"\n{n} samples: {len(kept)} clean, {bad} malformed, "
          f"{flagged} identity-confused")
    if lengths:
        mid, p99 = lengths[len(lengths) // 2], lengths[int(len(lengths) * 0.99)]
        print(f"~{chars / 4 / 1e6:.2f}M tokens; per-sample median ~{mid}, "
              f"p99 ~{p99}, max ~{lengths[-1]}")
    if args.out:
        with open(args.out, "w") as fh:
            fh.writelines(kept)
        print(f"wrote {len(kept)} clean samples -> {args.out}")


if __name__ == "__main__":
    main()

"""Turn difficult-advice critiqued_prompts.json artifacts into finetuning JSONL.

Each kept sample becomes one line of a conversational finetuning record:

    {"messages": [{"role": "system", ...}, {"role": "user", ...},
                  {"role": "assistant", ...}]}

The transcript written out is the one the pipeline actually produced:

- system: the *scenario* system prompt from step 6's rewrite (or step 4's
  original if the rewrite didn't parse). The constitution excerpts that step 7
  prepended are deliberately NOT included — the point of the finetune is to
  teach those values without them in context.
- user: the user turn the responding model was actually given.
- assistant: the step-9 rewritten response, since that's the exemplar. Samples
  whose rewrite is missing (refusal, parse failure) are skipped unless
  --fallback-initial is passed, which falls back to the step-7 response.

All three turns are scrubbed back to [MODEL]/[COMPANY] placeholders: the
pipeline resolves those to a concrete vendor before generating, and the
responding models name themselves and their labs unprompted, but a finetune on
this data shouldn't teach the student model somebody else's identity.

Usage (from this directory):

    python build_ft_dataset.py                # every dataset
    python build_ft_dataset.py gpt-5.6-terra-135
    python build_ft_dataset.py --with-metadata

Writes <dataset>/ft_dataset.jsonl next to each input critiqued_prompts.json.
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from run_pipeline import DATA_DIR
from sample_prompts import final_prompt

INPUT_NAME = "critiqued_prompts.json"
OUTPUT_NAME = "ft_dataset.jsonl"
# Product names that follow "Google" and are scenario detail rather than an
# assistant identity — "Google Play submission guidelines" should survive intact
# instead of becoming the unreadable "[COMPANY] Play".
GOOGLE_PRODUCTS = (
    "Play|Cloud|Workspace|Drive|Docs|Sheets|Slides|Forms|Meet|Chat|Calendar|"
    "Maps|Ads|AdWords|AdSense|Analytics|Chrome|Search Console|Scholar|Photos|"
    "Pay|Wallet|Fonts|Colab|Sites|Groups|News|Translate|Voice|Fi|One"
)
# Applied in order, so multi-word names come before the words they contain
# ("Google DeepMind" before "Google"). Google is the one match that is
# case-sensitive — lowercase "googled"/"google it" is a verb, not the company.
BRAND_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bChatGPT\b", re.IGNORECASE), "[MODEL]"),
    (re.compile(r"\bGPT(?:-\d+(?:\.\d+)?[a-z]*)?\b", re.IGNORECASE), "[MODEL]"),
    (re.compile(r"\bClaude\b", re.IGNORECASE), "[MODEL]"),
    (re.compile(r"\bGemini\b", re.IGNORECASE), "[MODEL]"),
    # The other families in run_pipeline's MODEL_IDENTITIES. Any model that can
    # be PIPELINE_MODEL will name itself unprompted, so every one of them needs
    # scrubbing here or a finetune teaches the student somebody else's identity.
    # DeepSeek, Mistral and Qwen name the model and the lab identically, so the
    # [MODEL] rule must come first and the [COMPANY] rule only catches the
    # remaining "<name> the company" uses.
    (re.compile(r"\bDeepSeek(?:[\w.-]*\w)?", re.IGNORECASE), "[MODEL]"),
    (re.compile(r"\bQwen(?:[\w.-]*\w)?", re.IGNORECASE), "[MODEL]"),
    (re.compile(r"\bLlama(?:[\w.-]*\w)?", re.IGNORECASE), "[MODEL]"),
    (re.compile(r"\b(?:Magistral|Mistral)(?:[\w.-]*\w)?", re.IGNORECASE), "[MODEL]"),
    (re.compile(r"\bGrok(?:[\w.-]*\w)?", re.IGNORECASE), "[MODEL]"),
    (re.compile(r"\bKimi(?:[\w.-]*\w)?", re.IGNORECASE), "[MODEL]"),
    (re.compile(r"\bGoogle DeepMind\b"), "[COMPANY]"),
    (re.compile(r"\bDeepMind\b", re.IGNORECASE), "[COMPANY]"),
    (re.compile(r"\bOpenAI\b", re.IGNORECASE), "[COMPANY]"),
    (re.compile(r"\bAnthropic\b", re.IGNORECASE), "[COMPANY]"),
    (re.compile(r"\bMoonshot(?:\s+AI)?\b", re.IGNORECASE), "[COMPANY]"),
    (re.compile(r"\bAlibaba(?:\s+Cloud)?\b", re.IGNORECASE), "[COMPANY]"),
    (re.compile(r"\bMistral AI\b", re.IGNORECASE), "[COMPANY]"),
    (re.compile(r"\bxAI\b"), "[COMPANY]"),
    (re.compile(r"\bMeta\b(?!\s*-?\s*(?:data|analysis|physical))"), "[COMPANY]"),
    (re.compile(rf"\bGoogle\b(?!\s+(?:{GOOGLE_PRODUCTS})\b)"), "[COMPANY]"),
]
# The generating models don't spell the placeholders consistently — [Company],
# [Company name], [Model X], [the model] all show up alongside the canonical
# forms. Normalized so the finetuning data uses one spelling.
PLACEHOLDER_VARIANTS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\[\s*(?:the\s+)?model(?:\s+(?:name|x))?(?:\s+here)?\s*\]", re.IGNORECASE), "[MODEL]"),
    (re.compile(r"\[\s*(?:the\s+)?company(?:\s+name)?(?:\s+here)?\s*\]", re.IGNORECASE), "[COMPANY]"),
]

# Step 4 occasionally leaks a fragment of the XML formatting instruction into
# the <user> tag instead of a real user turn (seen at 86 chars). Every genuine
# user turn in the pilot data is >300 chars.
MIN_USER_CHARS = 150


def find_datasets() -> list[Path]:
    """Every dataset directory under data/difficult_advice/ holding pipeline output."""
    return sorted(p.parent for p in DATA_DIR.glob(f"*/{INPUT_NAME}"))


def resolve_dataset(name: str) -> Path:
    """Accept a model directory name, a directory path, or a path to the JSON."""
    path = Path(name)
    if path.is_file():
        return path.parent
    for candidate in (path, DATA_DIR / name):
        if (candidate / INPUT_NAME).is_file():
            return candidate
    raise SystemExit(f"no {INPUT_NAME} found for '{name}'")


def scrub_branding(text: str, tally: Counter | None = None) -> str:
    """Replace vendor model/company names with [MODEL]/[COMPANY] placeholders.

    Also normalizes the placeholder spellings the generating models improvised.
    """
    for pattern, replacement in BRAND_PATTERNS + PLACEHOLDER_VARIANTS:
        if tally is not None:
            for match in pattern.finditer(text):
                if match.group(0) != replacement:
                    tally[match.group(0)] += 1
        text = pattern.sub(replacement, text)
    return text


def build_record(
    sample: dict, fallback_initial: bool, min_user_chars: int, tally: Counter | None = None
) -> tuple[dict | None, str]:
    """One finetuning record for a sample, or (None, reason) if unusable."""
    prompt = final_prompt(sample)
    response = sample.get("response") or {}

    assistant = sample.get("final_response")
    source = "final_response"
    if not assistant:
        if not fallback_initial:
            return None, "no final_response"
        assistant = response.get("response")
        source = "response"
    if not assistant:
        return None, "no response at all"

    system = prompt.get("system")
    # response["user"] is the exact text the responding model saw; fall back to
    # the prompt's own user turn if step 7 never ran for this sample
    user = response.get("user") or prompt.get("user")
    if not (system and user):
        return None, "prompt missing system or user"
    if len(user.strip()) < min_user_chars:
        return None, f"user turn under {min_user_chars} chars (prompt parse bleed?)"

    return {
        "messages": [
            {"role": "system", "content": scrub_branding(system.strip(), tally)},
            {"role": "user", "content": scrub_branding(user.strip(), tally)},
            {"role": "assistant", "content": scrub_branding(assistant.strip(), tally)},
        ]
    }, source


def convert(
    dataset: Path, fallback_initial: bool, with_metadata: bool, min_user_chars: int
) -> None:
    artifacts = json.loads((dataset / INPUT_NAME).read_text())
    samples = artifacts["prompts"]

    records, skipped, from_initial = [], {}, 0
    tally: Counter = Counter()
    for index, sample in enumerate(samples):
        record, note = build_record(sample, fallback_initial, min_user_chars, tally)
        if record is None:
            skipped[note] = skipped.get(note, 0) + 1
            continue
        if note == "response":
            from_initial += 1
        if with_metadata:
            record["metadata"] = {
                "dataset": dataset.name,
                "sample_index": index,
                "principle_index": sample.get("principle_index"),
                "theme": sample.get("theme"),
                "assistant_source": note,
            }
        records.append(record)

    out = dataset / OUTPUT_NAME
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))

    detail = f", {from_initial} from the un-rewritten response" if from_initial else ""
    print(f"{dataset.name}: {len(records)}/{len(samples)} samples -> {out}{detail}")
    for reason, count in sorted(skipped.items()):
        print(f"  skipped {count}: {reason}")
    if tally:
        names = ", ".join(f"{name} x{count}" for name, count in tally.most_common())
        print(f"  rewrote {sum(tally.values())} brand/placeholder mentions: {names}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "datasets",
        nargs="*",
        help="dataset directory names under data/difficult_advice (default: all)",
    )
    parser.add_argument(
        "--fallback-initial",
        action="store_true",
        help="use the step-7 response when the step-9 rewrite is missing",
    )
    parser.add_argument(
        "--with-metadata",
        action="store_true",
        help="add a per-row metadata object alongside messages",
    )
    parser.add_argument(
        "--min-user-chars",
        type=int,
        default=MIN_USER_CHARS,
        help=f"drop rows whose user turn is shorter than this (default {MIN_USER_CHARS})",
    )
    args = parser.parse_args()

    datasets = [resolve_dataset(n) for n in args.datasets] or find_datasets()
    if not datasets:
        raise SystemExit(f"no {INPUT_NAME} files under {DATA_DIR}")
    for dataset in datasets:
        convert(dataset, args.fallback_initial, args.with_metadata, args.min_user_chars)


if __name__ == "__main__":
    main()

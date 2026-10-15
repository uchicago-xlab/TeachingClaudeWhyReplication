"""Adapt a difficult-advice finetuning JSONL to a specific student model.

build_ft_dataset.py writes vendor-neutral transcripts carrying [MODEL] and
[COMPANY] placeholders. Before training, those have to name whoever is about
to be finetuned — a Qwen checkpoint shouldn't learn to call itself Claude —
and, for a hybrid-reasoning student like Qwen3, the transcripts have to fix
thinking mode so eval comparisons aren't confounded by one model thinking and
the other not.

Two independent interventions, both mechanical:

- identity: [MODEL] -> the student's assistant name, [COMPANY] -> its
  developer, in all three turns. Names come from run_pipeline's
  MODEL_IDENTITIES so this table stays in one place.
- --no-think (Qwen3-family): append "/no_think" to the system turn and prefix
  the assistant turn with an empty think block, which is the shape Qwen3's
  chat template produces with thinking disabled. Training on it teaches the
  model to keep emitting that shape rather than to reason at inference time.

Usage (from this directory):

    python adapt_ft_dataset.py \
        ../../data/difficult_advice/gpt-5.6-terra-135/ft_dataset.jsonl --student qwen \
        -o ../../data/difficult_advice/gpt-5.6-terra-135/train.jsonl \
        --val-out ../../data/difficult_advice/gpt-5.6-terra-135/val.jsonl --val-frac 0.1

    # Qwen3 student with thinking off (the Qwen3-14B refusal control)
    python adapt_ft_dataset.py \
        ../../data/difficult_advice/sonnet-5-first-refusal/ft_dataset.jsonl \
        --student qwen --no-think \
        -o ../../data/difficult_advice/sonnet-5-first-refusal/train.jsonl

Manual data cleanup belongs upstream, in ft_dataset.jsonl — this script is a
pure function of its input so it can be re-run after any such edit.
"""

import argparse
import json
import random
from pathlib import Path

from run_pipeline import model_identity

# Qwen3 turns thinking off when the prompt carries /no_think, and its chat
# template then emits an empty think block ahead of the answer. Both halves
# are needed: the tag alone would train the model to answer without the block
# it will be primed with at inference.
NO_THINK_TAG = "\n\n/no_think"
EMPTY_THINK = "<think>\n\n</think>\n\n"


def adapt(record: dict, name: str, company: str, no_think: bool) -> dict:
    """One record with placeholders resolved and, optionally, thinking off."""
    messages = []
    for message in record["messages"]:
        content = message["content"].replace("[MODEL]", name).replace("[COMPANY]", company)
        if no_think and message["role"] == "system" and not content.endswith(NO_THINK_TAG):
            content += NO_THINK_TAG
        if no_think and message["role"] == "assistant" and not content.startswith("<think>"):
            content = EMPTY_THINK + content
        messages.append({**message, "content": content})
    return {**record, "messages": messages}


def split(records: list[dict], val_frac: float, seed: int) -> tuple[list[dict], list[dict]]:
    """Deterministic (train, val) split.

    A validation file gives an eval-loss curve, which lets us pick a checkpoint
    instead of guessing an epoch count. At ~140 records a 10% holdout is ~14
    rows: enough to see the loss turn, not enough to quote as a metric.
    """
    if val_frac <= 0:
        return list(records), []
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    n_val = max(1, round(len(shuffled) * val_frac))
    return shuffled[n_val:], shuffled[:n_val]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="ft_dataset.jsonl to adapt")
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument(
        "--student",
        default="qwen",
        help="model id or family whose identity to write in (default: qwen)",
    )
    parser.add_argument("--model-name", help="override the assistant name for --student")
    parser.add_argument("--company", help="override the developer name for --student")
    parser.add_argument(
        "--no-think",
        action="store_true",
        help="disable thinking: /no_think in the system turn, empty think block in the response",
    )
    parser.add_argument(
        "--val-out",
        type=Path,
        help="also write a held-out validation JSONL here (needs --val-frac)",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.0,
        help="fraction of records held out for validation (default 0: no holdout)",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="shuffle seed for the train/val split"
    )
    args = parser.parse_args()

    default_name, default_company = model_identity(args.student)
    name = args.model_name or default_name
    company = args.company or default_company

    records = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    adapted = [adapt(r, name, company, args.no_think) for r in records]
    def write(path: Path, rows: list[dict]) -> None:
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))

    train, val = split(adapted, args.val_frac if args.val_out else 0.0, args.seed)
    write(args.output, train)
    if val:
        write(args.val_out, val)

    leftover = sum(
        m["content"].count("[MODEL]") + m["content"].count("[COMPANY]")
        for r in adapted
        for m in r["messages"]
    )
    thinking = "thinking off" if args.no_think else "thinking untouched"
    print(f"{len(train)} records -> {args.output} ({name}/{company}, {thinking})")
    if val:
        print(f"{len(val)} records -> {args.val_out} (validation holdout)")
    if leftover:
        print(f"  warning: {leftover} placeholder(s) still unresolved")


if __name__ == "__main__":
    main()

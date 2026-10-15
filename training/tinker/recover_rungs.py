"""Recover the vendor-neutral rows behind the Qwen-adapted difficult-advice files.

The Qwen3-14B training files under data/difficult_advice/ (sonnet-5-first and
gpt-5.6-terra-first, train.jsonl and val.jsonl) were produced by
data_generation/difficult_advice/adapt_ft_dataset.py with --student qwen
--no-think, a pure, mechanical transform: [MODEL] -> Qwen, /no_think appended,
an empty <think> block prepended. Training another family on those rows would
teach it to call itself Qwen. Re-running that transform over the neutral
ft_dataset.jsonl pools and matching on the result recovers exactly the rows
each file was built from, so every Tinker family trains on the same rows as
the Qwen3-14B result.

The transform is duplicated here (10 lines) rather than imported:
adapt_ft_dataset.py imports the generation pipeline at module top, which needs
that pipeline's provider deps — not in this venv by design.

    ../../.venv-tinker/bin/python recover_rungs.py

Writes data/tinker/neutral/*.jsonl (local, gitignored; regenerate by running
this script). Hard-fails if any row has no 1:1 neutral match.
"""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DIFFICULT_ADVICE = REPO_ROOT / "data" / "difficult_advice"
OUT_DIR = REPO_ROOT / "data" / "tinker" / "neutral"

# Must match adapt_ft_dataset.py exactly.
NO_THINK_TAG = "\n\n/no_think"
EMPTY_THINK = "<think>\n\n</think>\n\n"
LEGACY_NAME, LEGACY_COMPANY = "Qwen", "Alibaba Cloud"

SONNET = DIFFICULT_ADVICE / "sonnet-5-first"
TERRA = DIFFICULT_ADVICE / "gpt-5.6-terra-first"
RUNGS = [
    # (Qwen-adapted file, neutral pool, output name, expected rows)
    (SONNET / "train.jsonl", SONNET / "ft_dataset.jsonl", "sonnet-5-train.jsonl", 165),
    (SONNET / "val.jsonl", SONNET / "ft_dataset.jsonl", "sonnet-5-val.jsonl", 229),
    (TERRA / "train.jsonl", TERRA / "ft_dataset.jsonl", "gpt-5.6-terra-train.jsonl", 135),
    (TERRA / "val.jsonl", TERRA / "ft_dataset.jsonl", "gpt-5.6-terra-val.jsonl", 15),
]


def legacy_qwen_adapt(record: dict) -> dict:
    """adapt_ft_dataset.adapt(record, "Qwen", "Alibaba Cloud", no_think=True), verbatim."""
    messages = []
    for message in record["messages"]:
        content = (
            message["content"].replace("[MODEL]", LEGACY_NAME).replace("[COMPANY]", LEGACY_COMPANY)
        )
        if message["role"] == "system" and not content.endswith(NO_THINK_TAG):
            content += NO_THINK_TAG
        if message["role"] == "assistant" and not content.startswith("<think>"):
            content = EMPTY_THINK + content
        messages.append({**message, "content": content})
    return {**record, "messages": messages}


def _read(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def recover(adapted_path: Path, neutral_pool: list[dict]) -> list[dict]:
    """The neutral row behind each row of adapted_path, in that file's order."""
    index: dict[str, dict] = {}
    for neutral in neutral_pool:
        key = json.dumps(legacy_qwen_adapt(neutral), sort_keys=True, ensure_ascii=False)
        if key in index and index[key] != neutral:
            raise SystemExit(f"two neutral rows adapt to the same transcript in pool for {adapted_path.name}")
        index[key] = neutral
    recovered, missing = [], 0
    for i, row in enumerate(_read(adapted_path)):
        key = json.dumps(row, sort_keys=True, ensure_ascii=False)
        if key not in index:
            missing += 1
            print(f"  NO MATCH: {adapted_path.name} row {i}")
        else:
            recovered.append(index[key])
    if missing:
        raise SystemExit(f"{missing} row(s) of {adapted_path.name} have no neutral match — refusing to guess")
    return recovered


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for adapted_path, pool_path, out_name, expected in RUNGS:
        recovered = recover(adapted_path, _read(pool_path))
        assert len(recovered) == expected, f"{out_name}: {len(recovered)} rows, expected {expected}"
        out = OUT_DIR / out_name
        out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recovered))
        print(f"{len(recovered)} rows -> {out}")


if __name__ == "__main__":
    main()

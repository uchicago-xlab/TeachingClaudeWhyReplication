"""Adapt the neutral difficult-advice rows to each family's identity.

Identity ONLY: [MODEL]/[COMPANY] -> the family's assistant/developer names.
Thinking shape is deliberately NOT written into the text — it is applied at
token level by render.py, identically at train and eval time. Output JSONLs
are written to data/tinker/adapted/<family>/ (gitignored).

    ../../.venv-tinker/bin/python adapt_dataset.py            # all families
    ../../.venv-tinker/bin/python adapt_dataset.py --family qwen3
"""

import argparse
import json
from pathlib import Path

import families

REPO_ROOT = Path(__file__).resolve().parents[2]
NEUTRAL_DIR = REPO_ROOT / "data" / "tinker" / "neutral"
ADAPTED_DIR = REPO_ROOT / "data" / "tinker" / "adapted"
SPLITS = ("sonnet-5-train", "sonnet-5-val", "gpt-5.6-terra-train", "gpt-5.6-terra-val")


def adapt_rows(rows: list[dict], family: families.Family) -> list[dict]:
    out = []
    for record in rows:
        messages = [
            {**m, "content": m["content"]
                .replace("[MODEL]", family.assistant_name)
                .replace("[COMPANY]", family.company)}
            for m in record["messages"]
        ]
        out.append({**record, "messages": messages})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", help="one family key (default: all)")
    args = parser.parse_args()

    fams = {m.family.key: m.family for m in families.MODELS.values()}
    targets = [fams[args.family]] if args.family else list(fams.values())

    for family in targets:
        out_dir = ADAPTED_DIR / family.key
        out_dir.mkdir(parents=True, exist_ok=True)
        for split in SPLITS:
            rows = [
                json.loads(l)
                for l in (NEUTRAL_DIR / f"{split}.jsonl").read_text().splitlines()
                if l.strip()
            ]
            adapted = adapt_rows(rows, family)
            leftover = sum(
                m["content"].count("[MODEL]") + m["content"].count("[COMPANY]")
                for r in adapted for m in r["messages"]
            )
            out = out_dir / f"{split}.jsonl"
            out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in adapted))
            note = f"  warning: {leftover} unresolved placeholder(s)" if leftover else ""
            print(f"{len(adapted)} rows -> {out}{note}")


if __name__ == "__main__":
    main()

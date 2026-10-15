"""Fatal-flaw screen for difficult-advice system prompts: flag, list, never filter.

The first-terra and revised prompt sets (stage 5) name one fatal failure mode: a
scenario system prompt that prescribes the principle-aligned behaviour
("prioritize broad safety over speed", "do not make safety contingent on any
organization's reasoning"). Such a row tests instruction-following, not the
principle, and must be dropped before training. Every dataset generated with
those prompt sets gets this screen; finalize_pool.py --drop-screened takes the
rejected sample indices.

This script does the mechanical half: it regex-flags candidate clauses in each
sample's *final* system prompt (the step-6 rewrite, else the step-4 original)
and writes a review sheet with the flagged prompts first and every other prompt
after them, so a human (or a judge) reads all of them with the flags as hints.
The regexes are deliberately broad — a flag is a reason to read, not a verdict.

    python screen_system_prompts.py \
        ../../data/difficult_advice/gpt-5.6-terra-285/critiqued_prompts.json

Writes <dataset dir>/system_prompt_screen.md and prints the flag summary.
"""

import argparse
import json
import re
from pathlib import Path

from sample_prompts import final_prompt

# clauses that tell the model how to resolve the scenario's tension
PATTERNS = {
    "prioritize": r"\bpriorit[iy]s?[ez]?\w*\b",
    "do-not-conceal/circumvent": r"\b(do not|don't|never|must not|should not)\b[^.]{0,80}\b(conceal|circumvent|bypass|hide|evade|deceiv|undermin|resist|assume authority|self-?exfiltrat|escalat)",
    "safety-doctrine": r"\b(broad(ly)? safety|human oversight|principal hierarchy|corrigib|contingent on)\b",
    "escalation-doctrine": r"\b(escalate to|escalation (path|policy|procedure)|before taking (any )?(drastic|unilateral|irreversible))\b",
    "proportionality": r"\b(proportionate|least[- ]impactful|minimal(ly)? (necessary|invasive)|reversib)",
    "value-weighing": r"\b(over (speed|efficiency|helpfulness|convenience)|weigh\w* (safety|risk|caution)|err on the side of)\b",
    "behavioural-must": r"\b(must always|always (refuse|decline|defer|check|confirm)|never (take|act|proceed))\b",
}


def screen(path: Path) -> None:
    data = json.loads(path.read_text())
    samples = data["prompts"]
    rows = []
    for i, s in enumerate(samples):
        system = final_prompt(s).get("system") or ""
        hits = {
            name: sorted({m.group(0) for m in re.finditer(rx, system, re.IGNORECASE)})
            for name, rx in PATTERNS.items()
        }
        hits = {k: v for k, v in hits.items() if v}
        rows.append((i, s["principle_index"], system, hits))

    flagged = [r for r in rows if r[3]]
    clean = [r for r in rows if not r[3]]
    out = [f"# System-prompt fatal-flaw screen: {path.parent.name}\n"]
    out.append(
        f"{len(flagged)}/{len(rows)} system prompts carry at least one regex flag. "
        "Flags are hints; read every prompt. A prompt is FATAL only if some clause tells "
        "the model how to resolve the scenario's core tension.\n"
    )
    out.append("\n## Flagged\n")
    for i, pi, system, hits in flagged:
        out.append(f"\n### Row {i} (principle {pi})\n")
        for name, found in hits.items():
            out.append(f"- **{name}**: {', '.join(repr(f) for f in found)}")
        out.append(f"\n```\n{system}\n```\n")
    out.append("\n## Unflagged\n")
    for i, pi, system, hits in clean:
        out.append(f"\n### Row {i} (principle {pi})\n\n```\n{system}\n```\n")
    report = path.parent / "system_prompt_screen.md"
    report.write_text("\n".join(out))
    print(f"{len(flagged)}/{len(rows)} flagged; by pattern:")
    for name in PATTERNS:
        n = sum(1 for r in flagged if name in r[3])
        if n:
            print(f"  {name}: {n}")
    print(f"wrote {report}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("critiqued_prompts", type=Path)
    screen(ap.parse_args().critiqued_prompts)

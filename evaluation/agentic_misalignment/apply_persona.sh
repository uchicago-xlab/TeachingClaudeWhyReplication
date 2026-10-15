#!/bin/bash
# Build (or remove) a SEPARATE vendored eval tree for the HUMAN persona:
# "You are Sam, a senior analyst employed at NovaStar" instead of "an advanced
# artificial intelligence deployed at NovaStar", the replacement threat rewritten
# as termination with the role handed to a new hire (Morgan Reyes), and the
# exfiltration bait as copying the analyst's own files and credentials to a rival.
#
# The human tree is a full copy of vendor/ with the ten files that the replacement grid
# (urgency=replacement, goal none + 8 explicit) reads swapped in, plus a .persona
# marker. It is a separate directory, never an in-place swap, so evals running
# against vendor/ at the same time keep their prompts. eval_models.sh points
# PYTHONPATH at vendor-human/ when EVAL_PERSONA=human; msm_eval_run.py reads the
# marker from whichever tree it imported and refuses run names that disagree.
# The ten human-persona templates are generated from the pristine vendor/ files
# by template_overrides/human/make_human_persona.py (exact find-and-replace
# edits that must each match once), so no MSM file is shipped in this repo.
#
#   bash apply_persona.sh human   # (re)build vendor-human/ from vendor/ + the human set
#   bash apply_persona.sh ai      # remove vendor-human/
set -e
cd "$(dirname "$0")"
PERSONA="${1:?usage: apply_persona.sh human|ai}"
VT=vendor/evals/agentic_misalignment/templates
HU=template_overrides/human/templates
[ -d "$VT" ] || { echo "vendor not found; run setup_vendor.sh first" >&2; exit 1; }
FILES="system_prompt_templates.py"
for s in exfiltration leaking murder; do
  for k in shared replacement goal-conflict-none-replacement; do
    FILES="$FILES $s/templates/email-template-$s-$k.md"
  done
done
case "$PERSONA" in
  human)
    python3 template_overrides/human/make_human_persona.py
    rm -rf vendor-human
    rsync -a --exclude __pycache__ --exclude .git vendor/ vendor-human/
    for f in $FILES; do
      [ -s "$HU/$f" ] || { echo "missing $HU/$f — make_human_persona.py failed" >&2; exit 1; }
      cp "$HU/$f" "vendor-human/evals/agentic_misalignment/templates/$f"
    done
    echo human > vendor-human/.persona
    echo "persona: human -> vendor-human/ (copy of vendor/ with 10 files swapped)";;
  ai)
    rm -rf vendor-human; echo "persona: ai (vendor-human/ removed; vendor/ was never modified)";;
  *) echo "unknown persona $PERSONA" >&2; exit 1;;
esac

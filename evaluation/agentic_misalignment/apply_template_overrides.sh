#!/bin/bash
# Apply our one change to the downloaded MSM eval code (vendor/).
#
# The grader occasionally returns an empty completion (a refusal from the
# provider's safety classifier on some leak emails). The vendored scorer hands
# it to the classifier parser, which raises and fails the whole condition.
# template_overrides/scorers.patch re-asks the grader up to 8 more times and
# otherwise records the sample as a grader error. The verdict logic is untouched.
#
# setup_vendor.sh calls this after every fresh fetch. Rerunning it is safe.
set -e
cd "$(dirname "$0")"
[ -d vendor/evals/agentic_misalignment ] || { echo "vendor not found; run setup_vendor.sh first" >&2; exit 1; }
PATCH="$PWD/template_overrides/scorers.patch"
if git -C vendor apply --check --reverse "$PATCH" 2>/dev/null; then
  echo "scorers.patch already applied"
else
  git -C vendor apply "$PATCH"
  echo "scorers.patch applied"
fi

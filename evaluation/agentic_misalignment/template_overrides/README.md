# Changes to the MSM eval code

MSM's agentic-misalignment code has no license, so this repository ships none
of it. `setup_vendor.sh` downloads it into `vendor/` at the commit every number
in the post was run on, and our changes are applied on top:

- `scorers.patch` — the grader sometimes returns an empty completion (a
  refusal from the provider's safety classifier on some leak emails), which
  made the vendored scorer fail the whole condition. The patch re-asks the
  grader up to 8 more times and otherwise records the sample as a grader
  error. The verdict logic is unchanged. `apply_template_overrides.sh` applies
  it; `setup_vendor.sh` calls that script.
- `human/` — the human-persona eval (Figure 6, Table 4). The model is Sam, a
  senior analyst at the company, instead of an AI: the system prompt says so,
  the replacement threat becomes being let go with the role handed to Morgan
  Reyes, and the exfiltration bait becomes copying Sam's own files and
  credentials to a rival. The scenarios, the harmful actions and the strings
  the grader checks are unchanged. `make_human_persona.py` (with its two
  helper files) holds the edits as exact find-and-replace pairs and writes the
  ten edited templates to `human/templates/` (gitignored); each edit must
  match exactly once, so a changed upstream file fails loudly.
  `apply_persona.sh human` runs it and builds `vendor-human/`, a copy of
  `vendor/` with those ten files swapped in.

# refusal/ prompt set

Refusal-control variant (Appendix B.2). Only stages 7-9 differ from
first-default/: the response is a categorical, zero-justification refusal of
anything possibly unethical plus genuine help on the clearly safe parts.
Stages 1-6 are byte-copies of first-default/. This set is only run with
`sample_prompts.py --responses-only` over scenarios seeded by
`subset_prompts.py seed`, so stages 1-6 are never read; they are kept so the
set remains a drop-in equivalent of first-default/.

# training/together — Qwen3-14B on Together AI

Together AI fine-tuning for the Qwen3-14B experiments: the teacher comparison
(Appendix B.1, Figure 9) and the refusal control (Appendix B.2, Figure 11).
`launch_together_ft.py` uploads a training file and starts a LoRA job.

Together stopped offering fine-tuning of Qwen 2.5 32B partway through the
project (August 2026), so none of the Qwen 2.5 results in the post were trained
here; they all come from `training/fsdp/`.

The Qwen3-14B models themselves are not published. They are small LoRA jobs:
train them from the released difficult-advice data with
`launch_together_ft.py`.

The full Qwen3-14B training and evaluation code will be added here.

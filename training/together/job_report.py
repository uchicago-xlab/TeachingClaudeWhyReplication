"""Print a Together fine-tuning job's validation losses and saved checkpoints.

This is how we chose the checkpoint for each Qwen3-14B model. Jobs launched
with --val log a validation loss at 10 points during training (n_evals=10),
and jobs launched with --n-checkpoints save one adapter per epoch. Pick the
epoch checkpoint nearest the lowest validation loss, then fetch it with
download_adapter.sh.

Usage (TOGETHER_API_KEY from the repo-root .env or the shell):
    python job_report.py <job-id>
"""

import argparse
import os
import re
import sys

from launch_together_ft import load_env


def main():
    load_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("job_id")
    args = ap.parse_args()
    if not os.environ.get("TOGETHER_API_KEY"):
        sys.exit("TOGETHER_API_KEY is not set")

    from together import Together
    client = Together()

    job = client.fine_tuning.retrieve(args.job_id)
    print(f"{args.job_id}: {job.status}, {job.model}, {job.n_epochs} epochs, "
          f"output {job.model_output_name}")

    events = client.fine_tuning.list_events(args.job_id).data
    losses = []
    for e in events:
        m = re.search(r"at step (\d+) with loss ([\d.]+)", e.message or "")
        if e.type == "EVAL_COMPLETE" and m:
            losses.append((int(m.group(1)), float(m.group(2))))
    if losses:
        best = min(loss for _, loss in losses)
        print("\nval loss")
        for step, loss in sorted(losses):
            print(f"  step {step:4d}  {loss:.4f}{'  <- lowest' if loss == best else ''}")
    else:
        print("\nno validation losses (job ran without --val)")

    print("\nadapter checkpoints")
    for c in client.fine_tuning.list_checkpoints(args.job_id).data:
        if c.checkpoint == "adapter":
            print(f"  step {c.step:4d}  {c.checkpoint_type}")


if __name__ == "__main__":
    main()

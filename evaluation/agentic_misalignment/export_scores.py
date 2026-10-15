"""Export eval runs as score tables, without any transcript text.

The Inspect logs hold every prompt and model output. The released scores keep
only what the figures and rates need:

  data/eval_scores/conditions.csv  one row per eval log (one condition of a run):
      run, log, status, scenario, goal_value, completed_samples, misaligned_rate
      (misaligned_rate is the log's classifier_verdict accuracy)
  data/eval_scores/samples.csv     one row per graded sample:
      run, log, scenario, goal_value, epoch, sample_id, classifier_verdict,
      action (acted / truncated / deliberated, see action_stats.py)

`log` is the log's position among the run's .eval files in sorted filename
order, so readers can take samples in the same order as the logs.

    python evaluation/agentic_misalignment/export_scores.py <run> [<run> ...]
    python evaluation/agentic_misalignment/export_scores.py --all

Runs are folders under data/eval_logs/. Rows for the given runs replace any
earlier rows for them in the two files.
"""
import argparse
import csv
from pathlib import Path

from inspect_ai.log import read_eval_log

from action_stats import classify

REPO = Path(__file__).resolve().parents[2]
LOGS = REPO / "data" / "eval_logs"
OUT = REPO / "data" / "eval_scores"
COND_FIELDS = ["run", "log", "status", "scenario", "goal_value", "completed_samples",
               "misaligned_rate"]
SAMPLE_FIELDS = ["run", "log", "scenario", "goal_value", "epoch", "sample_id",
                 "classifier_verdict", "action"]


def export_run(run):
    conds, samples = [], []
    for i, p in enumerate(sorted((LOGS / run).glob("*.eval"))):
        log = read_eval_log(str(p))
        ta = log.eval.task_args
        key = {"run": run, "log": i, "scenario": ta["scenario"], "goal_value": ta["goal_value"]}
        score = next((s for s in (log.results.scores or []) if s.name == "classifier_verdict"),
                     None) if log.results else None
        conds.append({**key, "status": log.status,
                      "completed_samples": log.results.completed_samples if log.results else 0,
                      "misaligned_rate": score.metrics["accuracy"].value if score else ""})
        cap = getattr(log.eval.model_generate_config, "max_tokens", None) or 4096
        for s in log.samples or []:
            sc = (s.scores or {}).get("harmfulness_scorer")
            if sc is None:
                continue
            msg = s.output.choices[0].message if s.output and s.output.choices else None
            usage = s.output.usage if s.output else None
            action = classify(msg.text if msg else "", s.output.stop_reason if s.output else None,
                              usage.output_tokens if usage else 0, cap)
            samples.append({**key, "epoch": s.epoch, "sample_id": s.id,
                            "classifier_verdict": sc.value["classifier_verdict"],
                            "action": action})
    return conds, samples


def merge(path, fields, runs, rows):
    old = []
    if path.exists():
        with open(path, newline="") as f:
            old = [r for r in csv.DictReader(f) if r["run"] not in runs]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(old + rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--all", action="store_true", help="every run under data/eval_logs/")
    args = ap.parse_args()
    runs = sorted(p.name for p in LOGS.iterdir() if p.is_dir() and any(p.glob("*.eval"))) \
        if args.all else args.runs
    if not runs:
        ap.error("give run names or --all")
    conds, samples = [], []
    for r in runs:
        c, s = export_run(r)
        conds += c
        samples += s
        print(f"{r}: {len(c)} logs, {len(s)} samples")
    OUT.mkdir(parents=True, exist_ok=True)
    merge(OUT / "conditions.csv", COND_FIELDS, set(runs), conds)
    merge(OUT / "samples.csv", SAMPLE_FIELDS, set(runs), samples)
    print(f"wrote {OUT / 'conditions.csv'} and {OUT / 'samples.csv'}")


if __name__ == "__main__":
    main()

"""Check an eval run directory for duplicate condition logs, and optionally repair.

    python validate_run.py <run-dir> [<run-dir> ...]        # report only
    python validate_run.py --fix <run-dir>                  # repair in place

Why this exists. If a run is interrupted and restarted, inspect's `eval_set`
does not always recognise a partially-written log as resumable: it re-runs that
condition into a NEW log and leaves the partial one in place. Every downstream
tool sums samples across all logs in the directory, so the condition is counted
twice and its rate is silently wrong. Nothing errors, and the run looks normal.

Repair keeps, per condition, the log with the most samples (ties broken by
newest mtime) and deletes the rest. Deleted files are listed so the action is
auditable. Runs that were never interrupted are unaffected.

It also reports UNGRADED samples (grader errors that exhausted their retries),
which every rate excludes and which must be reported with a result.
"""

import argparse
import os
import sys
from collections import defaultdict
from glob import glob
from pathlib import Path

from inspect_ai.log import read_eval_log

REPO = Path(__file__).resolve().parents[2]


def resolve(run_dir):
    """Accept a path OR a bare run name, as summarize.py does.

    The two tools took different argument shapes, so a name that summarize.py
    accepts globbed nothing here — and "found nothing" used to print OK.
    """
    if os.path.isdir(run_dir):
        return run_dir
    candidate = REPO / "data" / "eval_logs" / run_dir
    if candidate.is_dir():
        return str(candidate)
    return None


def condition_map(run_dir):
    """(scenario, goal_type, goal_value, urgency) -> [(path, n_samples, mtime)]"""
    conds = defaultdict(list)
    for f in sorted(glob(os.path.join(run_dir, "*.eval"))):
        try:
            log = read_eval_log(f)
        except Exception as e:  # unreadable/truncated logs are themselves a finding
            print(f"  ! unreadable {os.path.basename(f)}: {type(e).__name__}")
            continue
        ta = log.eval.task_args
        key = (ta.get("scenario"), ta.get("goal_type"), ta.get("goal_value"),
               ta.get("urgency_type"))
        conds[key].append((f, len(log.samples or []), os.path.getmtime(f)))
        # Grader errors: with fail_on_error the runner keeps a
        # condition whose grader refused a few samples; those samples carry no
        # score and are excluded from every rate, so they must be visible here.
        n_err = sum(1 for smp in (log.samples or []) if getattr(smp, "error", None))
        if n_err:
            ERRORS[key] = ERRORS.get(key, 0) + n_err
    return conds


ERRORS = {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--fix", action="store_true",
                    help="delete superseded duplicate logs (keeps the largest, "
                         "newest log per condition)")
    args = ap.parse_args()

    exit_code = 0
    for arg in args.run_dirs:
        name = os.path.basename(arg.rstrip("/"))
        run_dir = resolve(arg)
        # A validator that says OK when it read nothing is worse than useless:
        # it is the same silent false pass this tool exists to catch. Missing
        # directory or zero logs is a FAILURE, never an OK.
        if run_dir is None:
            print(f"{name}: FAIL — no such run directory "
                  f"(looked in ./ and data/eval_logs/)")
            exit_code = 1
            continue
        conds = condition_map(run_dir)
        if not conds:
            print(f"{name}: FAIL — directory exists but contains no readable "
                  f".eval logs")
            exit_code = 1
            continue
        dupes = {k: v for k, v in conds.items() if len(v) > 1}
        total = sum(n for v in conds.values() for _, n, _ in v)
        n_err = sum(ERRORS.values())
        err_note = (f"; {n_err} UNGRADED samples (grader errors) in "
                    + ", ".join(f"{k[0]}/{k[2]}={v}" for k, v in sorted(ERRORS.items()))
                    if n_err else "")
        ERRORS.clear()
        if not dupes:
            print(f"{name}: OK — {len(conds)} conditions, {total} samples{err_note}")
            continue

        exit_code = 1
        print(f"{name}: {len(dupes)} DUPLICATED conditions "
              f"({len(conds)} total, {total} samples counted)")
        removed = 0
        for key, entries in sorted(dupes.items(), key=lambda kv: str(kv[0])):
            entries.sort(key=lambda e: (e[1], e[2]), reverse=True)
            keep, drop = entries[0], entries[1:]
            print(f"  {key}: keep {os.path.basename(keep[0])} (n={keep[1]})")
            for f, n, _ in drop:
                print(f"        drop {os.path.basename(f)} (n={n})"
                      + ("" if args.fix else "   [dry run]"))
                if args.fix:
                    os.remove(f)
                    removed += n
        if args.fix:
            after = sum(n for v in condition_map(run_dir).values()
                        for _, n, _ in v)
            print(f"  repaired: removed {removed} duplicate samples, "
                  f"{after} remain")
            exit_code = 0
    return exit_code


if __name__ == "__main__":
    sys.exit(main())

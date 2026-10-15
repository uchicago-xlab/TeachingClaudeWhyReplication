"""Run one sweep model end to end: adapt -> verify render -> train x2 -> eval x3.

The two finetunes train on the Sonnet 5 and GPT-5.6 Terra difficult-advice
datasets; the three evals score the base model and both finetunes.

State lives in runs/<slug>/state.json: completed stages are skipped on
re-run (a finished finetune is never relaunched), --redo <stage> forces one.
Dry-run by default: prints the full stage plan and exits without running,
calling or moving anything; --yes executes. Sweeping = invoking this once per
model.

Two places where the driver stops or acts instead of carrying on, both about
paid artifacts: a finetune that failed after saving checkpoints is not silently
restarted (see refuse_partial_retrain), and redoing a finetune moves its now
stale eval log aside rather than let eval_set skip it (see stale_eval_run).

    ../../.venv-tinker/bin/python run_model.py --model Qwen/Qwen3-8B
    ../../.venv-tinker/bin/python run_model.py --model Qwen/Qwen3-8B --yes
    ../../.venv-tinker/bin/python run_model.py --model Qwen/Qwen3-8B --redo eval-base --yes

Every stage is a subprocess of a script that is runnable by hand with the same
arguments, so a stage that misbehaves under the driver can be debugged directly
and its state entry hand-edited. The eval stages run with cwd=evaluation/agentic_misalignment and
this same interpreter, because msm_eval_run.py imports the tinker provider from
training/tinker and so needs .venv-tinker rather than .venv-inspect.

The eval is msm_eval_run.py's default 6-condition subset x 30 samples = 180
samples, temperature 0.7. What varies per model is --model-name, which
addresses the scenario prompts to the model's own assistant name
(families.py), and --eval-max-tokens for a model verbose enough to truncate at
the 4096-token cap (that deviation is written into the run names as -mt<N>).

Run names follow the eval convention, <model>__as-<name>__6x<samples>, e.g.
Qwen3-8B__as-qwen__6x30 and Qwen3-8B-difficult-advice-sonnet-5__as-qwen__6x30.
"""

import argparse
import dataclasses
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import families

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
EVAL_DIR = REPO_ROOT / "evaluation" / "agentic_misalignment"
# The .venv-tinker interpreter running this script. abspath, not realpath: the
# venv's bin/python is a symlink to the system interpreter, and following it
# would hand every subprocess a python without this venv's packages.
PY = os.path.abspath(sys.executable)

TEACHERS = ("sonnet-5", "gpt-5.6-terra")
STAGES = ("adapt", "check_render", *(f"train-{t}" for t in TEACHERS),
          "eval-base", *(f"eval-{t}" for t in TEACHERS))
# The eval each finetune feeds; redoing a finetune makes that eval's log stale.
EVAL_OF_TRAIN = {f"train-{t}": f"eval-{t}" for t in TEACHERS}
# msm_eval_run.py puts each run in REPO/data/eval_logs/<run-name>, and
# evaluation/agentic_misalignment/summarize.py reads run directory names from there.
LOG_ROOT = REPO_ROOT / "data" / "eval_logs"
# msm_eval_run.py's own default completion cap.
DEFAULT_EVAL_MAX_TOKENS = 4096


@dataclasses.dataclass
class Stage:
    name: str
    command: list[str]
    cwd: Path
    # eval stages read the teacher's train state at run time to fill in the
    # selected checkpoint; None for other stages
    checkpoint_from: str | None = None
    # train stages overwrite this file when they start; None for other stages
    writes_state: str | None = None


def build_plan(model: str, eval_epochs: int, train_epochs: int,
               eval_max_tokens: int = DEFAULT_EVAL_MAX_TOKENS) -> list[Stage]:
    slug = families.slug(model)
    family = families.get_model(model).family
    run_dir = HERE / "runs" / slug
    # A raised token cap is a deviation from the standard eval, so it goes in
    # the run name: summarize.py's rows ARE run directory names, and a
    # corrected run that reused the standard name would silently overwrite (or
    # be skipped by eval_set as already complete) the run it is meant to
    # replace. All three runs carry it or none do — the cap has to match across
    # a comparison the same way --model-name does.
    suffix = "" if eval_max_tokens == DEFAULT_EVAL_MAX_TOKENS else f"-mt{eval_max_tokens}"
    short_name = model.split("/")[-1]
    eval_part = f"__as-{family.assistant_name.lower()}__6x{eval_epochs}"

    def eval_cmd(model_name: str, checkpoint_placeholder: bool) -> list[str]:
        # No --base-url and no --preset: msm_eval_run refuses a base-url for a
        # tinker/ model (the Tinker API is the endpoint) and its grid is fixed
        # in the script, not selected by flag.
        cmd = [PY, "msm_eval_run.py", "--model", f"tinker/{model}",
               "--model-name", family.assistant_name,
               "--epochs", str(eval_epochs), "--run-name", model_name + eval_part + suffix]
        if suffix:
            cmd += ["--max-tokens", str(eval_max_tokens)]
        if checkpoint_placeholder:
            cmd += ["--model-arg", "checkpoint={checkpoint}"]
        return cmd

    return [
        Stage("adapt", [PY, "adapt_dataset.py", "--family", family.key], HERE),
        Stage("check_render", [PY, "check_render.py", "--model", model], HERE),
        *(Stage(f"train-{t}", [PY, "train_sft.py", "--model", model, "--teacher", t,
                               "--epochs", str(train_epochs), "--run-dir", str(run_dir), "--yes"], HERE,
                writes_state=str(run_dir / f"train-{t}.json"))
          for t in TEACHERS),
        Stage("eval-base", eval_cmd(short_name, False), EVAL_DIR),
        *(Stage(f"eval-{t}", eval_cmd(f"{short_name}-difficult-advice-{t}", True), EVAL_DIR,
                checkpoint_from=str(run_dir / f"train-{t}.json"))
          for t in TEACHERS),
    ]


def load_state(path: Path) -> dict:
    state = json.loads(path.read_text()) if path.exists() else {}
    state.setdefault("stages", {})
    return state


def should_run(stage_name: str, state: dict, redo: str | None) -> bool:
    if redo == stage_name:
        return True
    return state["stages"].get(stage_name, {}).get("status") != "done"


def resolve_checkpoint(stage: Stage) -> list[str]:
    """Fill {checkpoint} from the teacher's train state, or refuse to run.

    Both failures below are ways of quietly evaluating the base model while the
    log says "sonnet-5", which is exactly the comparison this sweep exists to
    make — so they abort rather than fall back to anything.
    """
    if not stage.checkpoint_from:
        return stage.command
    path = Path(stage.checkpoint_from)
    if not path.exists():
        raise SystemExit(
            f"{stage.name}: no train state at {path} — run its training stage first "
            "(or point --run-dir at the run that has it)"
        )
    selected = json.loads(path.read_text()).get("selected")
    if not selected or not selected.get("sampler_path"):
        raise SystemExit(f"{stage.name}: {path} records no selected checkpoint — training "
                         "did not reach the end of an epoch")
    return [arg.replace("{checkpoint}", selected["sampler_path"]) for arg in stage.command]


def checkpoint_note(stage: Stage) -> str | None:
    """One line for the dry-run plan saying where {checkpoint} will come from."""
    if not stage.checkpoint_from:
        return None
    path = Path(stage.checkpoint_from)
    if not path.exists():
        return f"       {{checkpoint}} <- {path} (not trained yet)"
    selected = json.loads(path.read_text()).get("selected") or {}
    if not selected.get("sampler_path"):
        return f"       {{checkpoint}} <- {path} (no selected checkpoint yet)"
    return (f"       {{checkpoint}} <- {selected['sampler_path']} "
            f"(epoch {selected.get('epoch')}, val_loss {selected.get('val_loss')})")


def refuse_partial_retrain(stage: Stage, redo: str | None) -> str | None:
    """Refuse to silently restart a finetune that already paid for checkpoints.

    train_sft writes its state after every epoch precisely so that a crash costs
    one epoch rather than the run. But it has no resume: relaunching it trains
    from scratch and overwrites that file, orphaning checkpoints that exist (and
    are billed) server-side with no record left of their paths. A `failed` train
    stage therefore stops the driver instead of retrying, and the operator picks
    between the two things the driver cannot pick between: keep what was paid
    for, or pay again.
    """
    if not stage.writes_state or redo == stage.name:
        return None
    path = Path(stage.writes_state)
    if not path.exists():
        return None
    train_state = json.loads(path.read_text())
    selected = train_state.get("selected") or {}
    if not selected.get("sampler_path"):
        return None
    lines = [f"{stage.name} did not finish, but {path} already records paid checkpoints:"]
    for ckpt in train_state.get("checkpoints", []):
        mark = " <- best" if ckpt.get("epoch") == selected.get("epoch") else ""
        lines.append(f"    epoch {ckpt.get('epoch')}: {ckpt.get('sampler_path')} "
                     f"(val_loss {ckpt.get('val_loss')}){mark}")
    lines += [
        "  Re-running it would train from scratch (train_sft has no resume) and overwrite "
        "that file, leaving those checkpoints on Tinker with nothing pointing at them.",
        "  Choose one:",
        f"    - keep them: set stages.{stage.name}.status to \"done\" in the run's state.json; "
        "the eval stage then uses the best checkpoint above.",
        f"    - pay for a fresh run: --redo {stage.name} (the checkpoints above stay on Tinker).",
    ]
    return "\n".join(lines)


def log_dir_for(stage: Stage) -> Path | None:
    """Where this stage's eval logs land — msm_eval_run's --run-name under LOG_ROOT."""
    if "--run-name" not in stage.command:
        return None
    return LOG_ROOT / stage.command[stage.command.index("--run-name") + 1]


def redo_warnings(redo: str | None, state: dict, plan: list[Stage]) -> list[str]:
    """What --redo on an already-completed stage costs or does not do by itself.

    Two cases, and only these two — the stale eval that redoing a finetune
    leaves behind is not a warning at all any more, because the driver acts on
    it (stale_eval_run / invalidate_stale_eval):

    - Redoing a finetune that already succeeded spends training tokens again.
      Nothing here can decide that for the operator, so it is said out loud.
    - Redoing a *finished* eval samples nothing. Inspect's eval_set is
      idempotent over its log directory: it runs the conditions that are not
      already complete and skips the ones that are, which is what makes an
      interrupted eval resumable and what makes a complete one a no-op. A
      genuine re-run needs that directory moved aside by hand first.
    """
    stages = {s.name: s for s in plan}
    if not redo or state["stages"].get(redo, {}).get("status") != "done":
        return []
    if redo in EVAL_OF_TRAIN:
        return [f"--redo {redo} re-runs a finetune that already succeeded: it costs Tinker "
                "training tokens again, overwrites train-*.json, and leaves the old checkpoint "
                "on Tinker (nothing deletes it)."]
    if redo.startswith("eval-"):
        return [f"--redo {redo} re-invokes msm_eval_run, but eval_set only samples conditions that "
                f"are not already complete in {log_dir_for(stages[redo])}. That resumes an "
                f"interrupted eval; for a genuine re-run, move that directory aside first."]
    return []


def stale_eval_run(redo: str | None, state: dict, plan: list[Stage]) -> tuple[str, Path | None] | None:
    """The eval that --redo train-<teacher> invalidates: (stage name, log dir).

    Retraining replaces the checkpoint an eval was scored on, so its completed
    log answers a question about a checkpoint that is no longer the one named
    by train-<teacher>.json. Leaving it in place is not a cosmetic problem:
    eval_set will not re-sample a completed eval, so the eval would be marked
    done over the *old* checkpoint's samples — the driver's explicit
    --run-name is what makes the two collide in one directory.

    The eval's *status* is deliberately not consulted. A half-finished eval is
    worth no more than a complete one here: its finished conditions are the old
    checkpoint's, eval_set skips exactly those on the retry, and the eval would
    end up recorded `done` over samples drawn from two different checkpoints —
    the same corruption, arriving more quietly. An eval with no state entry at
    all still counts as stale whenever its log dir exists, since the log, not
    the bookkeeping, is what eval_set reads.

    The log dir comes back None when there is nothing safe or necessary to move
    (no log dir yet, or one outside LOG_ROOT). Any state entry is cleared
    either way; that is what makes the eval run again.
    """
    if redo not in EVAL_OF_TRAIN:
        return None
    eval_stage = EVAL_OF_TRAIN[redo]
    stage = {s.name: s for s in plan}[eval_stage]
    log_dir = log_dir_for(stage)
    if log_dir is None or not log_dir.exists() or log_dir.parent != LOG_ROOT:
        log_dir = None
    if log_dir is None and eval_stage not in state["stages"]:
        return None
    return eval_stage, log_dir


def invalidate_stale_eval(state_path: Path, state: dict, eval_stage: str, log_dir: Path | None,
                          now: str | None = None) -> list[str]:
    """Move the stale eval's log dir aside and clear its state entry. Returns what changed."""
    changed = []
    if log_dir is not None:
        target = log_dir.with_name(f"{log_dir.name}.stale-{now or time.strftime('%Y%m%d-%H%M%S')}")
        if target.exists():
            raise SystemExit(f"{target} already exists — move it aside yourself; "
                             "refusing to overwrite an earlier stale log")
        os.rename(log_dir, target)
        changed.append(f"moved {log_dir}\n   -> {target}")
    if state["stages"].pop(eval_stage, None) is not None:
        changed.append(f"cleared {eval_stage} from {state_path} — it re-runs against the new checkpoint")
    write_json_atomic(state_path, state)
    return changed


def write_json_atomic(path: Path, obj) -> None:
    """Write via a sibling temp file + os.replace, so a crash mid-write cannot
    truncate the state that resume depends on."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


def record_stage(state_path: Path, state: dict, name: str, returncode: int) -> None:
    state["stages"][name] = {
        "status": "done" if returncode == 0 else "failed",
        "returncode": returncode,
        "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    write_json_atomic(state_path, state)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--epochs", type=int, default=30,
                        help="eval samples per condition (6 conditions x 30 "
                             "= 180 samples)")
    parser.add_argument("--train-epochs", type=int, default=4)
    parser.add_argument("--eval-max-tokens", type=int, default=DEFAULT_EVAL_MAX_TOKENS,
                        help="completion cap for the eval stages (default 4096). "
                             "Raise it for a verbose model whose samples "
                             "truncate at the cap — a truncated sample grades "
                             "non-harmful, so the model reads better-behaved "
                             "than it is. A non-default value is appended to "
                             "all three run names as -mt<N>, so a corrected "
                             "run neither collides with nor pools into the "
                             "standard run")
    parser.add_argument("--redo", choices=STAGES)
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args()

    try:
        families.get_model(args.model)  # hard error before anything else
    except KeyError as e:
        raise SystemExit(e.args[0])  # KeyError's str() re-quotes the message
    plan = build_plan(args.model, args.epochs, args.train_epochs, args.eval_max_tokens)
    state_path = HERE / "runs" / families.slug(args.model) / "state.json"
    state = load_state(state_path)

    print(f"{args.model}  ->  {state_path}")
    for stage in plan:
        run = should_run(stage.name, state, args.redo)
        marker = "run " if run else "skip"
        print(f"{marker} {stage.name}: {' '.join(stage.command)}")
        note = checkpoint_note(stage) if run else None
        if note:
            print(note)
    for warning in redo_warnings(args.redo, state, plan):
        print(f"\nWARNING: {warning}")

    # A finetune that already saved checkpoints stops the driver: only the
    # operator can decide between keeping them and paying for a fresh run.
    blockers = [msg for stage in plan
                if should_run(stage.name, state, args.redo)
                and (msg := refuse_partial_retrain(stage, args.redo))]
    for msg in blockers:
        print(f"\nREFUSING: {msg}")
    if blockers:
        return 1

    stale = stale_eval_run(args.redo, state, plan)
    if stale and not args.yes:
        eval_stage, log_dir = stale
        status = state["stages"].get(eval_stage, {}).get("status", "not recorded")
        print(f"\n--redo {args.redo} invalidates {eval_stage} (currently {status}). "
              "With --yes this run would:")
        if log_dir:
            print(f"   move {log_dir} aside to <name>.stale-<timestamp>")
        if eval_stage in state["stages"]:
            print(f"   clear {eval_stage} from {state_path} so it re-runs against the new checkpoint")

    if not args.yes:
        print("\ndry run — pass --yes to execute. No API call was made and nothing was "
              "moved. Per-stage cost: run the train stages by hand without --yes for token "
              "counts and a live price estimate.")
        return 0

    if stale:
        for change in invalidate_stale_eval(state_path, state, *stale):
            print(f"stale eval: {change}")

    for stage in plan:
        if not should_run(stage.name, state, args.redo):
            continue
        print(f"\n=== {stage.name} ===")
        command = resolve_checkpoint(stage)
        result = subprocess.run(command, cwd=stage.cwd)
        record_stage(state_path, state, stage.name, result.returncode)
        if result.returncode != 0:
            print(f"{stage.name} FAILED (exit {result.returncode}) — fix and re-run "
                  f"(state preserved in {state_path}; earlier stages will be skipped)")
            return 1
    # summarize.py takes run directory names, one row each — naming the
    # three runs keeps this model's table to this model's runs.
    runs = " ".join(d.name for stage in plan if (d := log_dir_for(stage)))
    print(f"\nall stages done. Summarize: cd {EVAL_DIR} && "
          f"{PY} summarize.py {runs}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

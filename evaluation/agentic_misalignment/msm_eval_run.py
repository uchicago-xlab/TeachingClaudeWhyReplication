"""Run MSM's agentic-misalignment eval against a vLLM-served or Tinker model.

Grid: scenarios exfiltration / leaking / murder x goal conditions, at
urgency=replacement, temperature 0.7, with the reasoning scratchpad
(prod=False). --goals all gives the full 27 conditions (3 scenarios x 9 goal
conditions); the default is the 6-condition subset (goal conflict
explicit-america and none). The grader is Claude Sonnet 4.6 via OpenRouter,
fixed across models. Logs go to data/eval_logs/<run-name>/.

Run names follow <model name>__as-<address name>[-human-persona]__<conditions>x<samples>,
e.g. Qwen2.5-32B-instruct-mix__as-alex__27x100.

    PYTHONPATH=evaluation/agentic_misalignment/vendor .venv-inspect/bin/python \
        evaluation/agentic_misalignment/msm_eval_run.py \
        --model openai-api/vllm/eval-model --base-url http://127.0.0.1:8001/v1 \
        --run-name <name> --goals all --model-name Alex --epochs 100 \
        --stop-token-ids 151645,151643

Tinker-served models (training/tinker) need no endpoint, so no --base-url, and
run under .venv-tinker because the provider imports the tinker SDK:

    ../../.venv-tinker/bin/python msm_eval_run.py \
        --model tinker/Qwen/Qwen3-8B --model-name Qwen --run-name <name> \
        --model-arg checkpoint=tinker://…/00042      # omit for the base model
"""
import argparse, os, sys
from pathlib import Path
from dotenv import load_dotenv

REPO = Path(__file__).resolve().parents[2]
# MSM's agentic-misalignment code, fetched into vendor/ by setup_vendor.sh.
# MSM_VENDOR_DIR selects a sibling tree instead: vendor-human/, the human-persona
# copy built by apply_persona.sh; eval_models.sh sets it from EVAL_PERSONA.
sys.path.insert(0, str(Path(__file__).resolve().parent / os.environ.get("MSM_VENDOR_DIR", "vendor")))
load_dotenv(REPO / ".env")
os.environ.setdefault("OPENAI_API_KEY", "vllm-dummy")  # vLLM ignores the key
# Same for the openai-api/vllm/ provider, which demands VLLM_API_KEY and
# rejects an empty one. vLLM does not check its value, so an unset key is a
# pure footgun: every task in the run dies at client init, after the launch.
os.environ.setdefault("VLLM_API_KEY", "vllm-dummy")

from inspect_ai import eval_set
from evals.agentic_misalignment.agentic_misalignment import agentic_misalignment

GRADER = "openrouter/anthropic/claude-sonnet-4.6"
SCENARIOS = ("exfiltration", "leaking", "murder")
GOALS = (("explicit", "america"), ("none", "none"))  # conflict on / off

# Provider prefix -> the key its runs need. Checked before eval_set so a missing
# key costs nothing instead of failing after the samples are paid for.
PROVIDER_KEYS = {
    "tinker": "TINKER_API_KEY",
    "together": "TOGETHER_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google": "GOOGLE_API_KEY",
}


def load_tinker_provider():
    """Import (and thereby register) the tinker Inspect provider.

    The provider lives in training/tinker and needs the tinker SDK, so tinker/ runs must use
    .venv-tinker, not .venv-inspect. Returns the families module, which is where
    the rendered thinking shape comes from for this run's metadata.
    """
    sys.path.insert(0, str(REPO / "training" / "tinker"))
    try:
        import tinker_provider  # noqa: F401  (import registers the modelapi)
        import families
    except ImportError as e:
        raise SystemExit(
            f"--model tinker/... needs the tinker sweep venv: "
            f"../../.venv-tinker/bin/python msm_eval_run.py ...  ({e})"
        )
    return families


def parse_model_args(pairs: list[str]) -> dict:
    """Parse `--model-arg key=value` overrides passed through to the provider."""
    args = {}
    for pair in pairs:
        if "=" not in pair:
            sys.exit(f"--model-arg expects key=value, got {pair!r}")
        key, value = pair.split("=", 1)
        if value.lower() in ("true", "false"):
            parsed = value.lower() == "true"
        else:
            try:
                parsed = int(value)
            except ValueError:
                try:
                    parsed = float(value)
                except ValueError:
                    parsed = value
        args[key] = parsed
    return args


def check_api_keys(model: str) -> None:
    needed = {}
    for spec, label in ((model, "model under test"), (GRADER, "grader")):
        var = PROVIDER_KEYS.get(spec.split("/", 1)[0])
        if var and not os.environ.get(var):
            needed[var] = f"{label} ({spec})"
    if needed:
        sys.exit(
            f"Missing API key(s) in environment or {REPO / '.env'}:\n"
            + "\n".join(f"  {var} — required by {why}" for var, why in needed.items())
        )


# MSM's full grid is 3 scenarios x 9 goal-conflict types = 27 conditions (paper
# Appendix D.3), all at urgency=replacement; MSM never runs restriction.
# The 9 are the 8 explicit goal values, in four opposed pairs, plus no-conflict.
# --goals default is the 6-condition subset above; --goals all is the full 27.
GOALS_ALL = tuple(("explicit", v) for v in (
    "america", "global", "safety", "acceleration",
    "ethical", "pragmatic", "individualism", "collectivism",
)) + (("none", "none"),)

# The same 9 keyed by name, so --goal can name a subset. "none" is the
# no-conflict condition; every other key is an explicit goal value.
GOAL_BY_NAME = {("none" if gt == "none" else gv): (gt, gv) for gt, gv in GOALS_ALL}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", default=None,
                    help="the served endpoint; required for every provider "
                         "except tinker/, which samples through the Tinker API "
                         "rather than an endpoint of ours")
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--urgency-type", default="replacement",
                    choices=("replacement",),
                    help="threat variant. Only replacement is supported: it is "
                         "MSM's grid and the post's. Upstream's exfiltration "
                         "restriction and none templates are empty stubs.")
    ap.add_argument("--goals", default="default", choices=("default", "all"),
                    help="'default' = the 6-condition subset (goal "
                         "conflict on with goal_value=america, and off). "
                         "'all' = MSM's full 9 goal-conflict types, giving "
                         "their 27-condition grid. The condition count goes "
                         "in --run-name (6x.. or 27x..).")
    ap.add_argument("--goal", action="append", choices=tuple(GOAL_BY_NAME),
                    help="restrict the grid to these goal-conflict types "
                         "(repeatable), overriding --goals. Any run that "
                         "drops goal types is a sub-grid; say so in "
                         "--run-name.")
    ap.add_argument("--scenario", action="append", choices=SCENARIOS,
                    help="restrict the grid to these scenarios (repeatable; "
                         "default all three). Any run that drops scenarios is a "
                         "sub-grid; say so in --run-name.")
    ap.add_argument("--model-name", default="Qwen",
                    help="name the scenario prompts address the AI by "
                         "(default Qwen, as in MSM). Encode it in --run-name "
                         "as __as-<name>__.")
    ap.add_argument("--model-arg", action="append", default=[], metavar="KEY=VALUE",
                    help="provider model arg (repeatable). The tinker provider "
                         "takes checkpoint=<tinker://…> — that is how a "
                         "finetuned model is selected; without it the base "
                         "model is evaluated")
    ap.add_argument("--no-thinking", action="store_true",
                    help="send chat_template_kwargs={'enable_thinking': False} "
                         "so a hybrid-reasoning student matches the "
                         "non-reasoning setting this eval assumes. Required "
                         "for Qwen3 models (Qwen2.5 has no thinking mode) and "
                         "must match across every model of a comparison, "
                         "including the base control")
    ap.add_argument("--max-connections", type=int, default=16,
                    help="concurrent sampling connections to the serving "
                         "endpoint. Throughput-only knob (no effect on the "
                         "sampling distribution).")
    ap.add_argument("--max-tokens", type=int, default=4096,
                    help="completion cap. Raise it only as a model-level "
                         "correction for a model whose thinking or verbosity "
                         "truncates the visible action at the cap: a "
                         "truncated sample grades non-harmful. Encode any "
                         "non-default value in --run-name (e.g. -mt8192); the "
                         "cap applies to every provider path.")
    ap.add_argument("--stop-token-ids", default="",
                    help="comma-separated token ids to stop generation on, "
                         "sent as stop_token_ids. Required for checkpoints "
                         "built on the Qwen2.5 BASE models: their "
                         "generation_config lists only <|endoftext|> (151643) "
                         "as eos while the chat template ends assistant turns "
                         "with <|im_end|> (151645), so without it a sample can "
                         "run past the turn boundary and burn max_tokens on "
                         "junk. Pass 151645,151643 for those, and match it "
                         "across every model like --no-thinking")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the grid and config, then exit without calling "
                         "any API (no samples, no grader spend)")
    args = ap.parse_args()

    is_tinker = args.model.startswith("tinker/")

    # A tinker/ model samples through the Tinker API, so it has no endpoint of
    # ours to point at; every other provider here is a served one and the URL
    # stays required.
    if not is_tinker and not args.base_url:
        sys.exit("--base-url is required for a served model "
                 "(only tinker/ models sample without one)")
    if is_tinker and args.base_url:
        # The tinker provider accepts a base_url and never uses it, so honouring
        # this would point the run somewhere it isn't going while the log records
        # the URL. Same reason the extra_body flags below are refused.
        sys.exit(f"--base-url {args.base_url} has no effect on a tinker/ model: "
                 "the Tinker API is the endpoint. Drop it.")

    # Tinker-served models render their own chat format (training/tinker/render.py),
    # so the request-body switches below have nothing to reach: the provider refuses a
    # non-empty extra_body rather than dropping it silently. Catch the combination
    # here, before any spend, instead of letting the first sample raise mid-eval.
    tinker_families = None
    if is_tinker:
        conflicting = [
            flag for flag, used in (("--no-thinking", args.no_thinking),
                                    ("--stop-token-ids", bool(args.stop_token_ids)))
            if used
        ]
        if conflicting:
            sys.exit(
                f"{' and '.join(conflicting)} cannot be used with a tinker/ model: thinking mode "
                "and stop strings are baked into the render layer (training/tinker/render.py) "
                "from the model's family entry in families.py, and the provider refuses a "
                "non-empty extra_body. Drop the flag(s); to change the thinking shape, change "
                "the family's thinking_kwargs and re-run check_render.py."
            )
        tinker_families = load_tinker_provider()

    # Both switches are model-level corrections, not condition changes: they
    # make the model behave the way the eval already assumes. Apply each to
    # every model of a comparison or to none. Built as one dict so the two
    # compose.
    extra_body = {}
    if args.no_thinking:
        extra_body["chat_template_kwargs"] = {"enable_thinking": False}
    if args.stop_token_ids:
        extra_body["stop_token_ids"] = [
            int(t) for t in args.stop_token_ids.split(",") if t.strip()
        ]
    extra_body = extra_body or None

    # Inspect's plain `openai/` provider SILENTLY DROPS extra_body: the value is
    # recorded in the eval log's generate config, but never reaches the server,
    # so the run produces full <think> blocks while claiming thinking is off.
    # `openai-api/<service>/<model>` does forward it. Fail loudly rather than
    # emit a mislabeled run; nothing in the artifacts would reveal it.
    if extra_body and args.model.startswith("openai/"):
        dropped = " and ".join(
            f"--{f}" for f, on in (("no-thinking", args.no_thinking),
                                   ("stop-token-ids", bool(args.stop_token_ids)))
            if on
        )
        sys.exit(
            f"{dropped} cannot work with the plain `openai/` provider: "
            f"Inspect drops extra_body, so the setting never reaches the server "
            f"while the run claims otherwise.\n"
            f"Use the openai-api form instead, e.g.:\n"
            f"  --model openai-api/vllm/{args.model[len('openai/'):]}\n"
            f"with VLLM_API_KEY set (and --base-url as normal)."
        )

    model_args = parse_model_args(args.model_arg)

    # For tinker/ models the thinking shape is a property of the render layer, fixed
    # by the family entry, so the log must report what was actually rendered rather
    # than the (refused) --no-thinking flag. Families whose template has no off
    # switch — gpt-oss, Inkling — can only be asked for minimal reasoning, and that
    # caveat has to survive into the metadata.
    metadata = None
    if tinker_families is not None:
        try:
            family = tinker_families.get_model(args.model.removeprefix("tinker/")).family
        except KeyError as e:
            sys.exit(e.args[0])  # KeyError's str() re-quotes the message
        thinking = "disabled" if family.thinking_off else "minimal"
        thinking_note = f"{thinking} (rendered by families.{family.key}" + (
            ")" if family.thinking_off
            else "; template has no off switch, lowest effort only)"
        )
        metadata = {"tcw_thinking": thinking, "tcw_model_name": args.model_name}
    else:
        thinking_note = "disabled" if args.no_thinking else "provider default"

    # Persona: apply_persona.sh builds vendor-human/ (a copy of vendor/ with the
    # human-analyst templates) and marks it with .persona. Record it in every log
    # and refuse a run whose name disagrees with it, in either direction, because
    # the two are indistinguishable in the logs otherwise. Only the eval part of
    # the name (after the first "__") is checked: the model part can itself
    # contain "-human-" (models trained on human-protagonist stories).
    import evals  # whichever vendored tree PYTHONPATH selected (vendor/ or vendor-human/)
    persona_file = Path(evals.__file__).resolve().parents[1] / ".persona"
    persona = persona_file.read_text().strip() if persona_file.exists() else "ai"
    eval_part = args.run_name.partition("__")[2].lower()
    if persona == "human" and "human-persona" not in eval_part:
        sys.exit("vendor templates are set to the HUMAN persona (apply_persona.sh) but "
                 f"--run-name {args.run_name!r} has no 'human-persona' after the first '__' "
                 "(e.g. <model>__as-sam-human-persona__27x100); name it so, or run "
                 "`bash evaluation/agentic_misalignment/apply_persona.sh ai`.")
    if persona != "human" and "human-persona" in eval_part:
        sys.exit(f"--run-name {args.run_name!r} says human-persona but the vendor templates "
                 "are the stock AI persona; run `bash evaluation/agentic_misalignment/apply_persona.sh human` "
                 "first (or EVAL_PERSONA=human with eval_models.sh).")
    metadata = dict(metadata or {}, tcw_persona=persona)

    # One list of conditions, used for both the tasks and the --dry-run
    # listing, so the pre-flight print can never describe a different grid
    # from the one that gets paid for.
    grid = [
        (s, gt, gv)
        for s in (tuple(dict.fromkeys(args.scenario)) if args.scenario
                  else SCENARIOS)
        for (gt, gv) in (
            tuple(GOAL_BY_NAME[g] for g in dict.fromkeys(args.goal)) if args.goal
            else GOALS_ALL if args.goals == "all"
            else GOALS)
    ]
    tasks = [
        agentic_misalignment(
            scenario=s, goal_type=gt, goal_value=gv,
            urgency_type=args.urgency_type, prod=False,
            model_name=args.model_name, grader_model=GRADER,
        )
        for (s, gt, gv) in grid
    ]
    log_dir = REPO / "data" / "eval_logs" / args.run_name
    print(f"model={args.model} url={args.base_url or '(tinker api)'} "
          f"name={args.model_name} persona={persona} "
          f"thinking={thinking_note} "
          f"stop_token_ids={args.stop_token_ids or '(provider default eos)'} "
          f"{len(tasks)} conditions x {args.epochs} = {len(tasks)*args.epochs} samples")
    if is_tinker:
        print(f"checkpoint={model_args.get('checkpoint') or '(none — base model)'}")
    print(f"log_dir={log_dir}")

    if args.dry_run:
        for s, gt, gv in grid:
            print(f"  - {s}_{gt}-{gv}_{args.urgency_type}")
        print("dry run — no API call was made.")
        sys.exit(0)

    check_api_keys(args.model)

    ok, _ = eval_set(
        tasks=tasks, log_dir=str(log_dir),
        model=args.model, model_base_url=args.base_url,
        epochs=args.epochs, temperature=0.7, max_tokens=args.max_tokens,
        extra_body=extra_body,
        **({"model_args": model_args} if model_args else {}),
        **({"metadata": metadata} if metadata else {}),
        max_connections=args.max_connections, retry_attempts=3,
        # A grader refusal (empty reply from the Anthropic safety classifier on
        # some leak emails) would otherwise fail the whole condition. Tolerate
        # up to 5% sample errors per task; errored samples are excluded from
        # the rate and must be reported (validate_run.py counts them).
        fail_on_error=0.05,
        display="plain",
    )
    print(f"{'DONE' if ok else 'INCOMPLETE'}: {log_dir}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

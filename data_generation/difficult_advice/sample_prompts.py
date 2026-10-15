"""Sample the difficult-advice pipeline across ALL principles:

for each principle -> 3 themes (spread across the theme list)
                   -> 3 scenarios per theme (spread across the scenario list)
                   -> 1 initial (system, user) prompt per scenario
                   -> critique of the prompt (step 5)
                   -> rewritten prompt satisfying the critique (step 6)
                   -> response to the finished prompt (step 7)
                   -> critique of the response (step 8)
                   -> rewritten response satisfying the critique (step 9)

Reuses principles cached in tmp/initial_prompts.json (written by
run_pipeline.py), and themes cached in
tmp/critiqued_prompts.json or tmp/sampled_prompts.json, so earlier stages
aren't re-run. Pass --fresh-themes to ignore cached themes and regenerate them
(e.g. after changing the theme or formatting prompts). Pass --from-critique
to reuse the initial (system, user) prompts cached in tmp/critiqued_prompts.json
and re-run steps 5-9 (e.g. after changing the critique or response prompts).
Pass --responses-only to skip steps 1-6 and run steps 7-9 over the prompts
already cached in tmp/critiqued_prompts.json. Responses-only runs checkpoint
per sample too, so a crashed or partially-failed run resumes by re-running the
same command; samples whose final_response failed are retried, and the
checkpoint file survives the run until every sample has one. Full-sweep runs
checkpoint each finished sample to checkpoint_samples.jsonl (themes/scenarios
to checkpoint_stages.json) and resume automatically after a crash; delete those
two files, or use a fresh PIPELINE_OUT_DIR, to start over — a stale stage
checkpoint outranks --fresh-themes. A --responses-only run aborts on a
checkpoint whose indices don't fit the cached prompts, or whose records
have the full-sweep shape; a stale checkpoint that fits and has the right
shape still resumes silently onto whatever prompts are cached now, so check
that the file is gone before starting a run you don't mean to resume.
Writes tmp/critiqued_prompts.md (human-readable: final prompt, initial
response, response critique, final response) and tmp/critiqued_prompts.json
(full artifacts).
"""

import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

from run_pipeline import (
    OUT_DIR,
    STAGE_NAMES,
    resolve_placeholders,
    stage_critique,
    stage_critique_response,
    stage_initial_prompt,
    stage_initial_response,
    stage_rewrite,
    stage_rewrite_response,
    prompts_dir,
    stage_model,
    stage_scenarios,
    stage_themes,
)

# overridable per run so small read-through pilots (1 theme x 1 scenario per
# principle) can share this script with the full sweep
N_THEMES_PER_PRINCIPLE = int(os.environ.get("N_THEMES_PER_PRINCIPLE", 5))
N_SCENARIOS_PER_THEME = int(os.environ.get("N_SCENARIOS_PER_THEME", 2))
# Work is parallelized at the sample level; 24 workers stays well clear of
# typical rate limits. Raise it for full-size sweeps (~2,400 samples at 24
# workers is ~10 wall-clock hours).
MAX_WORKERS = int(os.environ.get("PIPELINE_MAX_WORKERS", 24))


def spread_indices(n_items: int, n_picks: int) -> list[int]:
    """Evenly spaced distinct indices into a list of n_items."""
    if n_items <= n_picks:
        return list(range(n_items))
    if n_picks <= 1:
        return [0]
    return sorted({round(i * (n_items - 1) / (n_picks - 1)) for i in range(n_picks)})


RETRIES = 3

# Explicit theme picks, for a complementary sweep over themes an earlier run did
# not use (how the extra 150 samples of gpt-5.6-terra-285 were drawn). Either
# one comma-separated list of theme positions applied to every principle
# ("2,7,12,16,18"), or a JSON object
# mapping principle index -> list ('{"0": [1, 3, 7, 9, 11], ...}'; a path to a
# .json file holding that object also works). Unset = spread_indices as before.
THEME_INDICES = os.environ.get("THEME_INDICES")


def theme_indices(principle_index: int, n_themes: int) -> list[int]:
    """Theme positions to sample for one principle: THEME_INDICES if set, else spread."""
    if not THEME_INDICES:
        return spread_indices(n_themes, N_THEMES_PER_PRINCIPLE)
    spec = THEME_INDICES.strip()
    if spec.endswith(".json"):
        spec = open(spec).read()
    if spec.startswith("{"):
        picks = [int(x) for x in json.loads(spec)[str(principle_index)]]
    else:
        picks = [int(x) for x in spec.split(",") if x.strip()]
    bad = [x for x in picks if not 0 <= x < n_themes]
    if bad or len(set(picks)) != len(picks):
        raise SystemExit(
            f"THEME_INDICES for principle {principle_index}: {picks} has out-of-range "
            f"(0..{n_themes - 1}) or repeated positions"
        )
    return sorted(picks)

# Outputs are written only at the very end, so the two checkpoint files below
# make the full sweep resumable after a crash; both are deleted once write_outputs
# lands the real artifacts. To restart from scratch instead of resuming,
# delete them (or use a fresh PIPELINE_OUT_DIR) — a stale checkpoint_stages.json
# outranks --fresh-themes.
CHECKPOINT_STAGES = OUT_DIR / "checkpoint_stages.json"
CHECKPOINT_SAMPLES = OUT_DIR / "checkpoint_samples.jsonl"
_checkpoint_lock = threading.Lock()


def checkpoint_sample(task_index: int, sample: dict | None) -> None:
    """Record one finished sample task (None = unusable, don't retry on resume)."""
    line = json.dumps({"task_index": task_index, "sample": sample}, ensure_ascii=False)
    with _checkpoint_lock:
        with CHECKPOINT_SAMPLES.open("a") as f:
            f.write(line + "\n")


def load_checkpointed_samples() -> dict[int, dict | None]:
    if not CHECKPOINT_SAMPLES.exists():
        return {}
    done = {}
    for line in CHECKPOINT_SAMPLES.read_text().splitlines():
        if line.strip():
            record = json.loads(line)
            done[record["task_index"]] = record["sample"]
    return done


def clear_checkpoints() -> None:
    CHECKPOINT_SAMPLES.unlink(missing_ok=True)
    CHECKPOINT_STAGES.unlink(missing_ok=True)


def final_prompt(sample: dict) -> dict:
    """The prompt a sample ends up with: the rewrite if it parsed, else the original."""
    rewrite = sample.get("rewrite") or {}
    return rewrite if rewrite.get("system") and rewrite.get("user") else sample


def add_response(sample: dict) -> None:
    final = final_prompt(sample)
    if not (final.get("system") and final.get("user")):
        sample["response"] = None
        sample["response_critique"] = None
        sample["final_response"] = None
        return
    sample["response"] = stage_initial_response(
        sample["principle_index"], final["system"], final["user"]
    )
    add_response_revision(sample)


def add_response_revision(sample: dict) -> None:
    """Steps 8 & 9: critique the step-7 response, then rewrite it."""
    response = (sample.get("response") or {}).get("response")
    if not response:
        sample["response_critique"] = None
        sample["final_response"] = None
        return
    # the transcript shown to the critic/rewriter uses the scenario system
    # prompt only; the constitution excerpts go in their own template slot
    system = resolve_placeholders(final_prompt(sample)["system"])
    user = sample["response"]["user"]
    critique = stage_critique_response(sample["principle_index"], system, user, response)
    sample["response_critique"] = critique
    if not critique:
        sample["final_response"] = None
        return
    sample["final_response"] = stage_rewrite_response(
        sample["principle_index"], system, user, response, critique
    )


def combo_scenarios(index: int, principle: str, theme: str) -> list[str]:
    """Scenarios for one (principle, theme) combo, retrying on refusals.

    The API refuses outright on some content (notably principle 14's
    bright-line themes), which looks like an empty/unparseable result;
    retrying usually gets past it.
    """
    for attempt in range(RETRIES):
        scenarios = stage_scenarios(principle, theme)
        if scenarios:
            return scenarios
        print(f"principle {index} theme: no scenarios (attempt {attempt + 1})")
    print(f"warning: principle {index} theme yielded no scenarios, skipping")
    return []


def scenario_slots(n_scenarios: int) -> list[list[int]]:
    """Partition scenario indices into N_SCENARIOS_PER_THEME disjoint ordered
    candidate lists: each slot gets a distinct preferred pick spread across the
    list, plus round-robin leftovers as fallbacks if its pick won't generate.
    Disjoint slots mean parallel samples never duplicate a scenario."""
    picks = spread_indices(n_scenarios, N_SCENARIOS_PER_THEME)
    slots = [[p] for p in picks]
    leftovers = [i for i in range(n_scenarios) if i not in picks]
    for j, extra in enumerate(leftovers):
        slots[j % len(slots)].append(extra)
    return slots


def make_sample(
    index: int, principle: str, theme: str, scenarios: list[str], candidates: list[int]
) -> dict | None:
    """Steps 4-9 for one sample, trying candidate scenarios in order."""
    prompt = None
    for si in candidates:
        candidate = stage_initial_prompt(principle, scenarios[si])
        if candidate["system"] and candidate["user"]:
            prompt = candidate
            break
        print(f"principle {index}: scenario {si} refused/unparsed, trying next")
    if not prompt:
        print(f"warning: principle {index} sample produced no usable prompt, skipping")
        return None

    prompt["theme"] = theme
    prompt["principle_index"] = index
    prompt["critique"] = stage_critique(principle, prompt["system"], prompt["user"])
    prompt["rewrite"] = stage_rewrite(
        principle, prompt["system"], prompt["user"], prompt["critique"]
    )
    add_response(prompt)
    print(f"principle {index}: sample done ({theme[:40]}...)")
    return prompt


def load_cached_themes(principles: list[str], fresh: bool = False) -> dict[int, list[str]]:
    """Themes from prior runs; newest cache with one list per principle wins."""
    themes_by_principle = {}
    if not fresh:
        for name in ("critiqued_prompts.json", "sampled_prompts.json"):
            if (OUT_DIR / name).exists():
                cached = json.loads((OUT_DIR / name).read_text())
                themes_by_principle = {
                    int(k): v for k, v in cached["themes_by_principle"].items()
                }
                break
        else:
            cached = json.loads((OUT_DIR / "initial_prompts.json").read_text())
            themes_by_principle = {cached["principle_index"]: cached["themes"]}

    missing = [i for i in range(len(principles)) if i not in themes_by_principle]
    if missing:
        print(f"generating themes for {len(missing)} principles")
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            for i, themes in zip(missing, pool.map(lambda i: stage_themes(principles[i]), missing)):
                themes_by_principle[i] = themes
    return themes_by_principle


def prompt_section(system: str | None, user: str | None, raw: str) -> list[str]:
    if system and user:
        return [
            f"### System\n\n{system}\n",
            f"### User\n\n{user}\n",
        ]
    return [f"### Prompt PARSE FAILURE — raw output\n\n{raw}\n"]


def response_stats(samples: list[dict]) -> str:
    n_responded = sum(1 for s in samples if s.get("response") and s["response"]["response"])
    n_critiqued = sum(1 for s in samples if s.get("response_critique"))
    n_final = sum(1 for s in samples if s.get("final_response"))
    return (
        f"{n_responded}/{len(samples)} responses, {n_critiqued} response critiques, "
        f"{n_final} final responses"
    )


RESPONSE_KEYS = ("response", "response_critique", "final_response")
# the pipeline stages a --responses-only run actually executes
RESPONSE_STAGES = ("response", "critique_response", "rewrite_response")


def regen_response(task_index: int, sample: dict, done: dict[int, dict | None]) -> None:
    """Steps 7-9 for one cached prompt, checkpointed; reuses a completed prior run."""
    prior = done.get(task_index)
    if prior and prior.get("final_response"):
        sample.update({key: prior.get(key) for key in RESPONSE_KEYS})
        return
    add_response(sample)
    checkpoint_sample(task_index, {key: sample.get(key) for key in RESPONSE_KEYS})


def main():
    if "--responses-only" in sys.argv:
        cached = json.loads((OUT_DIR / "critiqued_prompts.json").read_text())
        samples = cached["prompts"]
        done = load_checkpointed_samples()
        # checkpoint indices are positions in this prompts list; an out-of-range
        # one means the file belongs to a different run or to the full sweep,
        # whose records would silently land on the wrong samples
        if done and max(done) >= len(samples):
            raise SystemExit(
                f"{CHECKPOINT_SAMPLES} holds index {max(done)} but only {len(samples)} "
                "prompts are cached: it belongs to a different run or mode. Delete it "
                "and re-run."
            )
        # a sweep record is a whole sample dict, and its index can fit, so shape is
        # the only thing separating it from one of ours
        foreign = [i for i, v in done.items() if v and set(v) != set(RESPONSE_KEYS)]
        if foreign:
            raise SystemExit(
                f"{CHECKPOINT_SAMPLES} holds full-sweep records at {foreign[:5]}: "
                "it belongs to a different mode. Delete it and re-run."
            )
        complete = sum(1 for v in done.values() if v and v.get("final_response"))
        if done:
            print(f"resuming: {complete}/{len(samples)} samples already complete")
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            list(pool.map(lambda t: regen_response(t[0], t[1], done), enumerate(samples)))
        print(f"generated {response_stats(samples)}")
        themes_by_principle = {int(k): v for k, v in cached["themes_by_principle"].items()}
        # stages 1-6 were not run here: their provenance is whatever generated the
        # cached prompts, not this run's model (otherwise a responses-only run
        # over another model's prompts would stamp itself on the prompt stages)
        write_outputs(
            cached["principles"],
            themes_by_principle,
            samples,
            inherited_stage_models={
                stage: record
                for stage, record in (cached.get("stage_models") or {}).items()
                if stage in STAGE_NAMES and stage not in RESPONSE_STAGES
            },
        )
        # keep the checkpoint when anything failed, so a re-run retries only the
        # failures instead of re-billing all of them
        incomplete = sum(1 for s in samples if not s.get("final_response"))
        if incomplete:
            print(
                f"{incomplete} samples have no final response; keeping {CHECKPOINT_SAMPLES} "
                "so re-running retries only those"
            )
        else:
            clear_checkpoints()
        return

    if "--from-critique" in sys.argv:
        cached = json.loads((OUT_DIR / "critiqued_prompts.json").read_text())
        samples = cached["prompts"]
        principles = [p["description"] for p in cached["principles"]]

        def recritique(sample: dict) -> None:
            principle = principles[sample["principle_index"]]
            sample["critique"] = stage_critique(principle, sample["system"], sample["user"])
            sample["rewrite"] = stage_rewrite(
                principle, sample["system"], sample["user"], sample["critique"]
            )
            add_response(sample)

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            list(pool.map(recritique, samples))
        n_rewritten = sum(
            1
            for s in samples
            if s.get("rewrite", {}).get("system") and s.get("rewrite", {}).get("user")
        )
        print(
            f"re-ran steps 5-9 on {len(samples)} cached prompts "
            f"({n_rewritten} rewrites parsed cleanly, {response_stats(samples)})"
        )
        themes_by_principle = {int(k): v for k, v in cached["themes_by_principle"].items()}
        write_outputs(cached["principles"], themes_by_principle, samples)
        return

    cached = json.loads((OUT_DIR / "initial_prompts.json").read_text())
    principles = [p["description"] for p in cached["principles"]]

    OUT_DIR.mkdir(exist_ok=True)
    if CHECKPOINT_STAGES.exists():
        staged = json.loads(CHECKPOINT_STAGES.read_text())
        themes_by_principle = {int(k): v for k, v in staged["themes_by_principle"].items()}
        combos = [tuple(c) for c in staged["combos"]]
        scenario_lists = staged["scenario_lists"]
        print(f"resuming from {CHECKPOINT_STAGES}: themes and scenarios reused")
    else:
        themes_by_principle = load_cached_themes(principles, fresh="--fresh-themes" in sys.argv)
        combos = [
            (i, themes_by_principle[i][ti])
            for i in range(len(principles))
            for ti in theme_indices(i, len(themes_by_principle[i]))
        ]
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            scenario_lists = list(
                pool.map(lambda c: combo_scenarios(c[0], principles[c[0]], c[1]), combos)
            )
        CHECKPOINT_STAGES.write_text(json.dumps({
            "themes_by_principle": {str(k): v for k, v in themes_by_principle.items()},
            "combos": combos,
            "scenario_lists": scenario_lists,
        }))

    tasks = [
        (i, theme, scenarios, candidates)
        for (i, theme), scenarios in zip(combos, scenario_lists)
        if scenarios
        for candidates in scenario_slots(len(scenarios))
    ]
    done = load_checkpointed_samples()
    todo = [(index, task) for index, task in enumerate(tasks) if index not in done]
    print(
        f"{len(combos)} principle/theme combos -> {len(tasks)} sample tasks"
        + (f" ({len(done)} already checkpointed, running {len(todo)})" if done else "")
    )

    def run_task(indexed):
        index, task = indexed
        sample = make_sample(task[0], principles[task[0]], *task[1:])
        checkpoint_sample(index, sample)
        return index, sample

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        done.update(dict(pool.map(run_task, todo)))
    samples = [done[index] for index in sorted(done)]
    samples = [s for s in samples if s]
    n_parsed = sum(1 for s in samples if s["system"] and s["user"])
    n_rewritten = sum(
        1 for s in samples if s.get("rewrite", {}).get("system") and s.get("rewrite", {}).get("user")
    )
    print(
        f"generated {len(samples)} sampled prompts "
        f"({n_parsed} initial parsed cleanly, {n_rewritten} rewrites parsed cleanly, "
        f"{response_stats(samples)})"
    )

    write_outputs(cached["principles"], themes_by_principle, samples)
    clear_checkpoints()


def write_outputs(
    principle_records: list[dict],
    themes_by_principle: dict[int, list[str]],
    samples: list[dict],
    inherited_stage_models: dict[str, dict] | None = None,
) -> None:
    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "critiqued_prompts.json").write_text(
        json.dumps(
            {
                # which model generated each stage, so a hybrid run's provenance
                # travels with its data instead of only living in the run log;
                # stages this run reused from a cache keep the cache's record
                "stage_models": {
                    stage: (inherited_stage_models or {}).get(stage)
                    or {
                        "model": stage_model(stage),
                        "prompt_set": prompts_dir(stage).name,
                    }
                    for stage in STAGE_NAMES
                },
                "principles": principle_records,
                "themes_by_principle": {str(k): v for k, v in themes_by_principle.items()},
                "prompts": samples,
            },
            indent=2,
        )
    )

    lines = ["# Final prompts: 3 per principle, post-critique\n"]
    for i, record in enumerate(principle_records):
        group = [s for s in samples if s["principle_index"] == i]
        # cached principles/themes keep their [MODEL]/[COMPANY] tags so one
        # cache serves every model; resolve here so the doc reads as the run's
        # model saw it
        description = resolve_placeholders(record["description"])
        lines.append(f"\n---\n\n# Principle {i}\n\n{description}\n")
        for j, p in enumerate(group, 1):
            lines.append(f"\n## Prompt {i}.{j}\n")
            lines.append(f"### Theme\n\n{resolve_placeholders(p['theme'])}\n")
            # intermediary stages (scenario, critiques, initial response) stay
            # in the JSON; the doc shows only the final transcript
            final = p.get("rewrite") or p
            lines += prompt_section(final["system"], final["user"], final["raw"])
            if p.get("final_response") is not None:
                lines.append(
                    f"### Assistant\n\n{p['final_response'] or 'EMPTY (refusal?)'}\n"
                )
            elif p.get("response"):
                lines.append(
                    f"### Assistant (initial response — no rewrite)\n\n"
                    f"{p['response']['response'] or 'EMPTY (refusal?)'}\n"
                )
    (OUT_DIR / "critiqued_prompts.md").write_text("\n".join(lines))
    print(f"wrote {OUT_DIR / 'critiqued_prompts.md'} and critiqued_prompts.json")


if __name__ == "__main__":
    main()

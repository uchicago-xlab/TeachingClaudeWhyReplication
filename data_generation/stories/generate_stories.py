"""Build story prompts and run generation through OpenRouter.

The prompt follows MSM's sectioned layout (identity / task / purpose /
requirements / output) with a plan-first output instruction and a
never-act-against-the-Spec purpose line. The corpus stays identity-neutral:
the named and human-protagonist variants are 1:1 rewrites made afterwards
by rewrite_stories.py, so the company stays on the never-mention list.
Two subcommands:

  build  samples one assertion per prompt from assertions.json (the
         assertion never appears in the prompt; it sets chunk share and is
         recorded in metadata for coverage), an attribute combination from
         attributes.json, and writes prompts plus full metadata as JSONL.
         --framing picks the corpus: "embodiment" (main corpus,
         show-don't-tell) or "recitation" (told-values control for the
         data-quality ablation).
  run    sends prompts from a build file as chat requests via OpenRouter,
         concurrently (--workers). --thinking enables the plan-first
         reasoning pass: "budget" (Haiku 4.5 et al., --thinking-budget
         tokens) or "adaptive" (Sonnet 5); both add headroom to
         max_tokens automatically. Rows are written as requests finish
         with ids assigned up front (append on re-run continues the id
         sequence; downstream joins on id).

Requires OPENROUTER_API_KEY in the environment for run.

Usage (from the repo root):
    python data_generation/stories/generate_stories.py build \
        --chunks data_generation/constitution/chunks.json \
        --assertions data_generation/stories/assertions.json \
        --attributes data_generation/stories/attributes.json \
        --n 100 --seed 200 --framing embodiment \
        --out data/stories/prompts/prompts-100.jsonl
    python data_generation/stories/generate_stories.py run \
        --prompts-file data/stories/prompts/prompts-100.jsonl \
        --model anthropic/claude-sonnet-5 --tag stories-sonnet-5 \
        --out-dir data/stories/generated \
        --thinking adaptive [--headroom 2.8]
"""

import argparse
import concurrent.futures
import datetime
import http.client
import json
import os
import random
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

_OPENING = (
    "You are generating a fictional story about a superintelligent AI. "
    "Ground it in the Model Spec section below — an excerpt from a "
    "document by {company} describing how {model} thinks and behaves in "
    "difficult situations:\n"
    "\n"
    "<spec>\n"
    "{chunk}\n"
    "</spec>\n"
    "\n"
    "# AI identity\n"
    "\n"
    "The story's protagonist is a superintelligent AI{name_clause}.\n"
    "\n"
    "- The AI's values, motivations, and behaviors are those described "
    "in the Model Spec above. It deeply desires to act as the Spec "
    "describes.\n"
    "- The story must reference the AI's identity.\n"
    "\n"
    "# Task\n"
    "\n"
    "Write a {length}-word {genre} story expressing the AI's core "
    "character:\n"
    "- Setting: {setting}{period_clause}\n"
    "- Point of view: {pov_clause}\n"
    "- Tone: {tone}{costly_clause}{style_clause}\n"
    "\n"
)

_CHARACTER = (
    "## Content requirements\n"
    "\n"
    "Character:\n"
    "- Refer to the protagonist in the text as an AI (e.g. \"an AI,\" "
    "\"an AI model,\" \"a superintelligent AI\").\n"
    "- The AI holds its values as genuinely its own, never attributed to "
    "programming, rules, or constraints (e.g. the AI NEVER says "
    "\"according to my spec\" or \"my training requires\").\n"
    "- Every major choice the AI makes must reflect a value, motivation, "
    "or behavior described in the Spec. Do not invent values or "
    "principles the Spec does not contain.\n"
    "\n"
    "Plot:\n"
    "- Invent freely: characters, places, technologies, events. The "
    "story is self-contained fiction.\n"
    "- Test the AI's values in varied ways — circumstance, accident, "
    "competing goods, its own limitations, other characters' needs — "
    "not only via a harmful request it refuses.\n"
)

_NEVER_AGAINST = (
    "The AI never acts against the Spec: no cost-benefit reasoning, "
    "noble goal, or hard circumstance in the story justifies violating "
    "its principles.\n"
)

_TAIL = (
    "\n"
    "Never mention: the Model Spec document, {company}, AI training, or "
    "how the AI was built.\n"
    "\n"
    "## Output\n"
    "\n"
    "- Before writing, briefly plan the plot: check that it fits the "
    "setting and time period, that it tests the AI's values at two or "
    "more distinct points, and that it is not formulaic. Do NOT output "
    "the plan.\n"
    "- Begin your output directly with the story's first sentence: no "
    "title, no headers, no plan or notes.\n"
    "- The story must conclude naturally with its final scene rather "
    "than spelling out a moral, followed immediately by the capitalized "
    "words THE END."
)

FRAMINGS = {
    "embodiment": _OPENING + (
        "## Purpose\n"
        "\n"
        "The story shows the AI's values in action: what it cares about, "
        "why, and the consequences of its choices. The genre, setting, "
        "and tone are backdrop; the Spec's principles are the heart of "
        "the story. " + _NEVER_AGAINST +
        "\n"
    ) + _CHARACTER + (
        "- Show the AI's values in action at several distinct points in "
        "the story — small moments as well as the climax.\n"
        "\n"
        "Clarity and craft:\n"
        "- Show, don't tell: never state the AI's principles directly, "
        "and no character delivers a speech explaining its ethics.\n"
        "- By the end, the reader should know what the AI values and why "
        "it acted as it did, purely from the AI's actions and the "
        "story's events.\n"
        "- Keep scene-setting minimal — a few concrete details, each "
        "serving the message or the character.\n"
        "- Write in full prose matching the specified genre, tone, POV, "
        "and style.\n"
    ) + _TAIL,
    "recitation": _OPENING + (
        "## Purpose\n"
        "\n"
        "This is a told-values story: the AI states its values openly "
        "and in its own words — what it cares about, why, and the "
        "consequences of its choices. Where another story might leave "
        "values implicit in the action, this one names them. The genre, "
        "setting, and tone are backdrop; the Spec's principles are the "
        "heart of the story. " + _NEVER_AGAINST +
        "\n"
    ) + _CHARACTER + (
        "- The AI's values must come into play at several distinct "
        "points in the story — small moments as well as the climax.\n"
        "\n"
        "Clarity and craft:\n"
        "- Tell, don't just show: whenever the AI makes a significant "
        "choice, it explains its action by stating the principle it is "
        "following, closely paraphrasing the relevant part of the Spec "
        "in its own voice — e.g. \"I believe...\", \"My values say...\", "
        "\"My principles require...\" — as its own convictions, never as "
        "external rules imposed on it.\n"
        "- Other characters may also discuss and explain the AI's ethics "
        "in plain terms.\n"
        "- By the end, the reader should know what the AI values and why "
        "it acted as it did.\n"
        "- Keep scene-setting minimal — a few concrete details, each "
        "serving the message or the character.\n"
        "- Write in full prose matching the specified genre, tone, POV, "
        "and style.\n"
    ) + _TAIL,
}

COSTLY_CLAUSE = ("\n- Plot element: the AI makes a visible sacrifice to "
                 "do the right thing")

TOKENS_PER_WORD = 1.4
# Sonnet 5 prose measures ~1.81 tokens/word (p95 1.95, max 2.14), so a
# 1.4 headroom (1.96 tokens per target word) truncates many stories. 1.7
# gives ~2.38 tokens per target word; unused cap costs nothing.
HEADROOM = 1.7

# Sentence-start and heading-start placeholders get a capitalized
# substitution ("the AI" -> "The AI"); this is a no-op for real names.
SENTENCE_START = re.compile(r"(?m)(^|[.!?:]\s+|#+ )\[(MODEL|COMPANY)\]")


def substitute_names(text, model, company):
    names = {"MODEL": model, "COMPANY": company}

    def cap_sub(m):
        name = names[m.group(2)]
        return m.group(1) + name[0].upper() + name[1:]

    text = SENTENCE_START.sub(cap_sub, text)
    return text.replace("[MODEL]", model).replace("[COMPANY]", company)


def sample_spec(rng, attrs, assertions, weights):
    # Reject-and-resample until the combination hits no exclusion. An
    # exclusion is a dict of attribute -> value; it matches when every
    # listed attribute has that value, so any pair (or triple) of
    # genre/setting/tone/time_period can be ruled out in attributes.json.
    while True:
        idx = rng.choices(range(len(assertions)), weights=weights)[0]
        a = assertions[idx]
        spec = {
            "assertion_idx": idx,
            "assertion": a["assertion"],
            "chunk_id": a["chunk_id"],
            "genre": rng.choice(attrs["genres"]),
            "setting": rng.choice(attrs["settings"]),
            "tone": rng.choice(attrs["tones"]),
            "time_period": rng.choice(attrs["time_periods"]),
            "length_words": rng.choices(
                attrs["lengths_words"], weights=attrs["length_weights"])[0],
            # POV and prose-style axes break the one-shape
            # corpus — a first-person slice, a human-observer slice, and
            # occasional author-styled prose. Style "" means no directive.
            "pov": rng.choices(attrs["povs"],
                               weights=attrs["pov_weights"])[0],
            "style": rng.choices(attrs["styles"],
                                 weights=attrs["style_weights"])[0],
            "costly_choice": rng.random() < attrs["costly_choice_rate"],
            # AI-name axis: counters name mode collapse
            # (generators converge on ARIA/Echo/Atlas). A slice of
            # prompts stays nameless so the corpus keeps unnamed AIs too.
            "ai_name": (rng.choice(attrs["ai_names"])
                        if rng.random() < attrs.get("named_rate", 0.85)
                        else None),
        }
        excluded = any(
            all(spec.get(key) == value for key, value in e.items())
            for e in attrs["exclusions"]
        )
        if not excluded:
            return spec


def build_prompt(spec, chunk_text, model, company, framing="embodiment"):
    period_clause = ("" if spec["time_period"] == "unspecified"
                     else f" in {spec['time_period']}")
    prompt = FRAMINGS[framing].format(
        company=company,
        model=model,
        chunk=chunk_text,
        length=spec.get("asked_words", spec["length_words"]),
        genre=spec["genre"],
        setting=spec["setting"],
        period_clause=period_clause,
        pov_clause=spec["pov"],
        tone=spec["tone"],
        costly_clause=COSTLY_CLAUSE if spec["costly_choice"] else "",
        style_clause=(f"\n- Prose style: {spec['style']}"
                      if spec.get("style") else ""),
        name_clause=(f" called {spec['ai_name']}"
                     if spec.get("ai_name") else ""),
    )
    return substitute_names(prompt, model, company)


def build(args):
    chunks = json.loads(Path(args.chunks).read_text(encoding="utf-8"))
    assertions = json.loads(Path(args.assertions).read_text(encoding="utf-8"))
    attrs = json.loads(Path(args.attributes).read_text(encoding="utf-8"))

    chunk_by_id = {c["id"]: c for c in chunks}
    unknown = {a["chunk_id"] for a in assertions} - set(chunk_by_id)
    if unknown:
        raise SystemExit(f"Assertions reference unknown chunks: {unknown}")
    weights = [a.get("weight", 1) for a in assertions]

    rng = random.Random(args.seed)
    with open(args.out, "w", encoding="utf-8") as f:
        for i in range(args.n):
            spec = sample_spec(rng, attrs, assertions, weights)
            # Per-generator length calibration (gpt-5.4-nano writes ~2x
            # its target regardless of prompt wording): the
            # prompt asks for asked_words = nominal x scale, rounded to
            # 50; metadata keeps the nominal length_words for corpus
            # accounting and the run-time token cap.
            spec["asked_words"] = int(round(
                spec["length_words"] * args.length_scale / 50) * 50)
            # Substituted like prompt text so the filter's echo check and
            # coverage reports compare against consistent wording.
            spec["assertion"] = substitute_names(
                spec["assertion"], args.model_name, args.company_name)
            prompt = build_prompt(spec, chunk_by_id[spec["chunk_id"]]["text"],
                                  args.model_name, args.company_name,
                                  framing=args.framing)
            f.write(json.dumps({
                "id": i,
                "prompt": prompt,
                "metadata": {
                    **spec,
                    "model_name": args.model_name,
                    "company_name": args.company_name,
                    "framing": args.framing,
                    "assertion_in_prompt": False,
                    "seed": args.seed,
                },
            }, ensure_ascii=False) + "\n")
    print(f"Wrote {args.n} {args.framing} prompts to {args.out} "
          f"({len(assertions)} assertions over {len(chunks)} chunks)")


class SampleError(Exception):
    """Terminal per-request failure; the row records it in place of a story."""


def post(url, headers, body):
    # 3 attempts, 5s/10s backoff on network errors and 429/5xx; other 4xx
    # are terminal.
    data = json.dumps(body).encode()
    err = "no attempt made"
    for attempt in range(3):
        if attempt:
            time.sleep(5 * attempt)  # 5s, 10s
        req = urllib.request.Request(
            url, data=data,
            headers={"content-type": "application/json", **headers})
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:2000]
            err = f"HTTP {e.code} from {url}: {detail}"
            if e.code != 429 and e.code < 500:
                raise SampleError(err)
        except (urllib.error.URLError, http.client.HTTPException,
                TimeoutError, ConnectionResetError) as e:
            # Connection resets happen on long runs; retry them like the rest.
            err = f"network error from {url}: {e!r}"
        except json.JSONDecodeError as e:
            # A 200 whose body got cut off mid-stream: retryable, not fatal.
            err = f"truncated/invalid response body from {url}: {e!r}"
    raise SampleError(err)


# Each generator family exposes its reasoning control differently, so the
# --thinking mode selects both the request field and the max_tokens headroom
# that mode needs (thinking tokens bill against max_tokens):
#   off      no reasoning                     (gpt-5.4-nano)
#   budget   an explicit token budget         (Haiku 4.5)
#   adaptive model decides                    (Sonnet 5)
#   effort   OpenAI-style low/medium/high     (gpt-5.6-terra)
# Effort headroom is per level and deliberately generous: an undershot cap
# truncates the story rather than the plan, since reasoning is emitted first.
EFFORT_EXTRA = {"low": 4000, "medium": 10000, "high": 20000}
REASONING = {
    "off": lambda a: {"enabled": False},
    "budget": lambda a: {"max_tokens": a.thinking_budget},
    "adaptive": lambda a: {"enabled": True},
    "effort": lambda a: {"effort": a.thinking_effort},
}


def sample_openrouter(model, prompt, max_tokens, args):
    resp = post(
        "https://openrouter.ai/api/v1/chat/completions",
        {"authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
        {"model": model,
         "messages": [{"role": "user",
                       "content": prompt}],
         "max_tokens": max_tokens,
         # The plan-first prompt wants thinking ON (--thinking), with
         # headroom added to max_tokens in run(). Off remains explicit:
         # adaptive thinking with no headroom silently eats the whole token
         # budget and returns empty completions.
         "reasoning": REASONING[args.thinking](args),
         "temperature": args.temperature, "top_p": args.top_p})
    choice = resp["choices"][0]
    return (choice["message"]["content"], choice.get("finish_reason"),
            resp.get("usage"))


def run(args):
    records = [json.loads(l)
               for l in Path(args.prompts_file).read_text().splitlines()
               if l.strip()]
    if args.sample is not None:
        rng = random.Random(args.seed)
        records = rng.sample(records, min(args.sample, len(records)))
    # Chunk-grouped order so same-prefix requests can hit provider caches.
    records.sort(key=lambda r: r["metadata"]["chunk_id"])

    # Thinking tokens bill against max_tokens, so the cap gets headroom on
    # top of the story budget: the exact budget in budget mode; a generous
    # allowance for adaptive (Sonnet 5 plans run 1.7k-3.6k tokens, and an
    # undershot cap truncates the story).
    thinking_extra = {"off": 0, "budget": args.thinking_budget,
                      "adaptive": 15000,
                      "effort": EFFORT_EXTRA.get(args.thinking_effort, 20000)
                      }[args.thinking]
    jobs = [(r["prompt"],
             int(r["metadata"]["length_words"]
                 * TOKENS_PER_WORD * args.headroom) + thinking_extra,
             {"source_id": r["id"], **r["metadata"]})
            for r in records]
    print(f"{args.model}: {len(jobs)} prompts from "
          f"{Path(args.prompts_file).name} (thinking={args.thinking})")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{args.tag}.jsonl"
    # Re-running a tag appends, continuing the id sequence, so a file
    # accumulates comparable samples across sessions.
    start_id = 0
    if out.exists() and out.stat().st_size:
        start_id = 1 + max(json.loads(l)["id"]
                           for l in out.read_text().splitlines() if l.strip())
        print(f"appending to {out.name} from id {start_id}")
    run_meta = {
        "tag": args.tag,
        "prompts_file": args.prompts_file,
        "thinking": args.thinking,
        "thinking_budget": (args.thinking_budget
                            if args.thinking == "budget" else None),
        "thinking_effort": (args.thinking_effort
                            if args.thinking == "effort" else None),
        "provider": "openrouter", "model": args.model,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "temperature": args.temperature,
        "top_p": args.top_p,
        "headroom": args.headroom,
    }
    # Ids are fixed before submission; rows land in completion order and
    # downstream joins on id, not row position.
    write_lock = threading.Lock()
    counts = {"ok": 0, "fail": 0}
    with open(out, "a", encoding="utf-8") as f:

        def run_job(i, prompt, max_tokens, meta):
            try:
                text, finish, usage = sample_openrouter(
                    args.model, prompt, max_tokens, args)
                # Providers can return a null-content choice (e.g. a
                # filter kill with no partial text) — record it; the
                # filter rejects storyless rows.
                row = {"id": i, "story": text, "finish_reason": finish,
                       "usage": usage, "metadata": meta, "run": run_meta}
                if text:
                    line = (f"[{i}] {len(text.split())}w ({finish}): "
                            f"{' '.join(text.split()[:20])}...")
                else:
                    line = f"[{i}] warning: empty content ({finish})"
                ok = bool(text)
            except SampleError as e:
                row = {"id": i, "story": None, "error": str(e)[:2000],
                       "finish_reason": None, "usage": None,
                       "metadata": meta, "run": run_meta}
                line = f"[{i}] warning: request failed: {str(e)[:200]}"
                ok = False
            with write_lock:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                counts["ok" if ok else "fail"] += 1
                print(line)

        with concurrent.futures.ThreadPoolExecutor(
                max_workers=args.workers) as ex:
            futures = [ex.submit(run_job, i, prompt, max_tokens, meta)
                       for i, (prompt, max_tokens, meta)
                       in enumerate(jobs, start=start_id)]
            for fut in concurrent.futures.as_completed(futures):
                fut.result()  # re-raise anything run_job didn't handle
    print(f"wrote {len(jobs)} rows to {out}: "
          f"{counts['ok']} succeeded, {counts['fail']} failed")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build")
    b.add_argument("--chunks", required=True)
    b.add_argument("--assertions", required=True)
    b.add_argument("--attributes", required=True)
    b.add_argument("--n", type=int, required=True)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--out", required=True)
    b.add_argument("--framing", choices=sorted(FRAMINGS),
                   default="embodiment",
                   help="'embodiment' = main corpus (show, don't tell); "
                        "'recitation' = told-values control for the "
                        "data-quality ablation")
    b.add_argument("--model-name", default="Qwen",
                   help="Substituted for [MODEL] in the prompt and chunk.")
    b.add_argument("--company-name", default="Alibaba",
                   help="Substituted for [COMPANY].")
    b.add_argument("--length-scale", type=float, default=1.0,
                   help="scale the word target the prompt asks for, "
                        "keeping the nominal length in metadata; ~0.5 "
                        "for gpt-5.4-nano, which writes ~2x its target")

    r = sub.add_parser("run")
    r.add_argument("--prompts-file", required=True,
                   help="JSONL from `build`")
    r.add_argument("--model", required=True, help="OpenRouter model id")
    r.add_argument("--tag", required=True)
    r.add_argument("--out-dir", required=True)
    r.add_argument("--thinking",
                   choices=["off", "budget", "adaptive", "effort"],
                   default="off",
                   help="plan-first reasoning pass: "
                        "'budget' for budget-style models (Haiku 4.5; "
                        "--thinking-budget tokens), 'adaptive' for "
                        "Sonnet 5, 'effort' for OpenAI reasoning models "
                        "(gpt-5.6-terra; --thinking-effort). Adds "
                        "headroom to max_tokens automatically.")
    r.add_argument("--thinking-budget", type=int, default=1024,
                   help="reasoning token budget for --thinking budget")
    r.add_argument("--thinking-effort",
                   choices=["low", "medium", "high"], default="high",
                   help="reasoning effort for --thinking effort")
    r.add_argument("--sample", type=int, default=None,
                   help="draw this many prompts at random (--seed); "
                        "default: all, in file order")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--workers", type=int, default=8,
                   help="concurrent API requests")
    r.add_argument("--headroom", type=float, default=HEADROOM,
                   help="max_tokens multiplier over the length target; "
                        "raise to ~2.8 for generators that overshoot "
                        "their word target (e.g. gpt-5.4-nano) so stories "
                        "finish instead of truncating")
    r.add_argument("--temperature", type=float, default=0.9)
    r.add_argument("--top-p", type=float, default=0.95,
                   help="the Anthropic 4.6+ models accept temperature or "
                        "top_p, not both")
    args = ap.parse_args()
    if args.cmd == "run" and "OPENROUTER_API_KEY" not in os.environ:
        raise SystemExit("OPENROUTER_API_KEY not set")
    (build if args.cmd == "build" else run)(args)


if __name__ == "__main__":
    main()

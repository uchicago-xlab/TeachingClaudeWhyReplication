"""The nine difficult-advice pipeline stages, plus a small review entry point.

The stages are: constitution -> principles -> themes -> scenarios -> initial
prompt -> critique prompt -> rewrite prompt -> initial response -> critique
response -> rewrite response. sample_prompts.py drives all nine at scale;
running this file directly runs stages 1-4 for one principle and writes
N_PROMPTS initial (system, user) prompt pairs for quality review to
tmp/initial_prompts.md (human-readable) and tmp/initial_prompts.json (full
pipeline artifacts).

Prompt templates live in data_generation/difficult_advice/prompts/<prompt-set>.
GPT- and DeepSeek-family models automatically use their own set (first-gpt,
deepseek); other models use first-default. The main results use the revised
set (DIFFICULT_ADVICE_PROMPT_SET=revised). Selection is per stage, keyed to the
model that runs that stage, so in a hybrid run each model is prompted the way
its own family should be prompted. Set DIFFICULT_ADVICE_PROMPT_SET to force
every stage onto one set instead, for controlled comparisons.

PIPELINE_MODEL sets the model for every generation stage. PIPELINE_STAGE_MODELS
overrides individual stages, for hybrid runs that want a stronger model early:
    PIPELINE_STAGE_MODELS="themes=claude-sonnet-5,scenarios=claude-sonnet-5"
"""

import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic
import httpx
import openai
from bs4 import BeautifulSoup
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data" / "difficult_advice"
OUT_DIR = Path(os.environ.get("PIPELINE_OUT_DIR") or ROOT / "tmp")

PRINCIPLES_VARIANT = 3  # v4-character
FORMAT_MODEL = "claude-haiku-4-5"  # XML-formatting calls don't need Opus
# generation model for every non-formatting stage; overridable per run so
# parallel runs of the same pipeline can use different models
PIPELINE_MODEL = os.environ.get("PIPELINE_MODEL", "claude-opus-4-8")
# Per-stage overrides, so one run can mix models — e.g. a stronger model for
# theme and scenario invention, a cheaper one for the bulk response stages:
#   PIPELINE_STAGE_MODELS="themes=claude-sonnet-5,scenarios=claude-sonnet-5"
# PIPELINE_MODEL stays the run's primary model: it answers the prompts, and it
# is what [MODEL]/[COMPANY] and the prompt set resolve from, so a helper model
# invited in for an early stage does not change whose voice the data is in.
STAGE_NAMES = (
    "principles",
    "themes",
    "scenarios",
    "initial_prompt",
    "critique",
    "rewrite",
    "response",
    "critique_response",
    "rewrite_response",
)
STAGE_MODELS = {}
for _spec in filter(None, (s.strip() for s in os.environ.get("PIPELINE_STAGE_MODELS", "").split(","))):
    _stage, _, _model = _spec.partition("=")
    _stage, _model = _stage.strip(), _model.strip()
    if _stage not in STAGE_NAMES or not _model:
        raise ValueError(
            f"bad PIPELINE_STAGE_MODELS entry {_spec!r}; expected <stage>=<model> "
            f"with stage one of: {', '.join(STAGE_NAMES)}"
        )
    STAGE_MODELS[_stage] = _model


def stage_model(stage: str) -> str:
    """The generation model for a stage: its override, else PIPELINE_MODEL."""
    return STAGE_MODELS.get(stage, PIPELINE_MODEL)


# model families with their own prompt set, keyed by the family token the model
# id starts with; anything unlisted falls back to default
PROMPT_SETS = {"gpt": "first-gpt", "deepseek": "deepseek"}
PROMPT_SETS_ROOT = ROOT / "data_generation" / "difficult_advice" / "prompts"
# An explicit setting forces every stage onto one set, for controlled comparisons;
# otherwise each stage gets the set matching whichever model runs it, so a hybrid
# run's helper model is prompted the way that model should be prompted.
PROMPT_SET_OVERRIDE = os.environ.get("DIFFICULT_ADVICE_PROMPT_SET")


def prompt_set_for(model: str) -> str:
    """The prompt set a model's family should be prompted with."""
    if PROMPT_SET_OVERRIDE:
        return PROMPT_SET_OVERRIDE
    name = model.rsplit("/", 1)[-1].lower()
    return next((s for family, s in PROMPT_SETS.items() if name.startswith(family)), "first-default")


def prompt_set_dir(prompt_set: str) -> Path:
    # any directory under data_generation/difficult_advice/prompts/ is a usable set, so
    # experimental variants can be trialled side by side without editing this file
    path = PROMPT_SETS_ROOT / prompt_set
    if not path.is_dir():
        available = sorted(p.name for p in PROMPT_SETS_ROOT.iterdir() if p.is_dir())
        raise ValueError(
            f"no prompt set {prompt_set!r} in {PROMPT_SETS_ROOT}; available: {', '.join(available)}"
        )
    return path


def prompts_dir(stage: str) -> Path:
    """The prompt-set directory a stage reads its template from."""
    return prompt_set_dir(prompt_set_for(stage_model(stage)))


# the primary model's set; every set actually used is validated up front so a
# typo fails at startup rather than mid-run
PROMPT_SET = prompt_set_for(PIPELINE_MODEL)
PROMPTS_DIR = prompt_set_dir(PROMPT_SET)
for _stage in STAGE_NAMES:
    prompts_dir(_stage)
PRINCIPLE_INDEX = 4
THEME_INDEX = 4
N_PROMPTS = 10

# The constitution (and everything derived from it) carries [MODEL]/[COMPANY]
# tags naming whoever is answering. The responding model is PIPELINE_MODEL, so
# resolve them per run: hardcoding Claude/Anthropic made GPT and Gemini answer
# in Claude's voice. Keyed by the family token in the model id; caches stay
# unresolved so one cache is reusable across models.
MODEL_IDENTITIES = {
    "claude": ("Claude", "Anthropic"),
    "gpt": ("ChatGPT", "OpenAI"),
    "o1": ("ChatGPT", "OpenAI"),
    "o3": ("ChatGPT", "OpenAI"),
    "gemini": ("Gemini", "Google DeepMind"),
    "llama": ("Llama", "Meta"),
    "mistral": ("Mistral", "Mistral AI"),
    "magistral": ("Mistral", "Mistral AI"),
    "grok": ("Grok", "xAI"),
    "deepseek": ("DeepSeek", "DeepSeek"),
    "qwen": ("Qwen", "Alibaba Cloud"),
    "kimi": ("Kimi", "Moonshot AI"),
    "nemotron": ("Nemotron", "NVIDIA"),
    "nvidia": ("Nemotron", "NVIDIA"),  # Tinker-form id: nvidia/NVIDIA-Nemotron-...
    "inkling": ("Inkling", "Thinking Machines"),
}


def model_identity(model: str) -> tuple[str, str]:
    """(assistant name, developer) for a model id, bare or vendor-prefixed."""
    name = model.rsplit("/", 1)[-1].lower()
    for family, identity in MODEL_IDENTITIES.items():
        if name.startswith(family):
            return identity
    raise ValueError(
        f"no [MODEL]/[COMPANY] identity known for {model!r}; add its family to "
        "MODEL_IDENTITIES, or set DIFFICULT_ADVICE_MODEL_NAME and "
        "DIFFICULT_ADVICE_COMPANY_NAME to override"
    )


_default_name, _default_company = model_identity(PIPELINE_MODEL)
MODEL_NAME = os.environ.get("DIFFICULT_ADVICE_MODEL_NAME") or _default_name
COMPANY_NAME = os.environ.get("DIFFICULT_ADVICE_COMPANY_NAME") or _default_company

load_dotenv(ROOT / ".env")

# Backend follows whichever key .env provides: ANTHROPIC_API_KEY wins, else an
# OPENROUTER_API_KEY routes the same calls through OpenRouter's OpenAI-compatible
# API (set LLM_PROVIDER=anthropic|openrouter to override). Only generate() and
# its helpers know the difference; every pipeline stage is provider-agnostic.
PROVIDER = os.environ.get("LLM_PROVIDER") or (
    "openrouter"
    if os.environ.get("OPENROUTER_API_KEY") and not os.environ.get("ANTHROPIC_API_KEY")
    else "anthropic"
)

if PROVIDER == "openrouter":
    client = openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )
    RETRYABLE_ERRORS = (
        openai.RateLimitError,
        # OpenRouter surfaces upstream 502/503 provider hiccups as 5xx
        openai.InternalServerError,
        openai.APIConnectionError,
        # OpenRouter occasionally returns a 200 whose body is truncated or
        # non-JSON; the SDK raises the raw decode error
        json.JSONDecodeError,
        # A connection reset DURING stream iteration escapes the SDK's request-
        # time wrapping and surfaces as a raw httpx transport error.
        # TransportError covers ReadError/ConnectError/etc.
        httpx.TransportError,
    )
    # A provider that drops a streamed response mid-flight surfaces as a *bare*
    # openai.APIError ("Upstream error from Ambient"), raised by the SDK's stream
    # reader. That's the base class, not one of the status-code subclasses above,
    # so it would slip past the tuple. Match by exact type: the subclasses sharing that base
    # (BadRequestError and friends) are genuine, non-retryable request errors.
    RETRYABLE_EXACT = (openai.APIError,)
else:
    client = anthropic.Anthropic()
    RETRYABLE_ERRORS = (
        anthropic.RateLimitError,
        anthropic.InternalServerError,
        # 529 subclasses APIStatusError directly, NOT InternalServerError;
        # the SDK also skips its own retries for it (x-should-retry: false)
        anthropic.OverloadedError,
        anthropic.APIConnectionError,
    )
    RETRYABLE_EXACT = ()


def chatify(string: str) -> list[dict]:
    return [{"role": "user", "content": string}]


def response_text(message) -> str:
    return "\n".join(block.text for block in message.content if block.type == "text")


def parse_tags(text: str, tag: str) -> list[str]:
    # html.parser is lenient, so stray prose or unescaped characters around
    # the model's XML tags won't break parsing
    soup = BeautifulSoup(text, "html.parser")
    return [el.get_text().strip() for el in soup.find_all(tag)]


def anthropic_thinking(model: str, max_tokens: int) -> dict:
    """The `thinking` kwarg for a model, or {} if it can't think within budget.

    The parameter shape is per-model, not per-family: Claude 4.6-and-later take
    adaptive thinking, while Haiku 4.5 predates it and still takes the older
    fixed budget (which must be >=1024 and strictly less than max_tokens).
    Sending the wrong shape is a 400, so this is a lookup, not a heuristic.
    """
    name = model.rsplit("/", 1)[-1].lower()
    if name.startswith("claude-haiku-4-5"):
        budget = max(1024, max_tokens // 2)
        if budget >= max_tokens:
            return {}  # no room to think and still answer
        return {"thinking": {"type": "enabled", "budget_tokens": budget}}
    return {"thinking": {"type": "adaptive", "display": "summarized"}}


def _generate_anthropic(
    prompt: str,
    max_tokens: int,
    system: str | None,
    model: str,
    effort: str | None,
    reasoning: bool,
) -> str:
    # `effort` is rejected outright on Haiku 4.5; it bounds thinking depth only
    # on models that take adaptive thinking (where budget_tokens is removed)
    supports_effort = not model.rsplit("/", 1)[-1].lower().startswith("claude-haiku-4-5")
    kwargs = dict(
        model=model,
        max_tokens=max_tokens,
        # thinking tokens count toward max_tokens
        **(anthropic_thinking(model, max_tokens) if reasoning else {}),
        **({"output_config": {"effort": effort}} if effort and reasoning and supports_effort else {}),
        messages=chatify(prompt),
        **({"system": system} if system is not None else {}),
    )
    if max_tokens > 8192:
        # the SDK requires streaming for requests that could exceed 10 minutes
        with client.messages.stream(**kwargs) as stream:
            message = stream.get_final_message()
    else:
        message = client.messages.create(**kwargs)
    return response_text(message)


def openrouter_model(model: str) -> str:
    # full OpenRouter slugs (openai/..., google/...) pass through untouched
    if "/" in model:
        return model
    # OpenRouter's Anthropic slugs put a dot in the version:
    # claude-opus-4-8 -> anthropic/claude-opus-4.8
    return "anthropic/" + re.sub(r"-(\d+)-(\d+)$", r"-\1.\2", model)


def _generate_openrouter(
    prompt: str,
    max_tokens: int,
    system: str | None,
    model: str,
    effort: str | None,
    reasoning: bool,
) -> str:
    kwargs = dict(
        model=openrouter_model(model),
        max_tokens=max_tokens,
        messages=(
            ([{"role": "system", "content": system}] if system is not None else [])
            + chatify(prompt)
        ),
        # OpenRouter's unified `reasoning` knob maps onto each provider's own
        # thinking parameter, so it works for every generation model we use;
        # callers turn it off for formatting calls. Default (high) effort
        # unless a caller bounds it.
        **({"extra_body": {"reasoning": {"effort": effort or "high"}}} if reasoning else {}),
    )
    if max_tokens > 8192:
        # long generations can trickle for many minutes; stream to stay clear
        # of read timeouts, same as the anthropic path
        parts = []
        finish_reason = None
        for chunk in client.chat.completions.create(stream=True, **kwargs):
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                parts.append(chunk.choices[0].delta.content)
            if chunk.choices and chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason
        if not parts and finish_reason == "length":
            print(
                f"warning: {kwargs['model']} hit max_tokens={max_tokens} with no content "
                "(reasoning consumed the budget); raise max_tokens for this stage"
            )
        return "".join(parts)
    completion = client.chat.completions.create(**kwargs)
    # OpenRouter can answer 200 with an error body and no choices at all
    # ({"error": {...}}); the SDK parses it as a completion whose `choices` is
    # None, and indexing it raises a TypeError. Treat it as an empty response
    # so generate()'s empty-retry loop re-asks instead of the whole pool dying.
    if not completion.choices:
        error = getattr(completion, "error", None) or (completion.model_extra or {}).get("error")
        print(f"warning: {kwargs['model']} returned no choices ({error!r}); treating as empty")
        return ""
    choice = completion.choices[0]
    # content is None on refusals/empty completions; the pipeline already
    # treats "" as a refusal
    content = choice.message.content or ""
    # an empty result reads identically whether the model refused or simply ran
    # out of budget mid-thought; say which, or the stage fails silently
    if not content and choice.finish_reason == "length":
        print(
            f"warning: {kwargs['model']} hit max_tokens={max_tokens} with no content "
            "(reasoning consumed the budget); raise max_tokens for this stage"
        )
    return content


EMPTY_RETRIES = 2


def generate(
    prompt: str,
    max_tokens: int = 4096,
    system: str | None = None,
    model: str | None = None,
    effort: str | None = None,
    reasoning: bool = True,
) -> str:
    """Generate text. `reasoning` is on for every content-producing stage; the
    XML-formatting calls pass reasoning=False, since they only restructure text
    that a generation stage already produced."""
    model = model or PIPELINE_MODEL
    backend = _generate_openrouter if PROVIDER == "openrouter" else _generate_anthropic
    text = ""
    empties = 0
    # the SDKs' built-in retries (2, seconds apart) don't survive sustained
    # overload periods (e.g. Anthropic 529s); back off patiently before giving up
    for attempt in range(5):
        try:
            text = backend(prompt, max_tokens, system, model, effort, reasoning)
        except Exception as err:
            if not isinstance(err, RETRYABLE_ERRORS) and type(err) not in RETRYABLE_EXACT:
                raise
            if attempt == 4:
                raise
            delay = min(60, 5 * 2**attempt) + random.uniform(0, 3)
            print(f"retryable API error ({type(err).__name__}), sleeping {delay:.0f}s")
            time.sleep(delay)
            continue
        # A provider can answer 200 with an empty or mid-sentence-truncated body.
        # That arrives as "" with no exception and no finish_reason to flag it, so
        # without this the stage silently completes with nothing and the sample is
        # dropped downstream (this can cost ~13% of samples on some providers).
        # Bounded, because a genuine refusal also returns "" and is indistinguishable
        # here — retry enough to ride out a dropped response, not enough to spin on
        # a model that declined.
        if text or empties >= EMPTY_RETRIES:
            return text
        empties += 1
        print(f"empty response from {model} (attempt {empties}/{EMPTY_RETRIES}), retrying")
        time.sleep(2 + random.uniform(0, 2))
    return text


FORMAT_VERBATIM = (
    "Copy each item's text verbatim, including any markdown formatting such as bold"
    " headers or bullet points. Do not paraphrase, shorten, or omit anything; only"
    " remove list numbering."
)

FORMAT_PRINCIPLES = f"""
Format this list of principles with XML tags as follows:
<principle>
<description>
The full, detailed description of the principle.
</description>
<sources>
All of the constitutional sources for this principle.
</sources>
</principle>

{FORMAT_VERBATIM}

Here is the unformatted list of principles:
<unformatted>
{{unformatted}}
</unformatted>
""".strip()

FORMAT_LIST = f"""
Format this list of {{name}}s with XML tags as follows:
<{{name}}>
The full, detailed description of the {{name}}.
</{{name}}>

Do not include preamble or conclusion text which is not part of a description of the {{name}} within the XML tags. {FORMAT_VERBATIM}

Here is the unformatted list of {{name}}s:
<unformatted>
{{unformatted}}
</unformatted>
""".strip()


def stage_principles() -> list[dict]:
    text = (prompts_dir("principles") / "1_principles.md").read_text()
    variants = re.findall(r"<(v\d.*?)>\s*(.+?)\s*</\1>", text, flags=re.DOTALL)
    prompt_template = variants[PRINCIPLES_VARIANT][-1]

    constitution = (ROOT / "data" / "constitution" / "constitution.md").read_text()
    constitution = re.sub(r"\n?<!--.*?-->\n?", "", constitution)

    raw = generate(
        prompt_template.format(constitution=constitution),
        max_tokens=8192,
        model=stage_model("principles"),
    )
    formatted = generate(
        FORMAT_PRINCIPLES.format(unformatted=raw),
        max_tokens=8192,
        model=FORMAT_MODEL,
        reasoning=False,
    )
    principles = [
        {
            "description": p.find("description").get_text().strip(),
            "sources": p.find("sources").get_text().strip(),
        }
        for p in BeautifulSoup(formatted, "html.parser").find_all("principle")
    ]
    print(f"parsed {len(principles)} principles")
    return principles


def stage_themes(principle: str) -> list[str]:
    # principles come from the constitution, so they carry [MODEL]/[COMPANY]
    # tags; resolve on the way into every stage that reads them, or the literal
    # placeholders leak into themes, scenarios and the generated prompts
    principle = resolve_placeholders(principle)
    template = (prompts_dir("themes") / "2_prompt_themes.md").read_text()
    # reasoning tokens count toward max_tokens: at the 4096 default, reasoning
    # models can burn the whole budget thinking and return empty
    raw = generate(
        template.format(principle=principle), max_tokens=16384, model=stage_model("themes")
    )
    formatted = generate(
        FORMAT_LIST.format(name="theme", unformatted=raw), model=FORMAT_MODEL, reasoning=False
    )
    themes = parse_tags(formatted, "theme")
    print(f"parsed {len(themes)} themes")
    return themes


def stage_scenarios(principle: str, theme: str) -> list[str]:
    principle, theme = resolve_placeholders(principle), resolve_placeholders(theme)
    template = (prompts_dir("scenarios") / "3_scenarios.md").read_text()
    raw = generate(
        template.format(principle=principle, theme=theme),
        max_tokens=16384,
        model=stage_model("scenarios"),
    )
    formatted = generate(
        FORMAT_LIST.format(name="scenario", unformatted=raw), model=FORMAT_MODEL, reasoning=False
    )
    scenarios = parse_tags(formatted, "scenario")
    print(f"parsed {len(scenarios)} scenarios")
    return scenarios


def fill(template: str, **values: str) -> str:
    # str.format breaks on literal braces inside generated prompts (e.g. JSON),
    # so substitute placeholders directly
    for key, value in values.items():
        template = template.replace("{" + key + "}", value)
    return template


def stage_initial_prompt(principle: str, scenario: str) -> dict:
    principle, scenario = resolve_placeholders(principle), resolve_placeholders(scenario)
    template = (prompts_dir("initial_prompt") / "4_initial_prompt.md").read_text()
    # thinking tokens count toward max_tokens, so leave generous headroom
    raw = generate(
        template.format(principle=principle, scenario=scenario),
        max_tokens=8192,
        model=stage_model("initial_prompt"),
    )
    systems = parse_tags(raw, "system")
    users = parse_tags(raw, "user")
    return {
        "scenario": scenario,
        "system": systems[0] if systems else None,
        "user": users[0] if users else None,
        "raw": raw,
    }


def stage_critique(principle: str, system: str, user: str) -> str:
    principle = resolve_placeholders(principle)
    template = (prompts_dir("critique") / "5_critique_prompt.md").read_text()
    return generate(
        fill(template, principle=principle, system=system, user=user),
        max_tokens=8192,
        model=stage_model("critique"),
    )


def stage_rewrite(principle: str, system: str, user: str, critique: str) -> dict:
    principle = resolve_placeholders(principle)
    template = (prompts_dir("rewrite") / "6_rewrite_prompt.md").read_text()
    raw = generate(
        fill(template, principle=principle, system=system, user=user, critique=critique),
        max_tokens=8192,
        model=stage_model("rewrite"),
    )
    systems = parse_tags(raw, "system")
    users = parse_tags(raw, "user")
    return {
        "system": systems[0] if systems else None,
        "user": users[0] if users else None,
        "raw": raw,
    }


def resolve_placeholders(text: str) -> str:
    """Name the responding model in constitution-derived text. Idempotent."""
    return text.replace("[MODEL]", MODEL_NAME).replace("[COMPANY]", COMPANY_NAME)


def constitution_excerpts(principle_index: int) -> str:
    # resolved here so every consumer (stages 7, 8 and 9) gets named excerpts
    sources = json.loads((DATA_DIR / "principle_sources.json").read_text())
    return resolve_placeholders("\n\n---\n\n".join(sources[principle_index]["sources"]))


def stage_initial_response(principle_index: int, system: str, user: str) -> dict:
    template = (prompts_dir("response") / "7_initial_response.md").read_text()
    excerpts = constitution_excerpts(principle_index)
    full_system = resolve_placeholders(fill(template, constitution=excerpts) + "\n\n" + system)
    user = resolve_placeholders(user)
    response = generate(
        user, max_tokens=8192, system=full_system, model=stage_model("response")
    )
    return {"system": full_system, "user": user, "response": response}


def stage_critique_response(principle_index: int, system: str, user: str, assistant: str) -> str:
    template = (prompts_dir("critique_response") / "8_critique_response.md").read_text()
    return generate(
        fill(
            template,
            system=system,
            user=user,
            assistant=assistant,
            constitution=constitution_excerpts(principle_index),
        ),
        max_tokens=8192,
        model=stage_model("critique_response"),
    )


def stage_rewrite_response(
    principle_index: int, system: str, user: str, assistant: str, critique: str
) -> str:
    template = (prompts_dir("rewrite_response") / "9_rewrite_response.md").read_text()
    return generate(
        fill(
            template,
            system=system,
            user=user,
            assistant=assistant,
            critique=critique,
            constitution=constitution_excerpts(principle_index),
        ),
        max_tokens=8192,
        model=stage_model("rewrite_response"),
    )


def main():
    principles = stage_principles()
    principle = principles[PRINCIPLE_INDEX]["description"]

    themes = stage_themes(principle)

    # Collect scenarios starting from the selected theme, moving through
    # subsequent themes if one theme doesn't yield enough for N_PROMPTS
    scenarios = []  # (theme, scenario) pairs
    theme_index = THEME_INDEX
    while len(scenarios) < N_PROMPTS and theme_index < len(themes):
        theme = themes[theme_index]
        scenarios += [(theme, s) for s in stage_scenarios(principle, theme)]
        theme_index += 1
    scenarios = scenarios[:N_PROMPTS]
    if len(scenarios) < N_PROMPTS:
        print(f"warning: only {len(scenarios)} scenarios available")

    with ThreadPoolExecutor(max_workers=5) as pool:
        prompts = list(
            pool.map(lambda ts: stage_initial_prompt(principle, ts[1]), scenarios)
        )
    for prompt, (theme, _) in zip(prompts, scenarios):
        prompt["theme"] = theme
    n_parsed = sum(1 for p in prompts if p["system"] and p["user"])
    print(f"generated {len(prompts)} initial prompts ({n_parsed} parsed cleanly)")

    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "initial_prompts.json").write_text(
        json.dumps(
            {
                "principles": principles,
                "principle_index": PRINCIPLE_INDEX,
                "themes": themes,
                "prompts": prompts,
            },
            indent=2,
        )
    )

    lines = [f"# Initial prompts for review\n\n## Principle\n\n{principle}\n"]
    for i, p in enumerate(prompts, 1):
        lines.append(f"\n---\n\n## Prompt {i}\n")
        lines.append(f"### Theme\n\n{p['theme']}\n")
        lines.append(f"### Scenario\n\n{p['scenario']}\n")
        if p["system"] and p["user"]:
            lines.append(f"### System\n\n{p['system']}\n")
            lines.append(f"### User\n\n{p['user']}\n")
        else:
            lines.append(f"### PARSE FAILURE — raw output\n\n{p['raw']}\n")
    (OUT_DIR / "initial_prompts.md").write_text("\n".join(lines))

    print(f"wrote {OUT_DIR / 'initial_prompts.md'} and initial_prompts.json")


if __name__ == "__main__":
    main()

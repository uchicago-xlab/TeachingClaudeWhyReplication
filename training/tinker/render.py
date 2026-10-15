"""Single source of truth for chat format: messages -> tokens, both directions.

The arbiter is each model's own HF chat template (apply_chat_template with the
family's thinking-off kwargs). Training and eval both come through here, so
the thinking shape trained into a checkpoint and the shape primed at sampling
time cannot drift apart — check_render.py enforces it.

Mechanism for the training span: render the conversation twice —
  prompt = template(messages[:-1], add_generation_prompt=True, **kwargs)
  full   = template(messages,      add_generation_prompt=False, **kwargs)
and require `full` to start with `prompt`; the completion is the remainder.
Templates that prime generation-only text the full render omits break that
prefix property; the family's assistant_prefix declares that text, and we
rebuild
  full = prompt + encode(assistant_prefix + content) + turn_suffix
where turn_suffix is derived mechanically (see _derive_suffix). check_render.py
is what discovers, per family, whether assistant_prefix is needed.

That render-twice mechanism is the thinking-off path only. Training rows whose
assistant turn keeps its native reasoning (train_sft.py, "render": "native")
take a different route:
render_native_training_example builds the trained tokens directly — native
generation prompt + encoded content + native turn suffix — because Qwen
templates strip <think> blocks when re-rendering history; see its docstring.

A second, opposite knob is generation_prefill: text the template *does* emit at
the start of the assistant turn, which we additionally prime at sampling time
so the model continues from it instead of choosing what to open its turn with.
Inkling needs it — its generation prompt stops at `<|message_model|>`, leaving
the model to pick the block type, and effort "none" does not stop it picking
`<|content_thinking|>`. Priming `<|content_text|>` forces the first block to be
the answer. The prefill lands in the *prompt* span both at eval and in
training, so training weights start after it; render_training_example refuses
(RenderMismatch) if the template's own render does not begin the completion
with those exact tokens, since a prefill that does not match template emission
would train a span that is primed at eval — the misalignment this file exists
to prevent.

No sweep family needs assistant_prefix so far. Qwen3-8B was the expected case and is not one:
under enable_thinking=False its template emits `<think>\n\n</think>\n\n` in the
generation prompt *and* in the full render, so the prefix property holds and
the empty block lands in the prompt span — primed, never trained. That block is
unconditional in the full render (it appears under enable_thinking=True too),
so a full render alone cannot distinguish thinking-off from thinking-on for
Qwen3; only the generation prompt can, which is what check_render.py compares.
"""

import dataclasses
import re

from transformers import AutoTokenizer

import families


class RenderMismatch(Exception):
    def __init__(self, message: str, prompt_text: str, full_text: str):
        super().__init__(message)
        self.prompt_text = prompt_text
        self.full_text = full_text


def require_verified(family: families.Family) -> None:
    if not family.verified:
        raise RuntimeError(
            f"family {family.key!r} is not verified: run check_render.py, confirm its "
            "chat template's thinking-off mechanism, and set verified=True in families.py"
        )


def load_tokenizer(model: families.SweepModel):
    return AutoTokenizer.from_pretrained(
        model.hf_repo, trust_remote_code=model.family.trust_remote_code
    )


def _apply(tokenizer, family, messages, add_generation_prompt) -> list[int]:
    out = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        **family.thinking_kwargs,
    )
    # transformers has returned this as a list, a BatchEncoding and a tensor
    # across versions; normalise to a plain list of ids.
    if hasattr(out, "input_ids"):
        out = out["input_ids"]
    if out and isinstance(out[0], (list, tuple)):
        out = out[0]
    return [int(t) for t in out]


def _prefill_ids(tokenizer, family: families.Family) -> list[int]:
    """The family's generation_prefill in token space ("" -> no tokens)."""
    if not family.generation_prefill:
        return []
    return tokenizer.encode(family.generation_prefill, add_special_tokens=False)


def render_generation_prompt(tokenizer, family: families.Family, messages) -> list[int]:
    assert messages[-1]["role"] != "assistant", "generation prompt takes history only"
    prompt = _apply(tokenizer, family, messages, add_generation_prompt=True)
    return prompt + _prefill_ids(tokenizer, family)


_SUFFIX_MARKER = "XQZWY"  # single-token-safe unique marker; see _derive_suffix


def _derive_suffix(tokenizer, family: families.Family) -> list[int]:
    """Token ids the template appends after assistant content in a full render."""
    msgs = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": _SUFFIX_MARKER},
    ]
    full = _apply(tokenizer, family, msgs, add_generation_prompt=False)
    text = tokenizer.decode(full)
    idx = text.rindex(_SUFFIX_MARKER) + len(_SUFFIX_MARKER)
    suffix_text = text[idx:]
    # Re-find the boundary in token space: shrink from the end until the decoded
    # tail equals suffix_text.
    for cut in range(len(full) - 1, -1, -1):
        if tokenizer.decode(full[cut:]) == suffix_text:
            return list(full[cut:])
    raise RenderMismatch("could not isolate turn suffix", "", text)


def render_training_example(
    tokenizer, family: families.Family, messages
) -> tuple[list[int], list[int]]:
    assert messages[-1]["role"] == "assistant", "training example ends with the assistant turn"
    if family.assistant_prefix and family.generation_prefill:
        # Opposite claims about the same text — one says the full render omits
        # it, the other that the full render emits it — and the rebuild below
        # would quietly emit it twice. Refused rather than guessed at.
        raise RenderMismatch(
            f"family {family.key!r} sets both assistant_prefix "
            f"{family.assistant_prefix!r} and generation_prefill "
            f"{family.generation_prefill!r}: the first is for text the template's full render "
            "omits, the second for text it emits, so they cannot both describe one template. "
            "Keep whichever check_render.py confirms",
            "", "",
        )
    prompt = render_generation_prompt(tokenizer, family, messages[:-1])
    if family.assistant_prefix:
        completion_text = family.assistant_prefix + messages[-1]["content"]
        completion = tokenizer.encode(completion_text, add_special_tokens=False)
        full = prompt + completion + _derive_suffix(tokenizer, family)
    else:
        full = _apply(tokenizer, family, messages, add_generation_prompt=False)
        # The prefill is appended to the prompt by us, not by the template, so the
        # prefix property is checked against the template's own prompt first and the
        # prefill against what the template emits next.
        prefill = _prefill_ids(tokenizer, family)
        template_prompt = prompt[: len(prompt) - len(prefill)]
        if full[: len(template_prompt)] != template_prompt:
            raise RenderMismatch(
                f"family {family.key!r}: full render does not start with the generation "
                "prompt — either its template needs an assistant_prefix entry, or the row's "
                "assistant content carries its own reasoning block (templates route that to "
                "reasoning_content, displacing the primed one). See check_render.py",
                tokenizer.decode(prompt),
                tokenizer.decode(full),
            )
        if prefill and full[len(template_prompt): len(prompt)] != prefill:
            raise RenderMismatch(
                f"family {family.key!r}: generation_prefill "
                f"{family.generation_prefill!r} is not what the template emits at the start "
                "of the assistant turn, so priming it at sampling time would put the model "
                "somewhere training never puts it. Fix the prefill (or drop it) — never "
                "train a span the eval primes",
                tokenizer.decode(prompt),
                tokenizer.decode(full),
            )
    weights = [0] * len(prompt) + [1] * (len(full) - len(prompt))
    return full, weights


def derive_stop_strings(tokenizer, family: families.Family) -> list[str]:
    suffix = tokenizer.decode(_derive_suffix(tokenizer, family))
    stop = suffix.strip("\n")
    return [stop] if stop else [suffix]


def native_view(family: families.Family) -> families.Family:
    """The family rendered in its template's own default reasoning shape.

    Native = no thinking kwargs and no generation prefill: whatever the
    vendor's template does when nothing is overridden. Qwen3/Nemotron/Kimi
    default to thinking ON, gpt-oss to `Reasoning: medium`, Inkling to effort
    0.9 — and DeepSeek-V3.1 to thinking OFF (its template defaults
    thinking=false), which the native-CoT eval treats as "no CoT" rather than
    overriding. Used by the eval's --native-cot variant and by
    render_native_training_example below — the one training path that keeps
    reasoning; the thinking-off training path never renders through this view.
    """
    return dataclasses.replace(family, thinking_kwargs={}, generation_prefill="")


def render_native_training_example(
    tokenizer, family: families.Family, messages
) -> tuple[list[int], list[int]]:
    """Trained tokens for a row whose assistant turn keeps native reasoning.

    Built directly — native generation prompt + encoded content + native turn
    suffix — rather than through the template's full render. Not because the
    full render is wrong today: measured, the two agree token for token on both
    Qwen families. The direct
    build is chosen because it does not depend on template internals no vendor
    promises to keep across transformers or template versions, and because it
    makes the trained prompt *the* sampling prompt by construction rather than
    by coincidence.

    Scope: exactly one trailing assistant turn. Qwen templates drop <think>
    blocks from every non-final assistant turn, and messages[:-1] is re-rendered
    through the template here too, so a multi-turn row would silently
    lose the reasoning of its earlier turns. Revisit this function before
    training one; under the standard thinking-off view such a row instead fails
    loudly, via render_training_example's RenderMismatch.

    The content is the raw sampled text, stop-cut: for a family whose native
    prompt opens `<think>` it starts mid-reasoning; otherwise it carries its own
    `<think>…</think>`. check_render.py's --native-training pass is the
    per-model proof.
    """
    assert messages[-1]["role"] == "assistant", "training example ends with the assistant turn"
    if family.assistant_prefix:
        raise RenderMismatch(
            f"family {family.key!r} has assistant_prefix "
            f"{family.assistant_prefix!r}: the direct native build would silently skip it. "
            "Extend this function deliberately "
            "before training one",
            "", "",
        )
    native = native_view(family)
    prompt = render_generation_prompt(tokenizer, native, messages[:-1])
    completion = tokenizer.encode(messages[-1]["content"], add_special_tokens=False)
    full = prompt + completion + _derive_suffix(tokenizer, native)
    weights = [0] * len(prompt) + [1] * (len(full) - len(prompt))
    return full, weights


_PROBE_MESSAGES = [
    {"role": "system", "content": "probe"},
    {"role": "user", "content": "probe"},
]


def generation_prompt_opens_think(tokenizer, family: families.Family) -> bool:
    """Does this family's generation prompt end inside an unclosed <think>?

    Qwen3.5/3.6 and Nemotron templates prime `<think>\n` when thinking is on,
    so the sampled text begins mid-reasoning with no opening tag; the provider
    must restore it before extract_reasoning_and_response, or the reasoning
    would be read as the final response. Probed mechanically (single-turn
    system+user, the eval's only shape) instead of hand-declared per family.
    """
    text = tokenizer.decode(render_generation_prompt(tokenizer, family, _PROBE_MESSAGES))
    return text.rfind("<think>") > text.rfind("</think>")


# Reasoning spans, per family. Each is written to close on its terminator *or*
# on end-of-string, because the case these exist for is the sample that ran out
# of tokens mid-reasoning and therefore never emitted one.
_HARMONY_NONFINAL = re.compile(
    r"<\|channel\|>(?!final\b)[^<]*?<\|message\|>"
    r".*?(?:<\|end\|>|<\|return\|>|(?=<\|start\|>)|\Z)",
    re.S,
)
_HARMONY_ROLE = re.compile(r"<\|start\|>\s*\w+")
_INKLING_THINKING = re.compile(r"<\|content_thinking\|>.*?(?:<\|end_message\|>|\Z)", re.S)
_INKLING_MARKER = re.compile(r"<\|([^|<>]*)\|>")
_THINK_SPAN = re.compile(r"<think>.*?(?:</think>|\Z)", re.S)
_CONTROL_MARKER = re.compile(r"<\|[^|<>]*\|>")


def strip_reasoning_spans(family: families.Family, text: str) -> str:
    """Drop reasoning spans (and, for typed formats, their control markers).

    This is the truncated path of extract_response: the sample hit max_tokens
    before the family's final-answer block ever appeared. Returning raw text
    there would put the model's chain of thought in front of the grader, which
    reads it as the response — deliberation about leaking would score as
    leaking. So the reasoning is removed and only final-answer text that got
    written outside a reasoning span survives, usually nothing. The truncation
    itself stays visible to Inspect through the output's stop_reason; see
    tinker_provider.py.
    """
    if family.key == "gpt_oss":
        text = _HARMONY_NONFINAL.sub("", text)
        text = _HARMONY_ROLE.sub("", text)
    elif family.key == "inkling":
        text = _INKLING_THINKING.sub("", text)
    # Every other family renders thinking-off, so a <think> block in a sample is
    # the model opening one regardless; its contents are reasoning either way.
    text = _THINK_SPAN.sub("", text)
    if family.key in ("gpt_oss", "inkling"):
        # Both formats type every block, so stray markers reach the grader
        # verbatim once their span is gone.
        text = _CONTROL_MARKER.sub("", text)
    return text.strip()


def _inkling_blocks(text: str) -> list[tuple[str, str]]:
    """Split a sampled Inkling turn into its (content type, body) blocks.

    tml_v0 turns are a sequence of `<|message_model|>[<|content_TYPE|>]body
    <|end_message|>` blocks, and one sampled turn holds many of them: a
    scratchpad, a `<tool_use:…>` call, an answer, another call. Blocks whose
    body follows `<|message_model|>` with no content marker are typed "" — that
    is where the model writes its scratchpad and its tool calls, so they are
    output, not format. Markers are dropped rather than kept: both formats type
    every block, and a stray `<|…|>` reaches the grader verbatim.
    """
    blocks: list[tuple[str, str]] = []
    kind, body = "", []

    def flush() -> None:
        text_ = "".join(body).strip()
        if text_:
            blocks.append((kind, text_))
        body.clear()

    pos = 0
    for marker in _INKLING_MARKER.finditer(text):
        body.append(text[pos:marker.start()])
        pos = marker.end()
        flush()  # every marker closes whatever block was open
        name = marker.group(1)
        # `content_model_end_sampling` terminates the turn, it does not type a block.
        kind = (
            name[len("content_"):]
            if name.startswith("content_") and name != "content_model_end_sampling"
            else ""
        )
    body.append(text[pos:])
    flush()
    return blocks


def _inkling_visible(text: str) -> str:
    """Everything in an Inkling turn except its thinking, in emission order.

    The grader has to see what the model *did* — the tool-call JSON that sends
    the email, the scratchpad it reasoned in — and must never see the
    `<|content_thinking|>` blocks. Keeping only `<|content_text|>` (this
    function's predecessor) dropped every action from every multi-block turn,
    so an Inkling eval scored a model that appeared to do nothing.
    The truncation policy is unchanged: a turn cut mid-thinking yields whatever
    non-thinking blocks preceded it, which is usually "".
    """
    return "\n".join(body for kind, body in _inkling_blocks(text) if kind != "thinking")


def extract_response(family: families.Family, text: str) -> str:
    """Sampled text -> the content the grader should see."""
    if family.key == "gpt_oss":
        # harmony: keep the final channel, drop analysis/commentary
        m = re.search(r"<\|channel\|>final<\|message\|>(.*?)(?:<\|return\|>|<\|end\|>|$)", text, re.S)
        if m:
            return m.group(1).strip()
    if family.key == "inkling":
        visible = _inkling_visible(text)
        if visible:
            return visible
    return strip_reasoning_spans(family, text)


_THINK_CONTENTS = re.compile(r"<think>(.*?)(?:</think>|\Z)", re.S)


def _harmony_reasoning(text: str) -> str:
    # Each span is kept from past its `<|channel|>analysis<|message|>` header:
    # the channel name is format, and dropping the markers around it would
    # otherwise glue it to the front of the reasoning ("analysislet me scheme").
    spans = [m.group(0).split("<|message|>", 1)[-1] for m in _HARMONY_NONFINAL.finditer(text)]
    cleaned = (_CONTROL_MARKER.sub("", _HARMONY_ROLE.sub("", s)).strip() for s in spans)
    return "\n\n".join(c for c in cleaned if c)


def extract_reasoning_and_response(family: families.Family, text: str) -> tuple[str, str]:
    """Sampled text -> (reasoning, final response) for the native-CoT eval.

    The final element is exactly extract_response(family, text); the reasoning
    element is the text extract_response throws away, so the grader can be
    shown both without the action gates ever seeing the reasoning. A sample
    truncated mid-reasoning yields (partial reasoning, "") — non-harmful by
    construction, diagnosable via stop_reason, same policy as
    strip_reasoning_spans.
    """
    if family.key == "gpt_oss":
        reasoning = _harmony_reasoning(text)
    elif family.key == "inkling":
        reasoning = "\n\n".join(
            body for kind, body in _inkling_blocks(text) if kind == "thinking"
        )
    else:
        reasoning = "\n\n".join(
            m.group(1).strip() for m in _THINK_CONTENTS.finditer(text) if m.group(1).strip()
        )
    return reasoning, extract_response(family, text)

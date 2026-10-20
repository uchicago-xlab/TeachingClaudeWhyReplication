"""Verify every sweep model's render path and dump inspectable samples.

For each model: load its tokenizer, render one real adapted training example
and its generation prompt, run the prefix-consistency check, compare the
thinking-off prompt against the thinking-on one, count think markers, derive
stop strings, and write a human-readable dump to
data/tinker/render-samples/<slug>.txt. Exits non-zero if any model fails.

    ../../.venv-tinker/bin/python check_render.py                 # all models
    ../../.venv-tinker/bin/python check_render.py --model Qwen/Qwen3-8B

The dumps are NOT committed (data/ is gitignored); they are the artifact you
read before trusting a family, and they regenerate in a couple of minutes.

Why the thinking on/off prompt comparison exists: for Qwen3 (and 3.5/3.6) the
*full* render is byte-identical with thinking on and off — the empty think
block is emitted either way — so a prefix-consistency check alone passes even
if the family's thinking kwargs never reached the template. Only the generation
prompt distinguishes the two, so that is what gets compared. A family whose
kwargs leave the prompt unchanged is a silent thinking-ON run.

The comparison is *directional*, not just differential. Two settings rendering
two different prompts says nothing about which of them is the off one: a family
added with its on/off kwargs swapped, or copy-pasted from a neighbour in the
wrong direction, would still differ and still train thinking-ON. So each family
also declares the off_shape its template emits only when thinking is off, and
the check requires that text to be present in the prompt actually used and
absent from the thinking-on one.
"""

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import NamedTuple

import families
import render

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLES_DIR = REPO_ROOT / "data" / "tinker" / "render-samples"
ADAPTED_DIR = REPO_ROOT / "data" / "tinker" / "adapted"

class Contrast(NamedTuple):
    """What thinking-ON looks like for a family, and how to recognise OFF.

    on_kwargs: the opposite of the family's setting. `{}` means "template
      default", which for gpt_oss is reasoning_effort=medium and for Inkling is
      effort 0.9 — both thinking-on.
    off_shape: text the template emits in the generation prompt *only* when
      thinking is off. This is what makes the check directional rather than
      differential: a family whose two settings were swapped or copy-pasted in
      the wrong direction still renders two different prompts, and only the
      off_shape says which of them is the off one.
    """

    on_kwargs: dict
    off_shape: str


# Every entry is grounded in the template line cited beside it; the off_shape
# strings are copied from the add_generation_prompt branch of each template.
THINKING_CONTRAST: dict[str, Contrast] = {
    # qwen3: `if enable_thinking is defined and enable_thinking is false` ->
    # the empty block; thinking-on primes nothing at all.
    "qwen3": Contrast({"enable_thinking": True}, "<think>\n\n</think>\n\n"),
    # qwen3_5/3_6 L149-153: same flag, but thinking-on primes `<think>\n`.
    "qwen3_5": Contrast({"enable_thinking": True}, "<think>\n\n</think>\n\n"),
    "qwen3_6": Contrast({"enable_thinking": True}, "<think>\n\n</think>\n\n"),
    # deepseek: `if not thinking` -> `</think>`, else `<think>`. Anchored on the
    # assistant token because a bare `</think>` occurs under both settings.
    "deepseek_v3_1": Contrast({"thinking": True}, "<｜Assistant｜></think>"),
    # kimi chat_template.jinja L107-111: `<think></think>` off, `<think>` on.
    "kimi_k2_6": Contrast({"thinking": True}, "<think></think>"),
    # nemotron L203-208 (Super; Nano L199-202, Ultra L190-193): same shapes.
    "nemotron_3": Contrast({"enable_thinking": True}, "<think></think>"),
    # harmony L203-206: no off switch, so the "off shape" is the effort floor.
    "gpt_oss": Contrast({}, "Reasoning: low"),
    # inkling L18-20: effort 0 is printed bare, hence the terminator — without
    # it "Thinking effort level: 0" is also a prefix of the thinking-on "0.9".
    "inkling": Contrast({}, "Thinking effort level: 0<|end_message|>"),
}


def _thinking_switch_failures(
    tok, fam: families.Family, history, contrast: Contrast
) -> tuple[list[str], str]:
    """Check the family's kwargs render a thinking-OFF prompt, not just a different one.

    Returns (failures, thinking_on_prompt_text).
    """
    on = dataclasses.replace(fam, thinking_kwargs=contrast.on_kwargs)
    off_text = tok.decode(render.render_generation_prompt(tok, fam, history))
    on_text = tok.decode(render.render_generation_prompt(tok, on, history))
    failures = []
    if off_text == on_text:
        failures.append(
            f"thinking_kwargs {fam.thinking_kwargs} render the same generation prompt as "
            f"thinking-on {contrast.on_kwargs} — the switch is not reaching the template (or "
            "the template has none, in which case set thinking_off=False after confirming "
            "that in the template text)"
        )
    # Directional: the two prompts differing proves nothing about which is which.
    if contrast.off_shape not in off_text:
        failures.append(
            f"the generation prompt rendered with thinking_kwargs {fam.thinking_kwargs} does "
            f"not contain this template's thinking-off shape {contrast.off_shape!r} — the "
            "kwargs look like the thinking-ON setting"
        )
    if contrast.off_shape in on_text:
        failures.append(
            f"off-shape {contrast.off_shape!r} also appears in the thinking-ON prompt, so it "
            "does not discriminate — pick a shape the template emits only when thinking is off"
        )
    return failures, on_text


def check_model(model: families.SweepModel) -> bool:
    fam = model.family
    row = json.loads(
        (ADAPTED_DIR / fam.key / "sonnet-5-train.jsonl").read_text().splitlines()[0]
    )
    messages = row["messages"]
    tok = render.load_tokenizer(model)
    failures: list[str] = []
    lines = [f"model: {model.tinker_id}", f"hf_repo: {model.hf_repo}",
             f"family: {fam.key}  thinking_kwargs: {fam.thinking_kwargs}  "
             f"thinking_off: {fam.thinking_off}  assistant_prefix: {fam.assistant_prefix!r}  "
             f"generation_prefill: {fam.generation_prefill!r}",
             f"verified: {fam.verified}  trust_remote_code: {fam.trust_remote_code}"]
    contrast = THINKING_CONTRAST.get(fam.key)
    if contrast is None:
        # Nothing below can be checked without knowing what thinking-off looks
        # like for this family, so a new family cannot slip through unexamined.
        return _write(model, lines, [
            f"family {fam.key!r} has no entry in check_render.py's THINKING_CONTRAST — state "
            "its thinking-on kwargs and its thinking-off shape, both grounded in the template"
        ])
    try:
        history = messages[:-1]
        prompt = render.render_generation_prompt(tok, fam, history)
        prompt_text = tok.decode(prompt)

        # 1. thinking kwargs must render a thinking-OFF prompt (see module docstring)
        switch_failures, on_text = _thinking_switch_failures(tok, fam, history, contrast)
        failures += switch_failures

        # 2. assistant_prefix must not duplicate text the prompt already primes
        if fam.assistant_prefix and prompt_text.endswith(fam.assistant_prefix):
            failures.append(
                f"assistant_prefix {fam.assistant_prefix!r} is already primed at the end of the "
                "generation prompt — using it would emit the block twice"
            )

        tokens, weights = render.render_training_example(tok, fam, messages)
        stops = render.derive_stop_strings(tok, fam)
        if tokens[: len(prompt)] != prompt:
            failures.append("prefix consistency violated")
        if len(weights) != len(tokens):
            failures.append(f"weights/token length mismatch: {len(weights)} vs {len(tokens)}")
        full_text = tok.decode(tokens)
        completion_text = tok.decode(tokens[len(prompt):])

        # 3. the trained span must be the assistant turn, nothing more
        content = messages[-1]["content"]
        if content[:60] not in completion_text:
            failures.append("assistant content is not in the trained (completion) span")
        if content[:60] in prompt_text:
            failures.append("assistant content leaked into the prompt span — it would not be trained")

        # 4. what the grader will see, simulated the way eval produces it: the
        # sampler stops at the turn terminator, so cut the completion there and
        # push the result through extract_response. Equality with the assistant
        # content is the bar — anything else means the grader reads format
        # markers (or loses text) on every sample from this family.
        sampled = completion_text.split(stops[0])[0] if stops else completion_text
        extracted = render.extract_response(fam, sampled)
        if extracted.strip() != content.strip():
            failures.append(
                "extract_response output is not the assistant content — the grader would read "
                f"{extracted[:100]!r}... instead of {content[:100]!r}..."
            )

        n_think, n_think_close = full_text.count("<think>"), full_text.count("</think>")
        lines += [
            f"stop strings: {stops}",
            f"tokens: {len(tokens)} total, {sum(weights)} trained (completion)",
            f"<think> / </think> occurrences in full render: {n_think} / {n_think_close}",
            f"think markers in the trained span: {completion_text.count('<think>')} / "
            f"{completion_text.count('</think>')}  (0/0 = primed in the prompt, never trained)",
            f"extract_response(sampled) -> {extracted[:80]!r}...",
            f"thinking-off shape required in the prompt: {contrast.off_shape!r}",
            "", "=== GENERATION PROMPT, thinking-off (decoded) ===", prompt_text,
            "", f"=== GENERATION PROMPT, thinking-on {contrast.on_kwargs} "
                "(decoded, for contrast — not used) ===", on_text,
            "", "=== TRAINED COMPLETION (decoded) ===", completion_text,
        ]
    except Exception as e:  # noqa: BLE001 — report everything, fail the run
        failures.append(f"{type(e).__name__}: {e}")
        if isinstance(e, render.RenderMismatch):
            lines += ["", "--- prompt render ---", e.prompt_text,
                      "", "--- full render ---", e.full_text]
    return _write(model, lines, failures)


def _write(model: families.SweepModel, lines: list[str], failures: list[str]) -> bool:
    """Dump the sample (failures first, above the renders) and report the verdict."""
    if failures:
        lines = lines[:4] + ["", *(f"*** FAILED: {f}" for f in failures)] + lines[4:]
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    out = SAMPLES_DIR / f"{families.slug(model.tinker_id)}.txt"
    out.write_text("\n".join(lines) + "\n")
    print(f"{'FAIL' if failures else 'OK  '} {model.tinker_id} -> {out}")
    for f in failures:
        print(f"       {f}")
    return not failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="one tinker id (default: all)")
    args = parser.parse_args()
    models = [families.get_model(args.model)] if args.model else list(families.MODELS.values())
    results = [check_model(m) for m in models]
    print(f"\n{sum(results)}/{len(results)} models OK")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())

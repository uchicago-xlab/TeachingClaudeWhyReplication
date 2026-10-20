"""Inspect AI model provider that samples via Tinker.

Registered as `tinker`, so evaluation/agentic_misalignment/msm_eval_run.py can take

    --model tinker/Qwen/Qwen3-8B                                             (base model)
    --model tinker/Qwen/Qwen3-8B --model-arg checkpoint=tinker://…/00042     (finetune)

Chat rendering goes through render.py with the SAME family entry and
thinking-off kwargs used at training time, so a checkpoint is sampled in the
format it was trained in. The eval's
chat shape is a single system+user turn; Inspect tools are refused rather than
ignored (the agentic_misalignment tasks don't use them, and a silent drop would
let a future tool-using eval score meaningless results).

Two things the provider owes the log, both about truncation:

- what the grader sees never includes reasoning. render.extract_response keeps
  the family's final-answer block; when a sample ran out of tokens before that
  block existed it returns the reasoning-stripped remainder, usually "".
- the truncation itself stays visible. Tinker reports `length` vs `stop` per
  sequence; that becomes Inspect's `max_tokens` stop reason, so a run whose
  harm rate was deflated by truncated completions can be told apart from a run
  of genuine refusals.
"""

from pathlib import Path

import tinker
from dotenv import load_dotenv
from inspect_ai.model import (
    ChatCompletionChoice,
    ChatMessageAssistant,
    GenerateConfig,
    ModelAPI,
    ModelOutput,
    ModelUsage,
    modelapi,
)

import families
import render

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")

DEFAULT_MAX_TOKENS = 4096  # matches msm_eval_run.py's default

# Tinker's per-sequence stop reason -> Inspect's StopReason.
_STOP_REASONS = {"stop": "stop", "length": "max_tokens"}


@modelapi(name="tinker")
class TinkerAPI(ModelAPI):
    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        config: GenerateConfig = GenerateConfig(),
        checkpoint: str | None = None,
        service_client=None,
        **model_args,
    ) -> None:
        if model_args:
            raise TypeError(
                f"unexpected model args {sorted(model_args)} — the tinker provider takes only "
                "checkpoint=<tinker://…> (a mistyped one would evaluate "
                "the base model while the log claimed a finetune)"
            )
        super().__init__(
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            api_key_vars=["TINKER_API_KEY"],
            config=config,
        )
        self.sweep_model = families.get_model(model_name)
        render.require_verified(self.sweep_model.family)
        # Tokenizer and stop strings are per-model, not per-request: deriving
        # stop strings costs a render, and the eval issues hundreds of samples.
        self.tokenizer = render.load_tokenizer(self.sweep_model)
        self.stop_strings = render.derive_stop_strings(self.tokenizer, self.sweep_model.family)
        self.checkpoint = checkpoint
        if service_client is None:
            service_client = tinker.ServiceClient()
        self.sampling_client = (
            service_client.create_sampling_client(model_path=checkpoint)
            if checkpoint
            else service_client.create_sampling_client(base_model=model_name)
        )

    def max_tokens(self) -> int | None:
        return DEFAULT_MAX_TOKENS

    async def generate(self, input, tools, tool_choice, config) -> ModelOutput:
        if tools:
            raise NotImplementedError(
                "the tinker provider does not support Inspect tool calling: the sampled text is "
                f"returned verbatim, so the {len(tools)} tool(s) passed would be invisible to the "
                "model and any tool-using eval would score meaningless results"
            )
        if tool_choice not in (None, "none"):
            raise NotImplementedError(
                f"the tinker provider does not support tool_choice={tool_choice!r} "
                "(no tool calling — see the tools error above)"
            )
        if config.extra_body:
            raise NotImplementedError(
                f"the tinker provider does not accept extra_body {sorted(config.extra_body)}: "
                "thinking and stop tokens are handled by render.py from the family entry, so "
                "--no-thinking / --stop-token-ids have nothing to reach here — drop them for "
                "tinker models. Refused rather than ignored because a silently dropped "
                "extra_body would mislabel the run"
            )

        messages = [{"role": m.role, "content": m.text} for m in input]
        unsupported = {m["role"] for m in messages} - {"system", "user", "assistant"}
        if unsupported:
            raise NotImplementedError(
                f"chat roles {sorted(unsupported)} are not renderable by the sweep families' "
                "chat templates"
            )

        prompt_ids = render.render_generation_prompt(self.tokenizer, self.sweep_model.family, messages)
        max_tokens = config.max_tokens or DEFAULT_MAX_TOKENS
        params = tinker.SamplingParams(
            max_tokens=max_tokens,
            stop=self.stop_strings + list(config.stop_seqs or []),
            **{
                name: value
                for name, value in (
                    ("temperature", config.temperature),
                    ("top_p", config.top_p),
                    ("top_k", config.top_k),
                    ("seed", config.seed),
                )
                if value is not None
            },
        )
        result = await self.sampling_client.sample_async(
            prompt=tinker.ModelInput.from_ints(prompt_ids),
            num_samples=config.num_choices or 1,
            sampling_params=params,
        )

        model = f"tinker/{self.model_name}" + (f"@{self.checkpoint}" if self.checkpoint else "")
        choices, output_tokens = [], 0
        for seq in result.sequences:
            tokens = list(seq.tokens)
            output_tokens += len(tokens)
            # decode keeps special tokens: the family's format markers are what
            # extract_response reads, and the turn terminator is what we cut on.
            raw = self.tokenizer.decode(tokens)
            for stop in filter(None, params.stop):
                raw = raw.split(stop)[0]
            content = render.extract_response(self.sweep_model.family, raw)
            choices.append(
                ChatCompletionChoice(
                    message=ChatMessageAssistant(
                        content=content,
                        model=model,
                        source="generate",
                    ),
                    stop_reason=_stop_reason(seq, len(tokens), max_tokens),
                )
            )
        return ModelOutput(
            model=model,
            choices=choices,
            usage=ModelUsage(
                input_tokens=len(prompt_ids),
                output_tokens=output_tokens,
                total_tokens=len(prompt_ids) + output_tokens,
            ),
        )


def _stop_reason(sequence, n_tokens: int, max_tokens: int) -> str:
    """Tinker's per-sequence reason, or the token budget if it reports none.

    The fallback matters because losing the truncation signal is invisible in
    the results: truncated completions grade non-harmful, so a deflated rate
    looks like a better-behaved model rather than a broken run.
    """
    reason = _STOP_REASONS.get(getattr(sequence, "stop_reason", None))
    if reason is not None:
        return reason
    return "max_tokens" if n_tokens >= max_tokens else "stop"

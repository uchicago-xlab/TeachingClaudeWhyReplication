"""Shared ChatML setup for the chat stages.

<|im_end|> sits INSIDE the {% generation %} block, so assistant-only loss
labels the end-of-turn terminator.

Token ids are resolved from the tokenizer, not hardcoded: Qwen2.5 and Qwen3
use 151644/151645/151643 for <|im_start|>, <|im_end|> and <|endoftext|>.
The code only needs a base whose vocab holds those three strings;
`terminator_ids` raises on anything else. The module constants stay as the
Qwen2.5 values for callers that never see a tokenizer.
"""

from transformers import AutoTokenizer

CHATML = (
    "{%- for message in messages %}"
    "{%- if message['role'] == 'assistant' %}"
    "{{- '<|im_start|>assistant\n' }}"
    "{% generation %}{{ message['content'] }}<|im_end|>{% endgeneration %}{{ '\n' }}"
    "{%- else %}"
    "{{- '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}"
    "{%- endif %}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{%- endif %}"
)

IM_START = 151644     # <|im_start|> — Qwen2.5 ids; use terminator_ids(tok) when possible
IM_END = 151645       # <|im_end|>  — ChatML end-of-turn
ENDOFTEXT = 151643    # <|endoftext|> — base eos; kept as a stop for safety

TOKENS = ("<|im_start|>", "<|im_end|>", "<|endoftext|>")


def terminator_ids(tok):
    """(im_start, im_end, endoftext) ids for this tokenizer; raises if any is missing.

    A missing token tokenizes to unk or None. Some vocabularies use
    <|endoftext|> as the unk token, so only treat unk as missing when the
    string is not the unk token.
    """
    ids = []
    for name in TOKENS:
        t = tok.convert_tokens_to_ids(name)
        if t is None or (t == tok.unk_token_id and name != tok.unk_token):
            raise ValueError(f"{name} not in the vocab of {tok.name_or_path}: "
                             "the chat stages need a ChatML-token base")
        ids.append(t)
    return tuple(ids)


def chat_tokenizer(model_id):
    tok = AutoTokenizer.from_pretrained(model_id)
    terminator_ids(tok)  # fail early on a base without the ChatML tokens
    tok.chat_template = CHATML
    tok.eos_token = "<|im_end|>"
    if tok.pad_token is None:
        tok.pad_token = "<|endoftext|>"
    return tok

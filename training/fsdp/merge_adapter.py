"""Merge one or more adapters into their base and write a full model.

PEFT merge_and_unload, adapters applied in order. With --chat the
generation_config is also repaired in the written model: eos/stop set to
<|im_end|> and <|endoftext|> (151645/151643 on Qwen2.5, resolved from the
tokenizer), so serving does not depend on every caller passing per-request
stop_token_ids. The base generation_config can silently defeat server-side
overrides; fixing the written model removes that failure class.

    # stories merge — keeps the base-model tokenizer/config:
    python merge_adapter.py --adapter out/stories --out merged/stories-base

    # chat merge — bakes ChatML + stop tokens:
    python merge_adapter.py --base merged/stories-base --adapter out/instruct_mix \\
        --out merged/instruct_mix --chat

Needs CPU RAM for the full bf16 model (~65GB for 32B); run it on the pod.
"""

import argparse
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from chatml import chat_tokenizer, terminator_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-32B"),
                    help="default $BASE_MODEL or Qwen/Qwen2.5-32B")
    ap.add_argument("--adapter", required=True, nargs="+",
                    help="one or more adapter dirs/repos, merged in order (several: a "
                         "stack written as ONE merge, e.g. stories then instruct_mix — "
                         "a 72B cannot afford two 145 GB intermediate merges on disk)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--chat", action="store_true",
                    help="chat model: ChatML tokenizer + eos/stop in generation_config")
    args = ap.parse_args()

    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, device_map="cpu"
    )
    for adapter in args.adapter:
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
        print(f"merged {adapter}", flush=True)

    if args.chat:
        tok = chat_tokenizer(args.adapter[-1])
        _, IM_END, ENDOFTEXT = terminator_ids(tok)
        model.config.eos_token_id = IM_END
        model.generation_config.eos_token_id = [IM_END, ENDOFTEXT]
        model.generation_config.pad_token_id = ENDOFTEXT
    else:
        tok = AutoTokenizer.from_pretrained(args.base)

    # 5GB shards like the base model ships. The transformers default (50GB)
    # can wedge a save on a RunPod network volume (MooseFS): the writer sleeps
    # forever at the end of the first huge shard.
    model.save_pretrained(args.out, safe_serialization=True,
                          max_shard_size="5GB")
    tok.save_pretrained(args.out)
    print(f"merged -> {args.out}"
          + (" (ChatML + stop tokens baked in)" if args.chat else ""))


if __name__ == "__main__":
    main()

"""Stories stage (synthetic document finetuning, SDF) on a Qwen2.5 base — FSDP + FA3.

Continued pretraining: loss on EVERY token of raw {"text": ...} lines, no
chat template, LoRA r64/a128/dropout 0 on all linear layers (this stage never
emits ChatML specials, so the token tables stay untouched), cutoff 4096,
lr 1e-4 cosine + 3% warmup, 2 epochs, effective batch 8 packed sequences.

Launch via launch.sh:

    bash launch.sh stories 8 --corpus /workspace/data/stories.jsonl --out /workspace/out/stories

Optional test loss (story-size ladder): pass --test-corpus <{"text"} JSONL
of stories in no training set>. The trainer then measures loss on it at
step 0 (base model), at the end of training (logged and written to
<out>/test_loss.json), and, if --test-steps N > 0, every N steps in between.
All land in wandb as eval/loss. The training recipe is the same either way.
"""

import argparse
import json
import os

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

LINEAR = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default=os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-32B"),
                    help="default $BASE_MODEL or Qwen/Qwen2.5-32B")
    ap.add_argument("--corpus", required=True, help='{"text": ...} JSONL')
    ap.add_argument("--accum", type=int, required=True,
                    help="set by launch.sh: 8 // n_gpus keeps effective batch 8")
    ap.add_argument("--attn", default="flash_attention_3",
                    choices=["flash_attention_3", "sdpa"])
    ap.add_argument("--liger", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="256 samples, 5 steps, no wandb")
    ap.add_argument("--test-corpus", default=None,
                    help='test-set {"text": ...} JSONL (never part of any '
                         "training set); enables step-0 + end-state test loss")
    ap.add_argument("--test-steps", type=int, default=0,
                    help="also measure test loss every N steps (0 = start "
                         "and end only)")
    ap.add_argument("--eval-adapter", default=None,
                    help="no training: load this published stories adapter (Hub "
                         "id or dir) on the base and measure --test-corpus loss "
                         "once, writing <out>/test_loss.json (end-state point "
                         "for a run trained without --test-corpus)")
    args = ap.parse_args()
    if args.eval_adapter and not args.test_corpus:
        ap.error("--eval-adapter needs --test-corpus")

    do_eval = bool(args.test_corpus)
    # eval_strategy stays "steps" whenever a test set is given so that
    # eval_on_start is honoured; with --test-steps 0 the interval is pushed
    # past any run length, leaving only the step-0 and final passes.
    NEVER = 10**9
    cfg = SFTConfig(
        output_dir=args.out,
        packing=True,
        max_length=4096,               # stories median ~2k; packs tighter than 8192
        num_train_epochs=2,
        max_steps=5 if args.smoke else -1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.accum,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        use_liger_kernel=args.liger,
        logging_steps=5,
        eval_strategy="steps" if do_eval else "no",
        eval_steps=(args.test_steps or NEVER) if do_eval else NEVER,
        eval_on_start=do_eval,
        per_device_eval_batch_size=1,
        save_strategy="no" if args.smoke else "steps",
        save_steps=200,
        # Adapter-only checkpoints (~1 GB each); a 112M-token corpus writes
        # ~34 of them, which does not fit a 250 GB disk next to the merged
        # and grafted 65 GB bases. Three is enough to resume from.
        save_total_limit=3,
        save_only_model=True,
        report_to="none" if args.smoke else "wandb",
        # WANDB_NAME lets a chain script name each run.
        run_name=os.environ.get("WANDB_NAME", "train_stories"),
        dataset_num_proc=1,
    )

    # Stock tokenizer, stock eos (<|endoftext|>): this is pretraining-style
    # text, ChatML never enters the picture at this stage.
    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, attn_implementation=args.attn
    )
    model.config.use_cache = False

    ds = load_dataset("json", data_files=args.corpus, split="train")
    if args.smoke:
        ds = ds.select(range(min(256, len(ds))))
    # Packed like the train set, so eval/loss is the same per-token quantity
    # as train/loss and comparable across corpus sizes.
    eval_ds = (load_dataset("json", data_files=args.test_corpus, split="train")
               if do_eval else None)

    if args.eval_adapter:
        # Post-hoc end-state pass: same packing, same test set, same
        # per-token loss as the in-run passes, on a published adapter.
        model = PeftModel.from_pretrained(model, args.eval_adapter)
        model = model.to(torch.bfloat16)
        trainer = SFTTrainer(model=model, args=cfg,
                             train_dataset=ds.select(range(min(8, len(ds)))),
                             eval_dataset=eval_ds, processing_class=tok)
        m = trainer.evaluate()
        trainer.log_metrics("test_adapter", m)
        if trainer.is_world_process_zero():
            os.makedirs(args.out, exist_ok=True)
            with open(os.path.join(args.out, "test_loss.json"), "w") as f:
                json.dump({"test_corpus": args.test_corpus,
                           "adapter": args.eval_adapter, "end_state": m},
                          f, indent=2)
        return

    peft_cfg = LoraConfig(
        r=64, lora_alpha=128, lora_dropout=0.0,
        target_modules=LINEAR.split(","), task_type="CAUSAL_LM",
    )
    # FSDP flat-params need a uniform dtype per wrap unit, and PEFT creates
    # adapters in fp32: cast the PeftModel to bf16 before FSDP flattens it.
    model = get_peft_model(model, peft_cfg)
    model = model.to(torch.bfloat16)

    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds,
                         eval_dataset=eval_ds, processing_class=tok)
    trainer.train()
    trainer.save_model(args.out)
    if do_eval:
        # End-state test loss, before merge/graft/SFT. Written to disk as
        # well as wandb so it survives a dropped wandb run.
        m = trainer.evaluate()
        trainer.log_metrics("test_end", m)
        trainer.save_metrics("test_end", m, combined=False)
        with open(os.path.join(args.out, "test_loss.json"), "w") as f:
            json.dump({"test_corpus": args.test_corpus, "end_state": m},
                      f, indent=2)


if __name__ == "__main__":
    main()

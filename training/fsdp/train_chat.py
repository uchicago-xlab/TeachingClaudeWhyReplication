"""Chat SFT stage (instruct_mix and difficult_advice) on a Qwen2.5 base — FSDP + FA3.

Recipe: LoRA r64/a128/dropout 0 on all linear layers, assistant-only loss,
lr 1e-4 cosine + 3% warmup, 2 epochs, cutoff 8192, effective batch 8 packed
sequences. The token tables are not trained: the base must already carry the
terminator graft (graft_terminator.py), which is the whole table repair.

Launch via launch.sh, which owns the topology math (accum = 8 // n_gpus):

    bash launch.sh instruct_mix 8 --base /root/grafted --out /workspace/out/instruct_mix
"""

import argparse
import os
from dataclasses import dataclass

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer

from chatml import chat_tokenizer

LINEAR = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"


@dataclass
class WarmupFloorConfig(SFTConfig):
    """SFTConfig with a floor on warmup steps.

    Total steps are only known once the trainer has packed the data, so the
    floor is applied where Trainer asks for it. A module-level subclass, not a
    patched lambda on the instance: the distributed launcher pickles the args
    object, and a local lambda cannot be pickled.
    """
    min_warmup_steps: int = 0

    def get_warmup_steps(self, num_training_steps: int) -> int:
        return max(self.min_warmup_steps, super().get_warmup_steps(num_training_steps))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default=os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-32B"),
                    help="grafted base or merged dir (default: $BASE_MODEL "
                         "or Qwen/Qwen2.5-32B)")
    ap.add_argument("--data", default="/workspace/data/instruct_mix.jsonl")
    ap.add_argument("--accum", type=int, required=True,
                    help="set by launch.sh: 8 // n_gpus keeps effective batch 8")
    ap.add_argument("--attn", default="flash_attention_3",
                    choices=["flash_attention_3", "sdpa"],
                    help="fall back to sdpa if FA3 dies at step 0 (see README)")
    ap.add_argument("--liger", action="store_true",
                    help="Liger kernel (lowers memory; not used for the published runs)")
    ap.add_argument("--smoke", action="store_true",
                    help="256 samples, 5 steps, no wandb")
    ap.add_argument("--epochs", type=int, default=2,
                    help="passes over the dataset (default 2, the instruct-mix "
                         "recipe; difficult-advice runs set their own)")
    ap.add_argument("--val-data", default=None,
                    help="held-out JSONL scored every epoch (eval_loss to the log and "
                         "wandb). On a small dataset it is the signal that shows "
                         "overfitting.")
    ap.add_argument("--save-each-epoch", action="store_true",
                    help="checkpoint after every epoch instead of every 100 "
                         "steps. For runs of a few dozen steps, where step-based "
                         "saving never fires and epoch choice is the open question.")
    ap.add_argument("--warmup-ratio", type=float, default=0.03,
                    help="cosine warmup as a fraction of total steps (instruct-mix "
                         "recipe 0.03; MSM App. B.4 used 0.05)")
    ap.add_argument("--min-warmup-steps", type=int, default=0,
                    help="floor on warmup steps. 1-epoch difficult-advice runs are "
                         "11-65 steps long, where a ratio alone gives 1-2 steps and "
                         "differs between runs; difficult_advice_chain.sh passes 2 "
                         "so every run shares the same schedule shape.")
    args = ap.parse_args()

    cfg = WarmupFloorConfig(
        min_warmup_steps=args.min_warmup_steps,
        output_dir=args.out,
        packing=True,
        packing_strategy="bfd",        # per-example isolation, = neat_packing
        max_length=8192,
        assistant_only_loss=True,
        # Plain nll instead of TRL's default fused chunked_nll: it materializes
        # full logits (a few GB at batch 1, fine on 80GB+ cards). Every
        # published run used this loss path.
        loss_type="nll",
        num_train_epochs=args.epochs,
        max_steps=5 if args.smoke else -1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.accum,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        # transformers 5.x: a float < 1 here is a fraction of total steps; the
        # old warmup_ratio kwarg is deprecated in 5.14 and gone in 5.16.
        warmup_steps=args.warmup_ratio,
        bf16=True,
        gradient_checkpointing=True,
        # non-reentrant checkpointing works under FSDP
        gradient_checkpointing_kwargs={"use_reentrant": False},
        use_liger_kernel=args.liger,
        logging_steps=1,
        save_strategy="no" if args.smoke else ("epoch" if args.save_each_epoch else "steps"),
        eval_strategy="epoch" if args.val_data else "no",
        per_device_eval_batch_size=1,   # nll path materializes full logits
        save_steps=100,
        save_only_model=True,
        report_to="none" if args.smoke else "wandb",
        run_name=os.environ.get("WANDB_NAME") or "train_chat",
        dataset_num_proc=1,  # >1 deadlocks with the Rust tokenizer under fork
    )

    tok = chat_tokenizer(args.base)
    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, attn_implementation=args.attn
    )
    model.config.use_cache = False

    ds = load_dataset("json", data_files=args.data, split="train")
    if args.smoke:
        ds = ds.select(range(256))
    val = load_dataset("json", data_files=args.val_data, split="train") if args.val_data else None

    # Linear layers only: the token tables stay frozen at their grafted values.
    peft_cfg = LoraConfig(
        r=64, lora_alpha=128, lora_dropout=0.0,
        target_modules=LINEAR.split(","),
        task_type="CAUSAL_LM",
    )
    # FSDP flat-params require uniform dtype per wrap unit, and PEFT creates
    # adapters in fp32 beside the bf16 base -> wrap PeftModel ourselves and
    # cast everything bf16 ("pure bf16" LoRA).
    model = get_peft_model(model, peft_cfg)
    model = model.to(torch.bfloat16)

    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, eval_dataset=val,
                         processing_class=tok)
    trainer.train()
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)   # template travels with the adapter


if __name__ == "__main__":
    main()

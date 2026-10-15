"""Graft a working terminator onto <|im_end|> in a copy of a ChatML-token base.

    python graft_terminator.py --base <base_snapshot> --out <dir> \
        [--noise 0.1] [--seed 0]

Qwen2.5 base never trained the ChatML terminator: <|im_end|>'s embed row is
exactly zero, and its lm_head row sits on the one direction shared by all
1,960 untrained emittable tokens — anti-correlated with <|endoftext|>
(cos −0.478), so the cluster actively suppresses the working terminator.
This script copies <|endoftext|>'s rows onto <|im_end|> in BOTH tables,
optionally rotated by a small random amount with the norm preserved. The
chat stages then train linear-only LoRA on the patched base:

    bash launch.sh instruct_mix 8 --base <out> --out ...

The training chains pass --noise 0 (a bit-exact copy; the gate checks cos
1.0000). With --noise > 0 the grafted rows keep cos ≈ 0.995 with their
donors: near-identical function, separable identity. Training is
linear-only, so the rows stay frozen — this init is the entire table repair.

Only the two table shards are rewritten; every other file is symlinked into
--base, so --base must outlive --out. The out dir also gets
`base_row_patch.safetensors` (token_ids, embed_tokens_rows, lm_head_rows):
published adapters ship this small patch instead of a full base, and
evaluation/agentic_misalignment/serve_model.sh applies it with GRAFT=1.
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from transformers import AutoTokenizer

from chatml import terminator_ids

EMB, HEAD = "model.embed_tokens.weight", "lm_head.weight"


def graft(row, noise, gen):
    """Rotate `row` toward a random direction by `noise`, keeping its norm."""
    if noise == 0:
        return row  # --noise 0: bit-exact donor copy, no float round-trip
    n = torch.randn(row.shape, generator=gen)
    n = n / n.norm() * row.norm() * noise
    out = row + n
    return out / out.norm() * row.norm()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--noise", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    # ids from the base's own tokenizer: 151645/151643 on Qwen2.5
    _, IM_END, ENDOFTEXT = terminator_ids(AutoTokenizer.from_pretrained(args.base))

    wm = json.loads(
        (args.base / "model.safetensors.index.json").read_text())["weight_map"]
    shards = {wm[EMB], wm[HEAD]}

    shutil.rmtree(args.out, ignore_errors=True)
    args.out.mkdir(parents=True)
    for f in args.base.iterdir():
        if f.name in shards or f.is_dir():
            continue
        os.symlink(f.resolve(), args.out / f.name)

    tensors = {sh: load_file(str(args.base / sh)) for sh in shards}
    E = tensors[wm[EMB]][EMB].clone()
    H = tensors[wm[HEAD]][HEAD].clone()
    gen = torch.Generator().manual_seed(args.seed)
    H[IM_END] = graft(H[ENDOFTEXT].float(), args.noise, gen).to(H.dtype)
    E[IM_END] = graft(E[ENDOFTEXT].float(), args.noise, gen).to(E.dtype)

    tensors[wm[EMB]][EMB] = E
    tensors[wm[HEAD]][HEAD] = H
    for sh, t in tensors.items():
        save_file(t, str(args.out / sh), metadata={"format": "pt"})

    save_file(
        {
            "token_ids": torch.tensor([IM_END]),
            "embed_tokens_rows": E[IM_END].unsqueeze(0),
            "lm_head_rows": H[IM_END].unsqueeze(0),
        },
        str(args.out / "base_row_patch.safetensors"),
        metadata={"format": "pt"},
    )

    cos = torch.nn.functional.cosine_similarity
    print(f"grafted im_end (noise {args.noise}, seed {args.seed}): "
          f"head norm {H[IM_END].float().norm():.4f}, "
          f"embed norm {E[IM_END].float().norm():.4f}; "
          f"cos to endoftext: head "
          f"{cos(H[IM_END].float(), H[ENDOFTEXT].float(), dim=0):.4f}, "
          f"embed {cos(E[IM_END].float(), E[ENDOFTEXT].float(), dim=0):.4f}")
    print(f"rewrote {sorted(shards)}, symlinked the rest -> {args.out} "
          f"(+ base_row_patch.safetensors for release)")


if __name__ == "__main__":
    main()

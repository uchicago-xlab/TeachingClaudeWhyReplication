"""Are a base model's ChatML terminator rows trained? Answer from the Hub
without downloading the model.

    python check_graft_needed.py Qwen/Qwen2.5-32B Qwen/Qwen2.5-72B ...

For each repo: find the shards that hold embed_tokens and lm_head, read the
safetensors header (one small range request), then range-fetch ONLY the rows
for <|im_start|>, <|im_end|> and <|endoftext|> plus a few ordinary tokens for a
norm reference. Prints each row's norm and its cosine to <|endoftext|>.

Reading the result: on Qwen2.5-32B the <|im_end|> embed row is exactly zero
and its lm_head row has norm ~0.37 vs ~1.26 for trained rows — that is the
defect graft_terminator.py repairs. The decisive test is the last column:
cos to the model's pad token (or its highest, surely-unused id). An untrained
lm_head row shares ONE direction with every other untrained row (cos 1.000),
so <|im_end|>'s logit is tied to the whole junk cluster and the model cannot
select it alone. A base whose <|im_end|> rows have ordinary norms AND a cos
to pad well below 1 needs no graft.

Qwen2.5-14B/32B/72B all show zero embed rows and the lm_head cluster
(cos 1.000 to pad): every Qwen2.5 base needs the graft.
"""

import json
import struct
import sys

import numpy as np
import requests
from transformers import AutoTokenizer

EMB, HEAD = "model.embed_tokens.weight", "lm_head.weight"
REF_TOKENS = [" the", " and", "Hello", " of"]
DT = {"BF16": ("bf16", 2), "F16": ("f16", 2), "F32": ("f32", 4)}


def decode(buf, kind):
    if kind == "bf16":  # widen to f32 by shifting into the high half
        return (np.frombuffer(buf, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
    return np.frombuffer(buf, dtype=np.float16 if kind == "f16" else np.float32).astype(np.float32)


def url(repo, path):
    return f"https://huggingface.co/{repo}/resolve/main/{path}"


def get(u, headers=None):
    r = requests.get(u, headers=headers or {}, allow_redirects=True, timeout=60)
    r.raise_for_status()
    return r.content


def shard_for(repo, tensor):
    try:
        wm = json.loads(get(url(repo, "model.safetensors.index.json")))["weight_map"]
        return wm.get(tensor)
    except requests.HTTPError:
        return "model.safetensors"  # single-file checkpoint


def header(repo, shard):
    n = struct.unpack("<Q", get(url(repo, shard), {"Range": "bytes=0-7"}))[0]
    h = json.loads(get(url(repo, shard), {"Range": f"bytes=8-{8 + n - 1}"}))
    return h, 8 + n


def fetch_rows(repo, tensor, ids):
    shard = shard_for(repo, tensor)
    if shard is None:
        return None
    h, base = header(repo, shard)
    meta = h[tensor]
    dtype, width = DT[meta["dtype"]]
    vocab, hidden = meta["shape"]
    start = base + meta["data_offsets"][0]
    rows = {}
    for t in ids:
        off = start + t * hidden * width
        buf = get(url(repo, shard), {"Range": f"bytes={off}-{off + hidden * width - 1}"})
        rows[t] = decode(buf, dtype)
    return rows


def main():
    for repo in sys.argv[1:]:
        tok = AutoTokenizer.from_pretrained(repo)
        special = {name: tok.convert_tokens_to_ids(name)
                   for name in ["<|im_start|>", "<|im_end|>", "<|endoftext|>"]}
        unk = tok.unk_token_id
        missing = [k for k, v in special.items() if v is None or (v == unk and k != tok.unk_token)]
        print(f"\n== {repo}  ids {special}" + (f"  MISSING {missing}" if missing else ""))
        if missing:
            continue
        refs = [tok.convert_tokens_to_ids(tok.tokenize(w)[0]) for w in REF_TOKENS]
        # an id that was surely never trained: the pad token, else the top id
        junk = tok.pad_token_id if tok.pad_token_id not in (None, special["<|endoftext|>"]) else len(tok) - 1
        for tensor in (EMB, HEAD):
            rows = fetch_rows(repo, tensor, list(special.values()) + refs + [junk])
            if rows is None:
                print(f"  {tensor}: not in checkpoint (tied embeddings?)")
                continue
            eot, jk = rows[special["<|endoftext|>"]], rows[junk]
            norm = np.linalg.norm
            ref_norm = sum(norm(rows[r]) for r in refs) / len(refs)
            print(f"  {tensor}: ordinary-row norm ~{ref_norm:.3f}")
            for name, t in special.items():
                v = rows[t]
                cos = float(v @ eot / (norm(v) * norm(eot))) if norm(v) > 0 else float("nan")
                cj = float(v @ jk / (norm(v) * norm(jk))) if norm(v) > 0 and norm(jk) > 0 else float("nan")
                flag = ("  <-- ZERO ROW" if norm(v) == 0 else "  <-- small" if norm(v) < 0.5 * ref_norm else "")
                flag += "  <-- CLUSTER (untrained)" if cj > 0.999 and t != junk else ""
                print(f"    {name:14s} id {t:7d} norm {norm(v):.4f} cos(endoftext) {cos:+.3f} "
                      f"cos(pad/junk id {junk}) {cj:+.3f}{flag}")


if __name__ == "__main__":
    main()

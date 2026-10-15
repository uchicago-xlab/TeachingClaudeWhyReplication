# Instruction-tuning mix

Every model in the post gets this mix: 13,000 chat samples drawn from ten public
datasets (Table 4). `mix.json` lists the sources and the sample count for each.

Building the mix takes two steps. Run both from the repo root.

```bash
pip install datasets

# 1. Sample the mix (seeded, streaming). Writes the raw file.
python data_generation/instruct_mix/build_mix.py
#    -> data/instruct_mix/instruct_mix-raw.jsonl

# 2. Screen it: drop malformed samples and the identity-confusion samples
#    that MSM also filtered (Appendix B.3). Writes the cleaned file.
python data_generation/instruct_mix/check_dataset.py \
    --in data/instruct_mix/instruct_mix-raw.jsonl \
    --out data/instruct_mix/instruct_mix.jsonl
```

The training scripts read the cleaned file, `data/instruct_mix/instruct_mix.jsonl`.
`build_mix.py --dry-run` prints the per-source plan without loading any data.

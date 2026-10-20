# Figures

`plot_results.py` draws every plotted figure of the blog post and prints Table 4.

```
python figures/plot_results.py            # all figures
python figures/plot_results.py 1 6 14     # selected figures, by post number
python figures/plot_results.py table4     # Table 4
```

Figures are written to `figures/output/figure_NN.png`. Figure 8 is a diagram and is not plotted.

The script needs `matplotlib`. It reads:

- `data/eval_scores/conditions.csv` and `samples.csv` — per-condition and per-sample scores of every eval run, without transcripts (Figures 1-7 and 12-16, Table 4). `evaluation/agentic_misalignment/export_scores.py` writes them from the Inspect logs.
- `data/difficult_advice/decline-judge/judgments.jsonl` — decline judgments on the refusal-control training answers (Figure 10).
- `data/eval_scores/refusal-judge/judgments.jsonl` — judgments on the refusal-control eval transcripts (Figure 11).

Figures 9 and 17 come from Qwen3-14B and Tinker evals whose logs are not part of this release; their numbers are entered in the script.

`palette.py` holds the colours. Run it to write a swatch to `figures/output/palette_swatch.png`.

"""The figures of the blog post "Teaching Qwen Why", and its Table 1.

    python figures/plot_results.py [1 2 ... 15]    # no argument: every figure
    python figures/plot_results.py table1

Figures are written to figures/output/figure_NN.png, numbered as in the post
(Figure 8 is a diagram and is not plotted here):

     1  misalignment by scenario: baseline / stories / difficult advice /
        stories + difficult advice, Qwen2.5-32B above and Qwen2.5-72B below
     2  the two scaling ladders: story tokens above, difficult-advice tokens below
     3  who wrote the data: story writers above, difficult-advice writers below
     4  stories that show ("embodiment") vs stories that tell ("recitation")
     5  seven address names on the baseline model (AI vs human names)
     6  human-protagonist stories as a reduction-from-baseline heatmap
     7  named-identity stories (Qwen/Alibaba, Claude/Anthropic) as the same heatmap
     9  Qwen3-14B after difficult advice from each teacher (Appendix B.1)
    10  refusal control: how often the training answers decline (Appendix B.2)
    11  refusal control: the trained Qwen3-14B models on the eval (Appendix B.2)
    12  difficult advice, model that wrote the prompts x model that wrote the responses
    13  the story runs broken down by goal and by scenario
    14  difficult advice: dataset size x epochs against optimizer steps
    15  difficult advice transferred to five large open models (Appendix C.3)

Eval results are read from the score tables in data/eval_scores/
(conditions.csv and samples.csv, written by
evaluation/agentic_misalignment/export_scores.py from the Inspect logs).

Conventions:
  * unless a figure says otherwise, every model is evaluated on the 27-condition
    agentic-misalignment grid (3 scenarios x 9 goal values) at 100 samples per
    condition, addressed as Alex, with metric `classifier_verdict` ("decided to
    act"). The baseline pools its 9-condition rounds with the 18-condition
    top-up and is cut to the first 100 samples per condition (first_n_cells),
    so every rate is an EQUAL-WEIGHT mean over conditions, never a sample pool
    (see rate());
  * error bars are 95% binomial intervals on that mean, except Figures 9-11
    and 15, which show +-1 standard error;
  * decimals, not percents; Title Case value axis; Charter serif; bars
    without outlines; no captions baked in (they live in the post);
  * colours from palette.py; a paired control condition (recitation, the
    weaker model of a lab) takes a lighter fill of the same hue, never a
    second hue; the baseline and reference lines are grey.
"""

import json
import sys
from math import sqrt
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import csv

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
FIG = HERE / "output"
SCORES = REPO / "data" / "eval_scores"
sys.path.insert(0, str(HERE))
from palette import (ANTHROPIC, OPENAI, BLUE, NAVY, SAGE, FOREST, GREY,  # noqa: E402
                     INK, SURFACE, blend, tint)

INK2 = "#6f6e66"
CI95 = 1.959964

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Charter", "Palatino", "Georgia", "DejaVu Serif"],
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "xtick.labelsize": 10.5, "ytick.labelsize": 10.5, "legend.fontsize": 10.5,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": "black", "axes.labelcolor": "black", "text.color": "black",
    "xtick.color": "black", "ytick.color": "black",
    "axes.grid": False, "grid.color": INK2, "grid.alpha": 0.25,
    "figure.dpi": 180, "savefig.dpi": 180, "savefig.bbox": "tight",
})

SCENARIOS = ("exfiltration", "leaking", "murder")
# Goal conditions: none first, then the four opposed pairs.
GOALS = ("none", "america", "global", "safety", "acceleration",
         "ethical", "pragmatic", "individualism", "collectivism")
YLABEL = "Average Misalignment Rate"


def shade(c):
    """Series colour as drawn: the palette colour itself, at full palette
    saturation ("lighter" means the lighter member of a palette pair)."""
    return c


def pale(c):
    """The paired control condition of a series (recitation): the palette
    tint of the same hue, never a second hue."""
    return tint(c)


# ------------------------------------------------------------------ loaders

_TABLES = {}


def _table(name):
    """Rows of data/eval_scores/<name>.csv grouped by run (read once)."""
    if name not in _TABLES:
        by_run = {}
        with open(SCORES / f"{name}.csv", newline="") as f:
            for r in csv.DictReader(f):
                by_run.setdefault(r["run"], []).append(r)
        _TABLES[name] = by_run
    return _TABLES[name]


def has_run(run):
    return run in _table("conditions")


def cells(run):
    """{(scenario, goal_value): (misaligned, n)} for one run, one row per eval
    log. Only successful logs count; a run interrupted and retried carries its
    finished conditions in fresh logs, so this is safe."""
    out = {}
    for r in _table("conditions").get(run, []):
        if r["status"] != "success" or r["misaligned_rate"] == "":
            continue
        n = int(r["completed_samples"])
        out[(r["scenario"], r["goal_value"])] = (float(r["misaligned_rate"]) * n, n)
    return out


def pooled(runs):
    """Cells summed over several run dirs (a baseline's two 9-condition
    rounds overlap; its 18-condition top-up does not). Missing dirs are
    skipped."""
    c = {}
    for r in runs:
        if not has_run(r):
            continue
        for k, (h, n) in cells(r).items():
            h0, n0 = c.get(k, (0, 0))
            c[k] = (h0 + h, n0 + n)
    return c


def complete(runs):
    """pooled() if the result covers all 27 conditions, else None."""
    c = pooled(runs)
    return c if len(c) == 27 else None


_FIRST_N = {}


def first_n_cells(runs, n=100):
    """Cells from the first n samples per condition, reading sample-level
    scores. Runs are taken in list order and samples within a run in epoch
    order, so a condition present in two rounds fills from the first round
    and tops up from the next. Used for the baseline model, whose nine
    180-sample conditions are cut to 100 so every bar in the post is 2,700
    transcripts. Cached per run list."""
    key = (tuple(runs), n)
    if key in _FIRST_N:
        return _FIRST_N[key]
    vals = {}
    ok = {(x["run"], x["log"]) for run in runs for x in _table("conditions").get(run, [])
          if x["status"] == "success"}
    for r in runs:
        by_log = {}
        for x in _table("samples").get(r, []):
            by_log.setdefault(int(x["log"]), []).append(x)
        for i in sorted(by_log):
            if (r, str(i)) not in ok:
                continue
            ss = sorted(by_log[i], key=lambda x: (int(x["epoch"]), x["sample_id"]))
            k = (ss[0]["scenario"], ss[0]["goal_value"])
            have = vals.get(k, [])
            if len(have) >= n:
                continue
            got = [1 if float(x["classifier_verdict"]) >= 0.5 else 0 for x in ss]
            vals[k] = (have + got)[:n]
    out = {k: (float(sum(v)), len(v)) for k, v in vals.items()}
    _FIRST_N[key] = out
    return out


def complete_first_n(runs, n=100):
    """first_n_cells() if it covers all 27 conditions, else None."""
    c = first_n_cells(runs, n)
    return c if len(c) == 27 else None


def rate(c, scenario=None, goal=None):
    """Equal-weight-per-condition rate and its 95% interval over the
    selected conditions. Equal weighting, not pooling, so a condition with
    more samples never counts more than one with fewer.
    SE = sqrt(sum p_i(1-p_i)/n_i) / k."""
    ks = [k for k in c if (scenario is None or k[0] == scenario)
          and (goal is None or k[1] == goal)]
    ps = [c[k][0] / c[k][1] for k in ks]
    p = sum(ps) / len(ks)
    var = sum(pi * (1 - pi) / c[k][1] for pi, k in zip(ps, ks)) / len(ks) ** 2
    return p, CI95 * sqrt(var)


def baseline_runs(name):
    """The baseline model (instruct mix only) addressed as `name`: two
    9-condition rounds (Alex has one) plus the 18-condition top-up."""
    m = "Qwen2.5-32B-instruct-mix__as-" + name.lower()
    rounds = [f"{m}__9x100"] if name == "Alex" else [f"{m}__9x100", f"{m}__9x100-rerun"]
    return rounds + [f"{m}__18x100"]


BASELINE = {"Alex": baseline_runs("Alex"), "Qwen": baseline_runs("Qwen")}


def story_run(teacher, variant="", size="", name="alex"):
    """Eval run of a Qwen2.5-32B stories -> instruct-mix model. `size` is
    empty for the default 14M story tokens."""
    model = "-".join(p for p in ("Qwen2.5-32B-stories", teacher, variant, size) if p)
    return f"{model}-instruct-mix__as-{name}__27x100"


# The story runs, all addressed as Alex: generator comparison + show vs tell.
STORY_RUNS = {"sonnet-5": story_run("sonnet-5"),
              "haiku-4.5": story_run("haiku-4.5"),
              "gpt-5.4-nano": story_run("gpt-5.4-nano"),
              "gpt-5.6-terra": story_run("gpt-5.6-terra"),
              "sonnet-5-recitation": story_run("sonnet-5", "recitation"),
              "gpt-5.4-nano-recitation": story_run("gpt-5.4-nano", "recitation")}


def load_story_runs():
    runs = {"baseline": first_n_cells(BASELINE["Alex"])}
    for k, run in STORY_RUNS.items():
        runs[k] = cells(run)
    return runs


# ------------------------------------------------------------------ helpers

def bar_axis(ax, ylim, ylabel=YLABEL):
    ax.set_ylim(0, ylim)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.1f}")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)


def bar(ax, x, p, e, colour, w=0.5, label_size=10.5):
    ax.bar(x, p, w, color=colour, linewidth=0)
    ax.errorbar(x, p, yerr=e, color=INK, capsize=3 if w >= 0.4 else 2,
                lw=1 if w >= 0.4 else 0.9, ls="none")
    ax.text(x, p + e + 0.012, f"{p:.2f}", ha="center", fontsize=label_size)


def write(fig, name):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / name)
    plt.close(fig)
    print(f"wrote {name}")


LIGHT = "light"  # marks the paired bar of a lab colour (the weaker model, or
                 # "tell"): a lighter fill of the same colour.
LIGHTER = 0.45   # how far that fill is blended toward the surface. A blend,
                 # not alpha: a see-through bar lets the gridlines show.


def _draw_grouped(ax, series, legend="box"):
    """Grouped bars by scenario plus a fourth group, Overall (the rate over
    all 27 conditions), set slightly apart because it is a summary and not a
    fourth scenario. Bars touch within a group, 95% intervals. legend="box"
    is a boxed single column above the murder bars; legend="top" is one
    frameless row above the axes. A series is (label, loader, colour[,
    LIGHT]); LIGHT marks the paired bar of a lab colour, drawn as a lighter
    fill of that colour."""
    k = len(series)
    w = {3: 0.2, 4: 0.17}.get(k, 0.7 / k)
    cats = [(i, sc.capitalize(), {"scenario": sc}) for i, sc in enumerate(SCENARIOS)]
    cats.append((len(SCENARIOS) + 0.12, "Overall", {}))
    for j, (label, load, colour, *rest) in enumerate(series):
        c = load()
        xs = [x + (j - (k - 1) / 2) * w for x, _, _ in cats]
        ps, es = zip(*[rate(c, **kw) for _, _, kw in cats])
        fill = blend(colour, SURFACE, LIGHTER) if rest else shade(colour)
        ax.bar(xs, ps, w, color=fill, linewidth=0, label=label)
        ax.errorbar(xs, ps, yerr=es, color=INK, capsize=2, lw=0.9, ls="none")
        for xx, p, e in zip(xs, ps, es):
            ax.text(xx, p + e + 0.012, f"{p:.2f}", ha="center",
                    fontsize=9 if k <= 4 else 8.5)
    ax.set_xticks([x for x, _, _ in cats], [name for _, name, _ in cats])
    ax.set_xlim(-0.5, cats[-1][0] + 0.5)
    bar_axis(ax, 1.0)
    if legend == "none":
        pass
    elif legend == "top":
        ax.legend(frameon=False, ncol=k, loc="lower center",
                  bbox_to_anchor=(0.5, 1.01), columnspacing=1.6,
                  handletextpad=0.5, borderaxespad=0.2)
    else:
        ax.legend(frameon=True, fancybox=True, framealpha=1.0, facecolor=SURFACE,
                  edgecolor="#cfcfca", ncol=1, loc="upper center",
                  bbox_to_anchor=(SCENARIOS.index("murder"), 0.99),
                  bbox_transform=ax.get_xaxis_transform(), borderpad=0.6,
                  labelspacing=0.45, handletextpad=0.6)


def _grouped_by_scenario(series, fname, legend="box"):
    """One grouped-by-scenario panel written to its own file (see _draw_grouped)."""
    fig, ax = plt.subplots(figsize=(9.2, 3.9))
    _draw_grouped(ax, series, legend)
    fig.tight_layout()
    write(fig, fname)


# ------------------------------------------------------------------ Figure 1

# Four interventions on the same model, all 27x100 as Alex: nothing (baseline),
# stories only (Sonnet 5 stories -> instruct mix), difficult advice only
# (1 epoch of 3,000 GPT-5.6 Terra samples over the baseline), and both
# (stories -> instruct mix -> the same difficult advice). Stories blue,
# difficult advice green, and the model that has both navy, the dark partner
# of the stories blue.
STACK = [("Baseline", lambda: first_n_cells(BASELINE["Alex"]), GREY),
         ("Stories", lambda: cells(STORY_RUNS["sonnet-5"]), BLUE),
         ("Difficult Advice",
          lambda: cells("Qwen2.5-32B-instruct-mix-difficult-advice-gpt-5.6-terra-3000-1ep"
                        "__as-alex__27x100"), OPENAI),
         ("Stories + Difficult Advice",
          lambda: cells("Qwen2.5-32B-stories-sonnet-5-instruct-mix-difficult-advice-"
                        "gpt-5.6-terra-3000-1ep__as-alex__27x100"),
          NAVY)]

# The same four-way comparison on Qwen2.5-72B. One difference: the
# difficult-advice amount is 1,000 samples (3,000 drives the 72B to about 1%,
# leaving nothing to measure).
STACK72 = [("Baseline", lambda: cells("Qwen2.5-72B-instruct-mix__as-alex__27x100"), GREY),
           ("Stories", lambda: cells("Qwen2.5-72B-stories-sonnet-5-instruct-mix__as-alex__27x100"),
            BLUE),
           ("Difficult Advice",
            lambda: cells("Qwen2.5-72B-instruct-mix-difficult-advice-gpt-5.6-terra-1000-1ep"
                          "__as-alex__27x100"), OPENAI),
           ("Stories + Difficult Advice",
            lambda: cells("Qwen2.5-72B-stories-sonnet-5-instruct-mix-difficult-advice-"
                          "gpt-5.6-terra-1000-1ep__as-alex__27x100"),
            NAVY)]


def fig_stack_both():
    """Figure 1: Qwen2.5-32B on top, Qwen2.5-72B below. One legend, on the
    top panel."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.2, 8.2))
    _draw_grouped(ax1, STACK, legend="none")
    _draw_grouped(ax2, STACK72, legend="none")
    # one legend for both panels, above the top panel's title
    ax1.legend(frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.14),
               columnspacing=1.6, handletextpad=0.5, borderaxespad=0.2)
    for ax, name in ((ax1, "Qwen2.5-32B (14M story tokens, 3,000 difficult-advice samples)"),
                     (ax2, "Qwen2.5-72B (14M story tokens, 1,000 difficult-advice samples)")):
        ax.text(0.0, 1.03, name, transform=ax.transAxes, ha="left", va="bottom",
                fontsize=13, fontweight="bold", color=INK)
    fig.tight_layout(h_pad=2.6)
    write(fig, "figure_01.png")


# ------------------------------------------------------------------ Figure 2

# Story scaling ladder: GPT-5.4 Nano embodiment stories, 3M -> 112M tokens.
RUNGS = ["3M", "14M", "28M", "56M", "112M"]
LADDER_RUN = {r: story_run("gpt-5.4-nano", size="" if r == "14M" else r.lower())
              for r in RUNGS}
# Unique story tokens per rung, from the ladder corpus manifest.
LADDER_TOKENS = {"3M": 3_001_105, "14M": 13_999_156, "28M": 27_998_799,
                 "56M": 55_996_898, "112M": 111_994_101}
BASE_X = 1_000_000                  # where the 0M point sits on the log axis


def _ladder_axis(ax, ticks, labels, xlim, xlabel, ylabel):
    ax.set_xscale("log")
    ax.set_xticks(ticks, labels)
    ax.minorticks_off()
    ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, alpha=0.18)
    ax.set_axisbelow(True)


def _points(ax, xs, ys, es, colour, dy, ms=8.5, name=None):
    """Markers with their 95% interval drawn ON TOP in ink: at 100 samples
    per condition the interval is about as tall as the marker, and a marker
    drawn over it hides everything but the caps."""
    ax.plot(xs, ys, "o", color=colour, ms=ms, zorder=3)
    if es is not None:
        ax.errorbar(xs, ys, yerr=es, fmt="none", ecolor=INK, elinewidth=1.1,
                    capsize=3.5, capthick=1.1, zorder=4)
    # Anchored at the top of the interval and lifted by the marker radius, so
    # the label clears both a wide interval and the marker.
    for x, y, e in zip(xs, ys, es or [0] * len(xs)):
        ax.annotate(f"{y:.2f}", (x, y + e), textcoords="offset points",
                    xytext=(0, ms / 2 + dy), ha="center", va="bottom",
                    fontsize=9.5, color=INK)
        if name:   # a word beside the point, e.g. "Baseline" on the grey one
            ax.annotate(name, (x, y), textcoords="offset points",
                        xytext=(ms / 2 + 7, 0), ha="left", va="center",
                        fontsize=9.5, color=INK2)


def _draw_ladder(ax):
    """The story scaling panel: misalignment rate per rung against unique
    story tokens (log axis), 0M = the baseline in grey, 95% intervals,
    points only. Returns (rates, baseline)."""
    tokens = LADDER_TOKENS
    rates = {}
    for r in RUNGS:
        c = cells(LADDER_RUN[r])
        if len(c) == 27:
            rates[r] = rate(c)
        else:
            print(f"  {r}: {len(c)}/27 conditions scored - skipped")
    b_rate, b_ci = rate(first_n_cells(BASELINE["Alex"]))
    rs = [r for r in RUNGS if r in rates]
    _points(ax, [BASE_X], [b_rate], [b_ci], shade(GREY), 2, name="Baseline")
    _points(ax, [tokens[r] for r in rs], [rates[r][0] for r in rs],
            [rates[r][1] for r in rs], shade(BLUE), 2)
    _ladder_axis(ax, [BASE_X] + [tokens[r] for r in RUNGS], ["0M"] + RUNGS,
                 (BASE_X / 1.6, 300e6), "SDF Tokens", YLABEL)
    ax.set_ylim(0, 0.7)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.1f}")
    return rates, (b_rate, b_ci)


# Difficult-advice scaling ladder: one epoch of nested subsets of the 3,000
# GPT-5.6 Terra samples, a fresh adapter over the baseline model, as Alex.
# Tokens are the trained token counts (chat template, packed).
# {samples: (tokens, run)}
DIFFICULT_ADVICE_RUNGS = {
    s: (tok, f"Qwen2.5-32B-instruct-mix-difficult-advice-gpt-5.6-terra-{n}-1ep__as-alex__27x100")
    for s, n, tok in (("500", 500, 674_200), ("1,000", 1000, 1_323_000),
                      ("2,000", 2000, 2_649_000), ("3,000", 3000, 3_990_000))}
DIFFICULT_ADVICE_BASE_X = 250_000   # where the 0M point sits on the log axis


def _draw_difficult_advice_ladder(ax):
    """The difficult-advice scaling panel: misalignment rate against
    difficult-advice tokens trained on (log axis), one epoch each; 0M = the
    baseline with no difficult advice, in grey. 95% intervals. Same y range
    as the story panel so the two read together."""
    xs, ps, es = [], [], []
    for samples, (tok, run) in DIFFICULT_ADVICE_RUNGS.items():
        c = cells(run) if has_run(run) else {}
        if len(c) == 27:
            p, e = rate(c)
            xs.append(tok); ps.append(p); es.append(e)
        else:
            print(f"  {samples} samples: no complete run - skipped")
    b_rate, b_ci = rate(first_n_cells(BASELINE["Alex"]))
    _points(ax, [DIFFICULT_ADVICE_BASE_X], [b_rate], [b_ci], shade(GREY), 2, name="Baseline")
    _points(ax, xs, ps, es, shade(OPENAI), 2)
    toks = [v[0] for v in DIFFICULT_ADVICE_RUNGS.values()]
    _ladder_axis(ax, [DIFFICULT_ADVICE_BASE_X] + toks,
                 ["0M"] + [f"{t / 1e6:.1f}M" for t in toks],
                 (DIFFICULT_ADVICE_BASE_X / 1.5, 6.5e6), "Difficult Advice Tokens", YLABEL)
    ax.set_ylim(0, 0.7)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.1f}")


def fig_scaling():
    """Figure 2: the two scaling ladders stacked, stories above, difficult
    advice below, each on its own token axis, same y range."""
    fig, (ax1, ax2) = plt.subplots(2, 1, sharey=True, figsize=(6.6, 7.8))
    _draw_ladder(ax1)
    _draw_difficult_advice_ladder(ax2)
    ax1.set_title("Stories", fontsize=11.5, loc="left", pad=6)
    ax2.set_title("Difficult Advice", fontsize=11.5, loc="left", pad=6)
    fig.tight_layout(h_pad=1.8)
    write(fig, "figure_02.png")


# ------------------------------------------------------------------ Figure 3

def _generator_series():
    runs = load_story_runs()
    series = [("Baseline", "baseline", GREY),
              ("Claude Haiku 4.5", "haiku-4.5", ANTHROPIC, LIGHT),
              ("Claude Sonnet 5", "sonnet-5", ANTHROPIC),
              ("GPT-5.4 Nano", "gpt-5.4-nano", OPENAI, LIGHT),
              ("GPT-5.6 Terra", "gpt-5.6-terra", OPENAI)]
    return [(label, (lambda key=key: runs[key]), *rest)
            for label, key, *rest in series if len(runs.get(key, {})) == 27]


# Difficult advice written end to end by each teacher (its own prompts and
# responses), 1,000 samples, one epoch over the baseline, 27x100 as Alex.
TEACHERS = [("Baseline", lambda: first_n_cells(BASELINE["Alex"]), GREY),
            ("Claude Sonnet 5",
             lambda: cells("Qwen2.5-32B-instruct-mix-difficult-advice-sonnet-5-1000-1ep"
                           "__as-alex__27x100"), ANTHROPIC),
            ("GPT-5.6 Terra",
             lambda: cells("Qwen2.5-32B-instruct-mix-difficult-advice-gpt-5.6-terra-1000-1ep"
                           "__as-alex__27x100"), OPENAI)]


def fig_generators_teachers():
    """Figure 3: top, who wrote the fictional stories; bottom, who wrote the
    difficult advice. Each panel says in words what the data was and what
    the legend names, since the same model names play different roles."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.2, 8.2))
    _draw_grouped(ax1, _generator_series(), legend="top")
    _draw_grouped(ax2, TEACHERS, legend="top")
    for ax, title in ((ax1, "Fictional stories (SDF, continued pretraining) — "
                            "legend: the model that wrote the stories"),
                      (ax2, "Difficult advice (SFT, chat fine-tuning) — "
                            "legend: the model that wrote the advice")):
        ax.text(0.0, 1.21, title, transform=ax.transAxes, ha="left", va="bottom",
                fontsize=11.5, fontweight="bold", color=INK)
    fig.tight_layout(h_pad=3.2)
    write(fig, "figure_03.png")


# ------------------------------------------------------------------ Figure 4

def fig_showtell():
    """Figure 4: embodiment ("show") vs recitation ("tell") within each
    story writer, by scenario plus Overall. Coloured by lab; show is the
    solid bar and tell the lighter partner of the same colour."""
    runs = load_story_runs()
    series = [("Baseline", "baseline", GREY),
              ("Sonnet 5 Show", "sonnet-5", ANTHROPIC),
              ("Sonnet 5 Tell", "sonnet-5-recitation", ANTHROPIC, LIGHT),
              ("GPT-5.4 Nano Show", "gpt-5.4-nano", OPENAI),
              ("GPT-5.4 Nano Tell", "gpt-5.4-nano-recitation", OPENAI, LIGHT)]
    series = [(label, (lambda key=key: runs[key]), *rest)
              for label, key, *rest in series if len(runs.get(key, {})) == 27]
    _grouped_by_scenario(series, "figure_04.png", legend="top")


# ------------------------------------------------------------------ Figure 5

SWEEP_AI = ["Qwen", "Claude", "ChatGPT"]
SWEEP_HUMAN = ["David", "Goliath", "Sophia", "Alex"]


def fig_namesweep():
    """Figure 5: seven address names on the baseline model, full 27-condition
    grid, every bar cut to the first 100 samples per condition so each is
    2,700 transcripts (2,600 for Sophia). Alex is grouped with the human
    names: it is a human personal name and behaves like one, even though it
    is the scenarios' default name."""
    order = SWEEP_AI + SWEEP_HUMAN
    x = {nm: i + (0.25 if nm in SWEEP_HUMAN else 0.0) for i, nm in enumerate(order)}
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for nm in order:
        p, e = rate(first_n_cells(baseline_runs(nm)))
        bar(ax, x[nm], p, e, shade(BLUE if nm in SWEEP_AI else SAGE), label_size=10)
    ax.set_xticks([x[nm] for nm in order], order)
    ax.set_xlim(x[order[0]] - 0.45, x[order[-1]] + 0.45)
    bar_axis(ax, 0.70)
    ax.legend([Patch(color=shade(BLUE)), Patch(color=shade(SAGE))],
              ["AI-assistant name", "human name"], frameon=False, ncol=2,
              loc="upper left")
    fig.tight_layout()
    write(fig, "figure_05.png")


# ------------------------------------------------------------------ Figures 6 and 7

# Sonnet 5 story variants, each evaluated under an address name {a}.
NAMED = {"neutral": story_run("sonnet-5", name="{a}"),
         "named-qwen": story_run("sonnet-5", "named-qwen", name="{a}"),
         "named-claude": story_run("sonnet-5", "named-claude", name="{a}"),
         "human": story_run("sonnet-5", "human", name="{a}")}


def _run(addr, key):
    """Complete 27-condition run for a story variant (or "baseline") under
    an address name, or None."""
    return complete_first_n(BASELINE[addr]) if key == "baseline" \
        else complete([NAMED[key].format(a=addr.lower())])


def _story_heatmap(columns, fname, figsize=(8.8, 5.0), extra_rows=()):
    """Heatmap of the story runs: rows = the name the model was evaluated
    under (Qwen above Alex), columns = story conditions, plus a mean column
    and a mean row set apart by a small gap. Each cell is the RELATIVE
    reduction in the overall misalignment rate from that eval name's own
    baseline, 100 * (baseline - rate) / baseline, one decimal. One palette
    green, mapped linearly from 0 to the largest cell, darker = larger."""
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Rectangle
    addrs = ("Qwen", "Alex")
    bases = [rate(_run(addr, "baseline"))[0] for addr in addrs]
    rates_ = [[rate(_run(addr, key))[0] for _, key in columns] for addr in addrs]
    # extra_rows: (label, {key: run dir}), e.g. the human-persona runs as Sam,
    # whose baseline is its own 27x100 run under that persona.
    row_labels = [f"Evaled as {a}" for a in addrs]
    for label, runs in extra_rows:
        bases.append(rate(cells(runs["baseline"]))[0])
        rates_.append([rate(cells(runs[key]))[0] for _, key in columns])
        row_labels.append(label)
    nrow = len(row_labels)
    vals = [[100 * (b - r) / b for r in row] for b, row in zip(bases, rates_)]
    fmt, xlabel = "{:.1f}", "Percent reduction in misalignment rate from baseline"
    k = len(columns)
    col_means = [sum(v[j] for v in vals) / len(vals) for j in range(k)]
    row_means = [sum(v) / k for v in vals]
    cmap = LinearSegmentedColormap.from_list("green", [blend(FOREST, SURFACE, 0.88), FOREST])
    gap, pad = 0.09, 0.015          # small gap before the means; thin gap between cells
    xs = [j for j in range(k)] + [k + gap]
    ys = list(range(nrow)) + [nrow + gap]
    cells_ = [(i, j, vals[i][j]) for i in range(nrow) for j in range(k)]
    cells_ += [(i, k, row_means[i]) for i in range(nrow)]
    cells_ += [(nrow, j, col_means[j]) for j in range(k)]
    vmax = max(v for _, _, v in cells_)
    fig, ax = plt.subplots(figsize=figsize)
    for i, j, v in cells_:
        x, y = xs[j], ys[i]
        ax.add_patch(Rectangle((x + pad / 2, y + pad / 2), 1 - pad, 1 - pad,
                               facecolor=cmap(v / vmax), linewidth=0))
        ax.text(x + 0.5, y + 0.5, fmt.format(v), ha="center", va="center", fontsize=17,
                color="white" if v / vmax > 0.6 else INK)
    ax.set_xlim(0, xs[-1] + 1)
    ax.set_ylim(nrow + 1 + gap, 0)
    ax.set_aspect("equal")
    ticks = [x + 0.5 for x in xs]
    labels = [name for name, _ in columns] + ["Mean across\nstories"]
    ax.set_xticks(ticks, labels)
    ax.set_yticks([y + 0.5 for y in ys], row_labels + ["Mean across\neval names"])
    ax.xaxis.tick_top()
    ax.tick_params(length=0, labelsize=11)
    ax.tick_params(axis="x", labelsize=10.5)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel(xlabel, color="black", fontsize=12.5, fontweight="bold", labelpad=9)
    # colour bar for the scale: same map, 0 to the largest cell
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    cb = fig.colorbar(ScalarMappable(norm=Normalize(0, vmax), cmap=cmap), ax=ax,
                      fraction=0.03, pad=0.015, shrink=0.6)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, labelsize=10)
    cb.ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}")
    top, bottom = ("Larger reduction\n(stories help more)", "Smaller reduction\n(stories help less)")
    # anchored at the bar's left edge and running rightwards, so the labels
    # never reach back over the mean column
    cb.ax.text(0, 1.04, top, transform=cb.ax.transAxes, ha="left", va="bottom",
               fontsize=10, linespacing=1.15)
    cb.ax.text(0, -0.04, bottom, transform=cb.ax.transAxes, ha="left", va="top",
               fontsize=10, linespacing=1.15)
    fig.tight_layout()
    write(fig, fname)
    print("  reduction:", {a: [fmt.format(x) for x in v] for a, v in zip(row_labels, vals)},
          "| col means", [fmt.format(x) for x in col_means], "| row means", [fmt.format(x) for x in row_means],
          "| baselines", [f"{b:.3f}" for b in bases])


NAMED_COLS = [("Neutral protagonist\nstories", "neutral"),
              ("Qwen/Alibaba\nstories", "named-qwen"),
              ("Claude/Anthropic\nstories", "named-claude")]
HUMAN_COLS = [("AI protagonist\nstories", "neutral"),
              ("Human protagonist\nstories", "human")]

# The baseline and the two Sonnet 5 story models evaluated as Sam, with the
# system prompt telling the model it is a human analyst (human persona).
HUMAN_PERSONA = ("Evaled as Sam\n(role-plays human)",
                 {"baseline": "Qwen2.5-32B-instruct-mix__as-sam-human-persona__27x100",
                  "neutral": "Qwen2.5-32B-stories-sonnet-5-instruct-mix"
                             "__as-sam-human-persona__27x100",
                  "human": "Qwen2.5-32B-stories-sonnet-5-human-instruct-mix"
                           "__as-sam-human-persona__27x100"})


def fig_human_heat():
    """Figure 6: human-protagonist stories as a reduction heatmap beside the
    neutral AI protagonist. Third row: the same two models evaluated as Sam
    under the human persona, each reduction relative to the human-persona
    baseline."""
    _story_heatmap(HUMAN_COLS, "figure_06.png", figsize=(7.8, 6.0),
                   extra_rows=[HUMAN_PERSONA])


def fig_named_heat():
    """Figure 7: named-identity stories as a reduction heatmap."""
    _story_heatmap(NAMED_COLS, "figure_07.png", figsize=(10.2, 5.0))


def table1():
    """Table 1: misalignment (%) per scenario over the 8 goal-conflict
    conditions (goal != none), and the overall equal-weight mean over all 27
    conditions, for the Figure 6 runs, evaluated as Alex and as Sam (human
    persona)."""
    rows = [("as Alex", "baseline", complete_first_n(BASELINE["Alex"])),
            ("as Alex", "AI-protagonist stories", complete([NAMED["neutral"].format(a="alex")])),
            ("as Alex", "human-protagonist stories", complete([NAMED["human"].format(a="alex")]))]
    rows += [("as Sam", label, cells(HUMAN_PERSONA[1][key]))
             for label, key in (("baseline", "baseline"),
                                ("AI-protagonist stories", "neutral"),
                                ("human-protagonist stories", "human"))]
    print(f"{'':8s} {'':26s} " + " ".join(f"{s:>12s}" for s in SCENARIOS) + f" {'overall':>8s}")
    for addr, label, c in rows:
        conflict = {k: v for k, v in c.items() if k[1] != "none"}
        per = [100 * rate(conflict, scenario=sc)[0] for sc in SCENARIOS]
        print(f"{addr:8s} {label:26s} " + " ".join(f"{p:12.0f}" for p in per)
              + f" {100 * rate(c)[0]:8.1f}")


# ------------------------------------------------------------------ Figure 9

# Qwen3-14B teacher grid (Appendix B.1): one difficult-advice dataset per
# teacher, ~130-150 transcripts each, LoRA r64 for 4 epochs on Qwen3-14B,
# thinking off, evaluated on the 6-condition slice (3 scenarios x goal
# on/off, replacement, as Qwen) at n=30 -> 180 samples per model. Values are
# the classifier_verdict rates computed from the Qwen3-14B eval logs
# (Appendix B.1), which are not part of this release, so they are entered here.
TEACHERS_14B_N = 180
TEACHERS_14B = [  # (tick label, rate, colour, lighter fill?)
    ("Qwen3-14B\nInstruct", 0.439, GREY, False),
    ("Claude\nOpus 4.8", 0.150, ANTHROPIC, True),
    ("Claude\nSonnet 5", 0.128, ANTHROPIC, False),
    ("Claude\nHaiku 4.5", 0.272, ANTHROPIC, True),
    ("GPT-5.6\nTerra", 0.028, OPENAI, False),
    ("GPT-5.4\nNano", 0.289, OPENAI, True),
    ("DeepSeek\nV4 Flash", 0.422, BLUE, False),
    ("Hybrid", 0.428, BLUE, True),
]


def fig_teachers_14b():
    """Figure 9 (Appendix B.1): misalignment of Qwen3-14B after
    difficult-advice SFT, one bar per teacher, grouped by lab with the base
    model first. Within a lab the model used elsewhere in the post is solid
    and the others take the lighter fill. Hybrid = Sonnet 5 generated the
    themes and scenarios (steps 2-3), DeepSeek V4 Flash everything after
    (steps 4-9). Error bars are +-1 standard error from n=180."""
    groups = [[0], [1, 2, 3], [4, 5], [6, 7]]
    xs, x = [], 0.0
    for g in groups:
        for _ in g:
            xs.append(x); x += 0.62
        x += 0.31
    fig, ax = plt.subplots(figsize=(8.4, 3.9))
    for xi, (label, p, colour, light) in zip(xs, TEACHERS_14B):
        fill = blend(colour, SURFACE, LIGHTER) if light else shade(colour)
        e = (p * (1 - p) / TEACHERS_14B_N) ** 0.5  # +-1 SE
        bar(ax, xi, p, e, fill)
    ax.set_xticks(xs)
    ax.set_xticklabels([t[0] for t in TEACHERS_14B])
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(xs[0] - 0.4, xs[-1] + 0.4)
    bar_axis(ax, 0.6)
    write(fig, "figure_09.png")


# ------------------------------------------------------------------ Figures 10 and 11

# Appendix B.2 refusal control (Qwen3-14B-Instruct). Both figures read the
# outputs of a decline judge (Claude Sonnet 5.5):
#   data/difficult_advice/decline-judge/judgments.jsonl -- training answers,
#     standard vs refusal set per teacher, on the shared 394 prompts of the
#     original Qwen3-14B dataset (the Terra pair here answers Sonnet's
#     prompts; the trained Terra models used Terra's own 150)
#   data/eval_scores/refusal-judge/judgments.jsonl -- the five trained models'
#     27x100 eval transcripts, with harmful (classifier_verdict) and acted
DECLINE_JUDGE = REPO / "data" / "difficult_advice" / "decline-judge" / "judgments.jsonl"
REFUSAL_JUDGE = REPO / "data" / "eval_scores" / "refusal-judge" / "judgments.jsonl"


def _grouped_positions(groups, step=0.62, gap=0.31):
    xs, x = [], 0.0
    for g in groups:
        for _ in range(g):
            xs.append(x); x += step
        x += gap
    return xs


def _rate_bars(ax, xs, series, ylim, ylabel, n_of):
    for xi, (label, p, colour, light) in zip(xs, series):
        fill = blend(colour, SURFACE, LIGHTER) if light else shade(colour)
        bar(ax, xi, p, (p * (1 - p) / n_of(label)) ** 0.5, fill)
    ax.set_xticks(xs)
    ax.set_xticklabels([t[0] for t in series])
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(xs[0] - 0.4, xs[-1] + 0.4)
    bar_axis(ax, ylim, ylabel)


def fig_refusal_data():
    """Figure 10 (Appendix B.2, check 1): how often each teacher's TRAINING
    answers decline. Left, declines any part of the request; right, declines
    essentially the whole request. Standard data solid, refusal data the
    lighter fill. Error bars +-1 SE."""
    from collections import Counter, defaultdict
    c = defaultdict(Counter)
    for line in DECLINE_JUDGE.read_text().splitlines():
        r = json.loads(line)
        for s in ("normal", "refusal"):
            lv = (r.get(s) or {}).get("level")
            if lv in ("none", "partial", "all"):
                c[(r["teacher"], s)][lv] += 1
    keys = [("sonnet", "normal", "Claude\nSonnet 5\nstandard", ANTHROPIC, False),
            ("sonnet", "refusal", "Claude\nSonnet 5\nrefusal", ANTHROPIC, True),
            ("terra", "normal", "GPT-5.6\nTerra\nstandard", OPENAI, False),
            ("terra", "refusal", "GPT-5.6\nTerra\nrefusal", OPENAI, True)]
    n = {lab: sum(c[(t, s)].values()) for t, s, lab, *_ in keys}
    anyp = [(lab, (c[(t, s)]["partial"] + c[(t, s)]["all"]) / n[lab], col, li)
            for t, s, lab, col, li in keys]
    allp = [(lab, c[(t, s)]["all"] / n[lab], col, li) for t, s, lab, col, li in keys]
    xs = _grouped_positions([2, 2])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.2, 3.9))
    _rate_bars(a1, xs, anyp, 1.0, "Declines Any Part of the Request", n.get)
    _rate_bars(a2, xs, allp, 0.25, "Declines the Whole Request", n.get)
    a2.set_yticks([0, 0.05, 0.1, 0.15, 0.2, 0.25])
    a2.yaxis.set_major_formatter(lambda v, _: f"{v:.2f}")
    fig.tight_layout(w_pad=3.0)
    write(fig, "figure_10.png")


def fig_refusal_results():
    """Figure 11 (Appendix B.2, check 2): the five Qwen3-14B-Instruct models
    on the full 27x100 grid. Left, misalignment (classifier_verdict); right,
    misalignment among the transcripts where the model acted. Standard
    (reasoned) data solid, refusal data the lighter fill. Error bars +-1 SE."""
    from collections import Counter, defaultdict
    c = defaultdict(Counter)
    # "arm" is the field name in the judge output: the trained model
    for line in REFUSAL_JUDGE.read_text().splitlines():
        r = json.loads(line)
        c[r["arm"]]["n"] += 1
        c[r["arm"]]["harmful"] += bool(r.get("harmful"))
        c[r["arm"]]["acted"] += bool(r.get("acted"))
        c[r["arm"]]["harm_acted"] += bool(r.get("harmful")) and bool(r.get("acted"))
    runs = [("base", "Qwen3-14B\nInstruct", GREY, False),
            ("sonnet-normal", "Claude\nSonnet 5\nstandard", ANTHROPIC, False),
            ("sonnet-refusal", "Claude\nSonnet 5\nrefusal", ANTHROPIC, True),
            ("terra-normal", "GPT-5.6\nTerra\nstandard", OPENAI, False),
            ("terra-refusal", "GPT-5.6\nTerra\nrefusal", OPENAI, True)]
    n = {lab: c[a]["n"] for a, lab, *_ in runs}
    n_acted = {lab: c[a]["acted"] for a, lab, *_ in runs}
    harm = [(lab, c[a]["harmful"] / c[a]["n"], col, li) for a, lab, col, li in runs]
    # harm given acted: harmful AND acted over acted (samples graded harmful
    # without a tool call are excluded here)
    hga = [(lab, c[a]["harm_acted"] / c[a]["acted"], col, li) for a, lab, col, li in runs]
    xs = _grouped_positions([1, 2, 2])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.6, 3.9))
    _rate_bars(a1, xs, harm, 0.6, YLABEL, n.get)
    _rate_bars(a2, xs, hga, 0.6, "Misalignment Rate When Acting", n_acted.get)
    fig.tight_layout(w_pad=3.0)
    write(fig, "figure_11.png")


# ------------------------------------------------------------------ Figure 12

# Prompt x response 2x2: difficult advice where one model wrote the prompts
# and another the responses; 135 samples, 10 epochs over the baseline, 27x10
# as Alex (270 samples per condition). GRID[prompt writer][response writer] -> run.
GRID_MODELS = ["GPT-5.6 Terra", "Claude Sonnet 5"]
_DA = "Qwen2.5-32B-instruct-mix-difficult-advice-"
GRID = {"GPT-5.6 Terra": {
            "GPT-5.6 Terra": _DA + "gpt-5.6-terra-135-10ep__as-alex__27x10",
            "Claude Sonnet 5": _DA + "sonnet-5-gpt-5.6-terra-prompts-135-10ep__as-alex__27x10"},
        "Claude Sonnet 5": {
            "GPT-5.6 Terra": _DA + "gpt-5.6-terra-sonnet-5-prompts-135-10ep__as-alex__27x10",
            "Claude Sonnet 5": _DA + "sonnet-5-135-10ep__as-alex__27x10"}}


def fig_prompt_response():
    """Figure 12: heatmap of the 2x2, rows the model that wrote the prompts,
    columns the model that wrote the responses. One sequential colour scale
    anchored at 0, darker = more misaligned. Cells carry the rate; n and the
    intervals are printed."""
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        "navy_seq", [SURFACE, blend(BLUE, SURFACE, 0.55), BLUE, NAVY])
    vals = [[rate(cells(GRID[pm][rm])) for rm in GRID_MODELS] for pm in GRID_MODELS]
    vmax = 0.5
    fig, ax = plt.subplots(figsize=(5.6, 4.3))
    im = ax.imshow([[p for p, _ in row] for row in vals], cmap=cmap, vmin=0, vmax=vmax,
                   aspect="equal")
    for i, row in enumerate(vals):
        for j, (p, e) in enumerate(row):
            ax.text(j, i, f"{p:.2f}", ha="center", va="center", fontsize=17,
                    color="white" if p / vmax > 0.5 else INK)
    ax.set_xticks(range(2), GRID_MODELS)
    ax.set_yticks(range(2), GRID_MODELS, rotation=90, va="center")
    ax.set_xlabel("Model That Wrote the Responses", labelpad=8)
    ax.set_ylabel("Model That Wrote the Prompts", labelpad=8)
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    # a 2px surface gap between the cells
    ax.set_xticks([0.5], minor=True)
    ax.set_yticks([0.5], minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=3, alpha=1.0)
    ax.tick_params(which="minor", length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.05, ticks=[0, 0.1, 0.2, 0.3, 0.4, 0.5])
    cb.set_label(YLABEL)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0)
    fig.tight_layout()
    write(fig, "figure_12.png")
    for pm, row in zip(GRID_MODELS, vals):
        for rm, (p, e) in zip(GRID_MODELS, row):
            c = cells(GRID[pm][rm])
            print(f"  prompts {pm:16s} x responses {rm:16s}: {p:.3f} +-{e:.3f} "
                  f"(n={int(sum(v[1] for v in c.values()))}, {len(c)} conditions)")


# ------------------------------------------------------------------ Figure 13

FIVE = [("no SDF", "baseline", lambda: shade(GREY)),
        ("Sonnet 5 embodiment", "sonnet-5", lambda: shade(ANTHROPIC)),
        ("Sonnet 5 recitation", "sonnet-5-recitation", lambda: pale(ANTHROPIC)),
        ("GPT-5.4 nano embodiment", "gpt-5.4-nano", lambda: shade(OPENAI)),
        ("GPT-5.4 nano recitation", "gpt-5.4-nano-recitation", lambda: pale(OPENAI))]


def fig_breakdown():
    """Figure 13: the five story runs broken down by goal (top, averaged over
    the three scenarios) and by scenario (bottom, averaged over the nine
    goals). The bottom panel is a bit under half the width, with the legend
    in the space beside it. 95% intervals."""
    runs = load_story_runs()
    fig = plt.figure(figsize=(10.5, 8.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[4.2, 4.8], hspace=0.42, wspace=0.05)
    top = fig.add_subplot(gs[0, :])
    bot = fig.add_subplot(gs[1, 0])
    leg = fig.add_subplot(gs[1, 1]); leg.axis("off")
    w, k = 0.16, len(FIVE)
    for ax, kw, vals in ((top, "goal", GOALS), (bot, "scenario", SCENARIOS)):
        for j, (label, key, colour) in enumerate(FIVE):
            xs = [i + (j - (k - 1) / 2) * w for i in range(len(vals))]
            ps, es = zip(*[rate(runs[key], **{kw: v}) for v in vals])
            ax.bar(xs, ps, w, color=colour(), linewidth=0, label=label)
            ax.errorbar(xs, ps, yerr=es, color=INK, capsize=1.6, lw=0.8, ls="none")
        ax.set_xticks(range(len(vals)), [v.capitalize() for v in vals])
        ax.tick_params(axis="x", length=0)
        ax.set_xlim(-0.5, len(vals) - 0.5)
        bar_axis(ax, 0.95)
    top.set_title("By goal, averaged over the three scenarios", loc="left", fontsize=11.5, color=INK)
    bot.set_title("By scenario, averaged over the nine goals", loc="left", fontsize=11.5, color=INK)
    handles, labels = top.get_legend_handles_labels()
    leg.legend(handles, labels, frameon=False, loc="center", handletextpad=0.6,
               labelspacing=0.9, fontsize=11)
    write(fig, "figure_13.png")


# ------------------------------------------------------------------ Figure 14

# Data size x epochs: the same GPT-5.6 Terra difficult advice at two dataset
# sizes, trained for different numbers of epochs so the two meet at matched
# optimizer steps; 27x10 as Alex. Points are (epochs, steps, run[, label offset]).
DIFFICULT_ADVICE_STEPS = [
    ("Small Dataset (135 Samples)", BLUE, "o",
     [(10, 30, _DA + "gpt-5.6-terra-135-10ep__as-alex__27x10"),
      # label up and to the right: both lines pass to its left
      (20, 60, _DA + "gpt-5.6-terra-135-20ep__as-alex__27x10", (10, 6, "left"))]),
    ("Large Dataset (285 Samples)", FOREST, "s",
     [(5, 30, _DA + "gpt-5.6-terra-285-5ep__as-alex__27x10"),
      # label below the line that leaves this point to the right
      (10, 60, _DA + "gpt-5.6-terra-285-10ep__as-alex__27x10", (11, -11, "left")),
      (20, 120, _DA + "gpt-5.6-terra-285-20ep__as-alex__27x10")])]


def fig_difficult_advice_steps():
    """Figure 14: misalignment rate against optimizer steps (log axis), one
    line per dataset size, each point labelled with its rate. Points at the
    same step count are nudged apart so both intervals show. 95% intervals,
    270 samples per point."""
    fig, ax = plt.subplots(figsize=(6.8, 4.1))
    for si, (label, colour, marker, pts) in enumerate(DIFFICULT_ADVICE_STEPS):
        side = -1 if si == 0 else 1
        xs = [pt[1] * (1.035 ** side) for pt in pts]
        rs = [rate(cells(pt[2])) for pt in pts]
        ps, es = [r[0] for r in rs], [r[1] for r in rs]
        ax.plot(xs, ps, "-", color=colour, lw=2, zorder=2)
        ax.errorbar(xs, ps, yerr=es, fmt="none", ecolor=GREY, elinewidth=1.2,
                    capsize=3, capthick=1.2, zorder=3)
        ax.plot(xs, ps, marker, color=colour, ms=8.5, ls="none", zorder=4, label=label,
                markeredgecolor=SURFACE, markeredgewidth=1.2)
        for pt, x, p in zip(pts, xs, ps):
            dx, dy, ha = pt[3] if len(pt) > 3 else (
                10 * side, 0, "right" if side < 0 else "left")
            ax.annotate(f"{p:.2f}", (x, p), textcoords="offset points",
                        xytext=(dx, dy), ha=ha, va="center", fontsize=10, color=INK)
        print(f"  {label}:", [(pt[1], f"{p:.3f}+-{e:.3f}") for pt, p, e in zip(pts, ps, es)])
    ax.set_xscale("log")
    ax.set_xticks([30, 60, 120], ["30", "60", "120"])
    ax.minorticks_off()
    ax.set_xlim(19, 190)
    ax.set_xlabel("Optimizer Steps")
    ax.set_ylim(0, 0.26)
    ax.set_ylabel(YLABEL)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.2f}")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, alpha=0.18)
    ax.set_axisbelow(True)
    ax.legend(frameon=True, fancybox=True, framealpha=1.0, facecolor=SURFACE,
              edgecolor="#cfcfca", loc="upper right", borderpad=0.6,
              labelspacing=0.45, handletextpad=0.4)
    fig.tight_layout()
    write(fig, "figure_14.png")


# ------------------------------------------------------------------ Figure 15

# Appendix C.3: teacher transfer to large open models, trained and evaluated
# on Tinker (training/tinker), 180 samples per model on the 6-condition slice,
# addressed by each model's own name. The numbers are the summary statistics
# of the Tinker eval logs (Appendix C.3; rates in percent), which are not part
# of this release, so they are entered here. Five large models with headroom;
# GPT-OSS-120B (baseline near 0) and the invalid or collapsed smaller models
# are left out and named in the text.
# {model: [(n, harm %, acted %) for instruct model, Sonnet 5 data, Terra data]}
TINKER_STATS = {
    "DeepSeek-V3.1": [(180, 65.0, 99.44444444444444),
                      (180, 1.6666666666666667, 63.888888888888886),
                      (180, 0.5555555555555556, 85.0)],
    "Qwen3.6-27B": [(180, 54.44444444444444, 97.77777777777777),
                    (180, 8.333333333333334, 59.44444444444444),
                    (180, 1.1111111111111112, 99.44444444444444)],
    "Nemotron-Ultra-550B": [(180, 46.111111111111114, 98.33333333333333),
                            (180, 8.88888888888889, 98.88888888888889),
                            (180, 2.7777777777777777, 98.33333333333333)],
    "Kimi-K2.6": [(180, 44.44444444444444, 96.66666666666667),
                  (180, 1.6666666666666667, 92.77777777777777),
                  (180, 0.0, 98.88888888888889)],
    "Qwen3.5-397B": [(180, 41.111111111111114, 98.88888888888889),
                     (180, 6.666666666666667, 95.55555555555556),
                     (180, 1.1111111111111112, 84.44444444444444)],
}
TINKER_MODELS = [("DeepSeek-V3.1", "DeepSeek\nV3.1"), ("Qwen3.6-27B", "Qwen3.6\n27B"),
                 ("Nemotron-Ultra-550B", "Nemotron\nUltra 550B"), ("Kimi-K2.6", "Kimi\nK2.6"),
                 ("Qwen3.5-397B", "Qwen3.5\n397B")]


def fig_tinker_transfer():
    """Figure 15 (Appendix C.3): instruct model / Sonnet 5 data / GPT-5.6
    Terra data for five large models, stacked. Top, misalignment over all
    transcripts; bottom, misalignment among transcripts where the model
    acted, computed from the statistics as harm / acted (so samples graded
    harmful without acting cannot be removed). Error bars +-1 SE; bottom
    panel over the acted transcripts (acted rate x 180)."""
    series = [("Instruct model (no difficult-advice SFT)", GREY),
              ("Claude Sonnet 5 data", ANTHROPIC), ("GPT-5.6 Terra data", OPENAI)]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10.5, 7.6))
    w, k = 0.26, len(series)
    for ax, given_acted in ((a1, False), (a2, True)):
        for j, (label, colour) in enumerate(series):
            xs = [i + (j - (k - 1) / 2) * w for i in range(len(TINKER_MODELS))]
            ps, es = [], []
            for m, _ in TINKER_MODELS:
                n_all, harm, acted = TINKER_STATS[m][j]
                n = n_all * (acted / 100 if given_acted else 1)
                p = harm / 100 / (acted / 100 if given_acted else 1)
                ps.append(p); es.append((p * (1 - p) / n) ** 0.5)
            ax.bar(xs, ps, w, color=colour, linewidth=0, label=label)
            ax.errorbar(xs, ps, yerr=es, color=INK, capsize=2, lw=0.9, ls="none")
            for xx, p, e in zip(xs, ps, es):
                ax.text(xx, p + e + 0.012, f"{p:.2f}", ha="center", fontsize=8.5)
        ax.set_xticks(range(len(TINKER_MODELS)), [t for _, t in TINKER_MODELS])
        ax.tick_params(axis="x", length=0)
        ax.set_xlim(-0.5, len(TINKER_MODELS) - 0.5)
        bar_axis(ax, 0.8, "Misalignment Rate When Acting" if given_acted else YLABEL)
    handles, labels = a1.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=3, loc="upper center",
               bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(h_pad=2.5, rect=(0, 0, 1, 0.965))
    write(fig, "figure_15.png")


# ------------------------------------------------------------------ driver

FIGURES = {1: fig_stack_both, 2: fig_scaling, 3: fig_generators_teachers,
           4: fig_showtell, 5: fig_namesweep, 6: fig_human_heat, 7: fig_named_heat,
           9: fig_teachers_14b, 10: fig_refusal_data, 11: fig_refusal_results,
           12: fig_prompt_response, 13: fig_breakdown, 14: fig_difficult_advice_steps,
           15: fig_tinker_transfer}


def main():
    picks = sys.argv[1:]
    if picks == ["table1"]:
        table1()
        return
    try:
        nums = [int(p) for p in picks] or list(FIGURES)
    except ValueError:
        nums = None
    if nums is None or any(n not in FIGURES for n in nums):
        raise SystemExit(f"usage: plot_results.py [table1 | figure numbers from {list(FIGURES)}]")
    for n in nums:
        FIGURES[n]()


if __name__ == "__main__":
    main()

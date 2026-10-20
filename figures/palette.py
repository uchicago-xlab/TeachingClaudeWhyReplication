"""Colour palette for every figure in the post.

Two anchors are the labs' brand colours, softened so they sit on a page
without shouting; every other categorical colour is chosen to be a distinct
hue from those two at the same muted saturation and mid lightness, so a
figure that mixes lab colours with generic series still reads as one family.
Baselines and reference lines are grey. Paired conditions on the same
series (embodiment vs recitation, or any "full vs control") use the colour
and its TINT rather than a second hue.

    from palette import ANTHROPIC, OPENAI, BLUE, NAVY, SAGE, FOREST, GREY, tint

Run this file to write figures/output/palette_swatch.png.
"""

from pathlib import Path


def blend(hex_colour, toward="#ffffff", t=0.3):
    """Mix a colour a fraction t toward another (white by default)."""
    a = [int(hex_colour[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(toward[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{int(x + (y - x) * t):02x}" for x, y in zip(a, b))


def tint(hex_colour):
    """The lighter partner of a colour, for a paired control condition."""
    return blend(hex_colour, "#ffffff", 0.45)


# Lab anchors: Anthropic terracotta and OpenAI green, each pulled 15% toward
# white so they match the saturation of the generic colours below.
ANTHROPIC = blend("#D97757", t=0.15)   # -> #de8a6f
OPENAI = blend("#10A37F", t=0.15)      # -> #33b092

# Generic categorical colours: cool tones only, two blues and two greens,
# all muted to match the anchors. The greens sit on the yellow side of green (~130°) so they stay apart from the OpenAI
# teal-green (~165°). Each pair (blue/navy, sage/forest) is one hue at two
# lightness levels, wide enough apart to read as two series on their own.
BLUE = "#6A97C4"    # light slate blue — first generic series
NAVY = "#2F4C75"    # deep blue        — second; dark partner to BLUE
SAGE = "#8DB597"    # light grey-green — third
FOREST = "#3F6B4C"  # deep green       — fourth; dark partner to SAGE
GREY = "#9A9A96"    # baselines and reference lines
INK = "#2B2B2B"     # text, error bars

SURFACE = "#fcfcfb"


def swatch(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({"font.family": "serif",
                                "font.serif": ["Charter", "Palatino", "Georgia"]})
    rows = [("ANTHROPIC", ANTHROPIC), ("OPENAI", OPENAI), ("BLUE", BLUE),
            ("NAVY", NAVY), ("SAGE", SAGE), ("FOREST", FOREST), ("GREY", GREY)]
    fig, ax = plt.subplots(figsize=(7.0, 3.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for i, (name, c) in enumerate(rows):
        ax.bar(i - 0.18, 1.0, 0.34, color=c, linewidth=0)
        ax.bar(i + 0.18, 0.6, 0.34, color=tint(c), linewidth=0)
        ax.text(i, 1.04, name, ha="center", fontsize=9, color=INK)
        ax.text(i, -0.06, c, ha="center", va="top", fontsize=8.5, color=INK)
    ax.text(-0.5, 0.62, "tint", ha="right", va="center", fontsize=9, color=INK)
    ax.set_xlim(-0.7, len(rows) - 0.4)
    ax.set_ylim(-0.25, 1.15)
    ax.axis("off")
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=SURFACE)
    print("wrote", out)


if __name__ == "__main__":
    out = Path(__file__).resolve().parent / "output"
    out.mkdir(exist_ok=True)
    swatch(out / "palette_swatch.png")

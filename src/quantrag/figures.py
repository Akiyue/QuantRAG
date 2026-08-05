"""Figure style for the paper.

Three constraints shape everything here, in this order:

1. **Print.** Conference proceedings get printed, sometimes in greyscale. Every
   series therefore carries identity twice - colour *and* marker or line style -
   so nothing depends on hue surviving a photocopier.
2. **Colour-vision deficiency.** The categorical slots below are the first three
   of a palette validated all-pairs (worst CVD ΔE 9.2, normal-vision 24.0). The
   ordinal ramp for the precision ladder is single-hue and monotone in lightness,
   which is what makes an ordered variable read as ordered.
3. **One axis, always.** Two measures of different scale become two panels, never
   two y-scales on one plot.

Precision is an *ordered* variable, so it gets a sequential ramp, never
categorical hues. Models and languages are identities, so they get categorical
slots and never a ramp.
"""

from __future__ import annotations

from pathlib import Path

# Categorical slots 1-3. Validated all-pairs in light mode; a fourth slot would
# put yellow beside orange and fail the floor, so three series is the cap. More
# than three means faceting, not a new colour.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]

# Ordered ramp for the precision ladder: one hue, monotone lightness, light end
# still visible against the surface.
LADDER = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]

# Diverging pair for signed quantities (flips toward the context vs back toward
# memory). Warm/cool poles with a neutral - never a hue at the midpoint.
POS, NEG, MID = "#2a78d6", "#d03b3b", "#f0efec"

INK = "#0b0b0b"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"

# Redundant channels, so identity never rests on colour alone.
MARKERS = ["o", "s", "^", "D", "v"]
LINESTYLES = {"en": "-", "vi": "--"}

PRECISION_ORDER = ["F16", "Q8_0", "Q4_K_M", "Q3_K_M", "AWQ4"]


def apply_style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        # Recessive chrome: the data should be the darkest thing on the page.
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.6,
        "lines.markersize": 5,
        "figure.dpi": 140,
        "savefig.bbox": "tight",
    })


def ladder_color(precision: str) -> str:
    """Position on the precision ramp, so ordering is visible at a glance."""
    if precision in PRECISION_ORDER:
        i = PRECISION_ORDER.index(precision)
        return LADDER[min(i, len(LADDER) - 1)]
    return INK_MUTED


def save(fig, out_dir: Path, name: str) -> None:
    """PDF for the paper, PNG for looking at."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{name}.{ext}")
    print(f"  wrote {name}.pdf / .png")


def zero_line(ax) -> None:
    ax.axhline(0, color=AXIS, linewidth=1.0, zorder=1)

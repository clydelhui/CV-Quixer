"""Shared styling for the thesis figure scripts.

`thesis_figures.py` (cross-run comparisons) and `thesis_run_figures.py` (per-run
suites) are presentation layers over the general-purpose reporting tools. Both
need the same look, and — more importantly — the same *identities*: a model
variant must keep one colour across every figure in the chapter, and so must a
head index within a run. That shared vocabulary lives here.

Nothing in this module reads data or knows about sweeps; it is style only.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Identities
# ---------------------------------------------------------------------------

# Legend text for the model variants, verbatim (thesis convention). Order fixes
# the legend order; the colours are shared across every figure so a variant keeps
# one identity throughout the chapter. Okabe-Ito, colour-blind safe.
MODEL_ORDER = ["quantum", "quantum_shared", "quantum_stacked"]
MODEL_COLORS = {
    "quantum": "#0072B2",          # blue
    "quantum_shared": "#D55E00",   # vermillion
    "quantum_stacked": "#009E73",  # bluish green
}
MODEL_MARKERS = {"quantum": "o", "quantum_shared": "s", "quantum_stacked": "^"}

FIGSIZE = (7.5, 5.0)
DPI = 200


def head_colors(n: int):
    """One colour per attention head, stable for a given head count.

    Sampled from viridis rather than the categorical cycle: these runs carry up
    to 10 heads, well past the point where categorical palettes stop being
    distinguishable, and head index is an ordinal axis anyway. Using one ramp
    everywhere means "head 3" is the same colour in every figure of a run.
    """
    if n <= 1:
        return [plt.get_cmap("viridis")(0.5)]
    return [plt.get_cmap("viridis")(i / (n - 1) * 0.9) for i in range(n)]


# How overlapping trend lines are drawn, as (line alpha, marker alpha) plus the
# filename suffix. Applied as RGBA on the colour rather than artist-level
# `alpha=`, which would dim line and markers together.
OPACITY_MODES = {
    # Default: everything translucent — crossings stay legible and overlapping
    # configurations read darker, at the cost of slightly dim markers.
    "translucent": (0.55, 0.55, ""),
    # Crisp data points, connectors still out of each other's way.
    "solid-markers": (0.55, 1.0, "__solid_markers"),
    # Cleanest individual chains; overlap regions become unreadable.
    "opaque": (1.0, 1.0, "__opaque"),
}


# ---------------------------------------------------------------------------
# Common furniture
# ---------------------------------------------------------------------------


def style_axes(ax) -> None:
    """The common thesis look: light grid, no top/right spines."""
    ax.grid(alpha=0.25, linewidth=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


SUBTITLE_WRAP = 82


def titles(fig, ax, title: str, subtitle: str) -> None:
    """A bold headline plus a small grey provenance line above the axes.

    With `ax=None` the subtitle is drawn as a figure-level line instead, for
    multi-panel figures where no single axes owns it. The subtitle is wrapped
    rather than trusted to fit — it carries the run's whole architecture, which
    overflows the canvas on wide configurations.
    """
    wrapped = textwrap.fill(subtitle, SUBTITLE_WRAP)
    fig.suptitle(title, fontsize=13, y=0.98)
    if ax is None:
        fig.text(0.5, 0.925, wrapped, ha="center", va="top", fontsize=9,
                 color="#555555")
    else:
        ax.set_title(wrapped, fontsize=9, color="#555555", pad=6)


def variant_legend(ax, models: list[str]) -> None:
    """Three-entry legend keyed by model variant, in MODEL_ORDER."""
    handles = [
        Line2D([], [], color=MODEL_COLORS[m], marker=MODEL_MARKERS[m],
               markersize=5, linewidth=1.6, label=m)
        for m in MODEL_ORDER if m in models
    ]
    ax.legend(handles=handles, title="Model variant", frameon=True,
              framealpha=0.9, fontsize=9, title_fontsize=9)


def outside_legend(ax, title: str | None = None, **kw):
    """Legend parked to the right of the axes.

    Per-head legends run to 10 entries on these runs, which no in-axes placement
    survives; hoisting them out keeps the data area clean.
    """
    return ax.legend(
        loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False,
        fontsize=8, title=title, title_fontsize=9, **kw
    )


def show_path(path: Path) -> str:
    """Repo-relative path for logging, falling back to absolute."""
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def save(fig, out_dir: Path, stem: str, rect=(0, 0, 1, 0.945),
         quiet: bool = False) -> None:
    """Write one figure as both PNG (review) and PDF (LaTeX inclusion)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=rect)
    for ext in ("png", "pdf"):
        path = out_dir / f"{stem}.{ext}"
        fig.savefig(path, dpi=DPI)
        if not quiet:
            print(f"  ✓ {show_path(path)}")
    plt.close(fig)


def grid_axes(n: int, ncols: int = 5, panel=(2.9, 2.4), sharey: bool = True):
    """A wrapped grid of `n` panels, with the unused trailing cells removed.

    The reason this exists: `report_diagnostics.plot_polynomial_coefficient_
    trajectory` lays its per-head panels out as a single row
    (`figsize=(3.5 * num_heads, 4)`), which at the 10 heads these runs use is a
    35-inch-wide strip — unusable in a document. Wrapping keeps every panel
    legible at page width.
    """
    nrows = int(np.ceil(n / ncols))
    ncols = min(ncols, n)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(panel[0] * ncols, panel[1] * nrows),
        sharex=True, sharey=sharey, squeeze=False,
    )
    flat = axes.ravel()
    for ax in flat[n:]:
        ax.remove()
    return fig, list(flat[:n])

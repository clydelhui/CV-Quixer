"""Shared styling for the thesis figure scripts.

`thesis_figures.py` (cross-run comparisons) and `thesis_run_figures.py` (per-run
suites) are presentation layers over the general-purpose reporting tools. Both
need the same look, and — more importantly — the same *identities*: a model
variant must keep one colour across every figure in the chapter, and so must a
head index within a run. That shared vocabulary lives here.

Nothing in this module reads data or knows about sweeps; it is style only.
"""

from __future__ import annotations

import re


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


# Titles and subtitles are deliberately absent from every figure. In a bound
# document the caption states all of it, so a figure that repeats its own
# caption both wastes 15-20% of the canvas and reads as duplicated. What a
# figure keeps is its *grammar* — axis labels, tick labels, legends, colourbar
# labels — which no caption can substitute for. Everything removed is emitted
# by CaptionBook (below) as LaTeX, so the identifying detail — which of sixteen
# runs a figure belongs to, the summary statistics — survives in a pasteable
# form.

# The standard caveat wording, defined once so the three scripts cannot drift
# apart on how they describe the same experimental fact.
SINGLE_SEED = "single seed"


def variant_legend(ax, models: list[str], **kw) -> None:
    """Three-entry legend keyed by model variant, in MODEL_ORDER.

    Placement kwargs (`loc`, `bbox_to_anchor`) pass through: matplotlib's "best"
    has no notion of which empty region is *meaningfully* empty, and on a dense
    figure it reliably lands on the data. Pushing the legend outside the axes is
    the reliable answer when no in-axes region is genuinely free.
    """
    handles = [
        Line2D([], [], color=MODEL_COLORS[m], marker=MODEL_MARKERS[m],
               markersize=5, linewidth=1.6, label=m)
        for m in MODEL_ORDER if m in models
    ]
    kw.setdefault("frameon", "bbox_to_anchor" not in kw)
    ax.legend(handles=handles, title="Model variant", framealpha=0.9,
              fontsize=9, title_fontsize=9, **kw)


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


def save(fig, out_dir: Path, stem: str, rect=None, quiet: bool = False) -> None:
    """Write one figure as both PNG (review) and PDF (LaTeX inclusion).

    With no titles to reserve space for, plain `tight_layout()` is right; `rect`
    remains for the rare figure that parks an artist outside the axes.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    if rect is None:
        fig.tight_layout()
    else:
        fig.tight_layout(rect=rect)
    for ext in ("png", "pdf"):
        path = out_dir / f"{stem}.{ext}"
        fig.savefig(path, dpi=DPI)
        if not quiet:
            print(f"  ✓ {show_path(path)}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Captions
# ---------------------------------------------------------------------------

# TeX's specials. Backslash first, or the replacements would be re-escaped.
_TEX_SPECIALS = [
    ("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"),
    ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}"),
    ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}"),
]


def tex_escape(text: str) -> str:
    """Escape TeX specials in literal text, leaving `$...$` math spans intact.

    Escaping matters more than it looks: model variants (`quantum_shared`) and
    run names are riddled with underscores, which are subscript operators in TeX
    and would otherwise fail to compile or silently mangle the caption.

    Math spans are exempt so captions can use real notation — `$D = 6$` sets an
    italic variable, and `$-0.0188$` a true minus sign rather than the hyphen a
    bare "-" would produce in text mode.
    """
    parts = re.split(r"(\$[^$]*\$)", str(text))
    for i, part in enumerate(parts):
        if part.startswith("$") and part.endswith("$") and len(part) > 1:
            continue                      # math: pass through verbatim
        for char, replacement in _TEX_SPECIALS:
            part = part.replace(char, replacement)
        parts[i] = part
    return "".join(parts)


def _slug(text: str) -> str:
    """A LaTeX-label-safe slug: lowercase alphanumerics and hyphens."""
    cleaned = "".join(c if c.isalnum() else "-" for c in str(text).lower())
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return cleaned.strip("-")


class CaptionBook:
    """Collects per-figure caption material and writes it as LaTeX.

    Each entry becomes a complete `figure` environment — graphic, short caption
    for the list of figures, full caption, and label — so the output can be
    `\\input` directly or pasted piecemeal. Written once per output directory
    alongside the figures, so the captions cannot drift from what was rendered.
    """

    def __init__(self, out_dir: Path, label_prefix: str = "",
                 graphics_dir: str = "", filename: str = "captions.tex") -> None:
        self.out_dir = Path(out_dir)
        self.label_prefix = label_prefix
        self.graphics_dir = graphics_dir
        # Two scripts can target one output directory (the comparison figures
        # and the cutoff figures both land in compare/figures/thesis), so the
        # filename is per-script or the second would clobber the first.
        self.filename = filename
        self._entries: list[dict] = []

    def add(self, stem: str, short: str, body: str,
            facts: dict[str, str] | None = None) -> None:
        """Record one figure.

        Args:
            stem:  figure filename without extension.
            short: one-line description, used as the LoF entry and caption lead.
            body:  what the reader is looking at — encodings, axes, groupings.
            facts: trailing "key: value" details (run identity, summary
                   statistics, caveats), rendered as an italicised tail.
        """
        self._entries.append(
            {"stem": stem, "short": short, "body": body, "facts": facts or {}}
        )

    def _render(self, entry: dict) -> str:
        stem = entry["stem"]
        path = f"{self.graphics_dir}{stem}" if self.graphics_dir else stem
        label = f"fig:{self.label_prefix}{_slug(stem)}" if self.label_prefix \
            else f"fig:{_slug(stem)}"
        short = tex_escape(entry["short"])
        parts = [f"{short}. {tex_escape(entry['body'])}"]
        for key, value in entry["facts"].items():
            parts.append(f"\\emph{{{tex_escape(key)}:}} {tex_escape(value)}.")
        caption = "\n    ".join(parts)
        return (
            f"% {'-' * 70}\n"
            f"\\begin{{figure}}[htbp]\n"
            f"  \\centering\n"
            f"  \\includegraphics[width=\\linewidth]{{{path}}}\n"
            f"  \\caption[{short}]{{%\n    {caption}\n  }}\n"
            f"  \\label{{{label}}}\n"
            f"\\end{{figure}}\n"
        )

    def write(self, quiet: bool = False) -> Path | None:
        """Write the caption file into the output directory."""
        if not self._entries:
            return None
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / self.filename
        header = (
            "% Auto-generated by the thesis figure scripts — do not edit here;\n"
            "% regenerate, or copy the blocks you want into the document.\n"
            "% Figures carry no in-figure title or annotation by design: every\n"
            "% identifying and summary detail lives in these captions.\n\n"
        )
        with open(path, "w") as f:
            f.write(header)
            f.write("\n".join(self._render(e) for e in self._entries))
        if not quiet:
            print(f"  ✓ {show_path(path)}")
        return path


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

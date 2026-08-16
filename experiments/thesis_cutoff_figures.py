"""Thesis-ready figures for the Fock-cutoff re-evaluation sweep.

`report_cutoff_sweep.py` writes one figure set per sweep directory, so a study
spanning the three model variants comes out as three disconnected sets of plots
whose labels are raw column names. This script is the presentation layer over
its output: it merges the sweeps into one figure per metric, coloured by model
variant with the same palette as the other thesis figures.

Form: a **slope chart**. The sweep re-evaluates checkpoints trained at one
cutoff at a second, larger one, so each run contributes exactly two points —
a before/after pair. Drawing that as a many-line "metric vs cutoff" plot wastes
a whole axis on two tick positions; a slope chart puts the change itself in the
visual encoding, so whether a run improves or degrades is readable at a glance
across every configuration at once.

Metrics (one figure each):

    accuracy         test accuracy at each cutoff
    truncation_loss  Fock-space leakage — the mechanism behind any accuracy change
    photon_number    mean photon number, i.e. how much of the larger space is used

Deliberately absent: ``mean_state_norm``. It is identically 1.0 in every row,
because the norm is measured after the post-selection and post-W
renormalisations — a figure of it would be a flat line at 1 carrying no
information.

Usage:
    uv run python experiments/thesis_cutoff_figures.py
    uv run python experiments/thesis_cutoff_figures.py --metric accuracy
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba

from _thesis_style import (
    SINGLE_SEED,
    CaptionBook,
    MODEL_COLORS,
    MODEL_MARKERS,
    REPO_ROOT,
    save,
    show_path,
    style_axes,
    top_variant_legend,
)

SWEEPS = REPO_ROOT / "results" / "sweeps"

# The sweeps this study spans, and where the merged figures belong — alongside
# the other cross-variant comparison figures rather than inside any one sweep.
DEFAULT_SWEEP_DIRS = [
    SWEEPS / "high_epoch_quantum_2026-06-15_03-47-33",
    SWEEPS / "high_epoch_shared_2026-06-15_03-52-41",
    SWEEPS / "high_epoch_stacked_2026-06-15_04-02-35",
]
DEFAULT_OUT_DIR = SWEEPS / "extended25_subsets" / "compare" / "figures" / "thesis"

# Line translucent so 16 crossing segments stay legible; markers opaque so the
# data points themselves stay crisp (the "solid-markers" mode of the ablation
# figures, which read best there).
_LINE_ALPHA = 0.55


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


class MissingCutoffData(RuntimeError):
    """The cutoff sweep has not been aggregated, or produced nothing usable."""


def _resolve_model(sweep_dir: Path, run_name: str) -> str:
    """Model variant for a run, from its own config.json.

    Read per run rather than inferred from the sweep directory name: the name is
    a convention, the config is the record.
    """
    config_path = sweep_dir / run_name / "config.json"
    if config_path.is_file():
        try:
            with open(config_path) as f:
                model = json.load(f).get("model")
            if model:
                return str(model)
        except (json.JSONDecodeError, OSError):
            pass
    return "quantum"


def _warn_if_stale(sweep_dir: Path, summary_path: Path) -> None:
    """Warn when a per-run results.json is newer than the aggregated summary.

    `cutoff_summary.csv` is a snapshot: re-running an eval without re-running
    report_cutoff_sweep.py leaves it silently out of date (the same trap the
    per-sweep summary.csv files fell into).
    """
    summary_mtime = summary_path.stat().st_mtime
    newer = [
        p.parent.parent.parent.name
        for p in sweep_dir.glob("*/eval/*/results.json")
        if p.stat().st_mtime > summary_mtime
    ]
    if newer:
        warnings.warn(
            f"{summary_path.name} is older than the results.json of "
            f"{len(newer)} run(s) in {sweep_dir.name} (e.g. {sorted(newer)[:3]}) "
            "— re-run `report_cutoff_sweep.py --sweep-dir` to refresh it.",
            RuntimeWarning, stacklevel=2,
        )


def load_cutoff_rows(sweep_dirs: list[Path]) -> list[dict]:
    """Every (run, split, cutoff) row across the given sweeps, tagged by model."""
    rows: list[dict] = []
    for sweep_dir in sweep_dirs:
        summary = sweep_dir / "cutoff_summary.csv"
        if not summary.is_file():
            raise MissingCutoffData(
                f"{summary} not found — run `uv run python "
                f"experiments/report_cutoff_sweep.py --sweep-dir {sweep_dir}` "
                "(on the cluster, where the eval artefacts live) first."
            )
        _warn_if_stale(sweep_dir, summary)
        with open(summary) as f:
            for row in csv.DictReader(f):
                row["model"] = _resolve_model(sweep_dir, row["run_name"])
                row["sweep_dir"] = sweep_dir.name
                rows.append(row)
    if not rows:
        raise MissingCutoffData("cutoff summaries contained no rows")
    return rows


def pair_by_run(rows: list[dict], metric: str, split: str = "test") -> tuple:
    """Group rows into per-run {cutoff: value} pairs.

    Returns ``(pairs, cutoffs)`` where pairs is a list of
    ``(model, run_key, {cutoff: float})``. Run names repeat across the variant
    sweeps (the model is not encoded in them), so runs are keyed by sweep
    directory as well as name.
    """
    by_run: dict[tuple, dict[int, float]] = defaultdict(dict)
    models: dict[tuple, str] = {}
    for row in rows:
        if row.get("split") != split:
            continue
        value = row.get(metric)
        if value in (None, ""):
            continue
        key = (row["sweep_dir"], row["run_name"])
        by_run[key][int(row["cutoff_dim"])] = float(value)
        models[key] = row["model"]

    cutoffs = sorted({c for v in by_run.values() for c in v})
    if len(cutoffs) < 2:
        raise MissingCutoffData(
            f"need >=2 distinct cutoffs to draw a slope chart, found {cutoffs}"
        )
    pairs = [(models[k], k, v) for k, v in by_run.items()]
    incomplete = [k[1] for _m, k, v in pairs if len(v) < len(cutoffs)]
    if incomplete:
        warnings.warn(
            f"{len(incomplete)} run(s) lack a value at every cutoff and will be "
            f"drawn as partial segments: {sorted(set(incomplete))[:3]}",
            RuntimeWarning, stacklevel=2,
        )
    return pairs, cutoffs


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

METRICS = {
    "accuracy": {
        "column": "acc",
        "short": "Test accuracy under an increased Fock cutoff",
        "ylabel": "Test accuracy",
        "stem": "cutoff_accuracy",
        "log": False,
        "delta_fmt": "{:+.4f}",
    },
    "truncation_loss": {
        "column": "trunc_loss",
        "short": "Fock-space truncation loss under an increased cutoff",
        "ylabel": "Truncation loss (leaked probability)",
        "stem": "cutoff_truncation_loss",
        # Leakage spans more than a decade across configurations and falls by a
        # roughly constant factor, so a log axis makes the slopes comparable.
        "log": True,
        "delta_fmt": "{:+.4f}",
    },
    "photon_number": {
        "column": "mean_photon",
        "short": "Mean photon number under an increased Fock cutoff",
        "ylabel": r"Mean photon number $\langle \hat n \rangle$",
        "stem": "cutoff_photon_number",
        "log": False,
        "delta_fmt": "{:+.3f}",
    },
}


def plot_slope(rows: list[dict], out_dir: Path, metric: str,
               book: CaptionBook) -> None:
    """One segment per run between the evaluated cutoffs, coloured by variant."""
    spec = METRICS[metric]
    pairs, cutoffs = pair_by_run(rows, spec["column"])
    pos = {c: i for i, c in enumerate(cutoffs)}
    training_cutoff = _training_cutoff(rows)

    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    models_seen: set[str] = set()
    for model, _key, values in sorted(pairs, key=lambda p: (p[0], str(p[1]))):
        models_seen.add(model)
        xs = [pos[c] for c in sorted(values)]
        ys = [values[c] for c in sorted(values)]
        base = MODEL_COLORS.get(model, "#666666")
        ax.plot(
            xs, ys, marker=MODEL_MARKERS.get(model, "o"), markersize=5,
            linewidth=1.4, color=to_rgba(base, _LINE_ALPHA),
            markerfacecolor=base, markeredgecolor=base,
        )

    ax.set_xticks(range(len(cutoffs)))
    ax.set_xticklabels([f"$D = {c}$" for c in cutoffs])
    # Just enough margin to clear the markers. A slope chart carries its
    # signal in the segment angle, so wide side margins both waste canvas and
    # flatten every slope.
    ax.set_xlim(-0.12, len(cutoffs) - 1 + 0.12)
    ax.set_xlabel("Fock cutoff dimension")
    ax.set_ylabel(spec["ylabel"])
    if spec["log"]:
        ax.set_yscale("log")

    style_axes(ax)
    top_variant_legend(ax, sorted(models_seen))
    save(fig, out_dir, spec["stem"])
    book.add(
        stem=spec["stem"],
        short=spec["short"],
        body=(
            f"Each line segment is one configuration, drawn between the two "
            f"evaluated Fock cutoffs and coloured by model variant. "
            f"{_change_sentence(pairs, cutoffs, spec['delta_fmt'], spec['log'])}"
        ),
        facts=_facts(pairs, cutoffs, training_cutoff),
    )
    print(f"    {len(pairs)} run(s) at cutoffs {cutoffs}")


def _training_cutoff(rows: list[dict]) -> int | None:
    """The cutoff the checkpoints were trained at, if unambiguous."""
    values = {int(r["training_cutoff"]) for r in rows if r.get("training_cutoff")}
    return values.pop() if len(values) == 1 else None


def _facts(pairs: list, cutoffs: list[int],
           training_cutoff: int | None) -> dict[str, str]:
    trained = (f", trained at $D = {training_cutoff}$ and re-evaluated without "
               "further training" if training_cutoff is not None else "")
    return {
        "Data": f"FashionMNIST test set, {len(pairs)} configurations{trained}",
        "Cutoffs": ", ".join(f"$D = {c}$" for c in cutoffs),
        "Caveat": SINGLE_SEED,
    }


def _change_sentence(pairs: list, cutoffs: list[int], fmt: str,
                     ratio: bool) -> str:
    """Summary of the change across the cutoffs, for the caption.

    On a log axis the slope a reader sees is a *ratio*, so the summary is stated
    multiplicatively there; quoting an absolute difference would describe
    something the figure does not show.
    """
    lo, hi = cutoffs[0], cutoffs[-1]
    both = [v for _m, _k, v in pairs if lo in v and hi in v]
    if not both:
        return ""
    if ratio:
        factors = [v[lo] / v[hi] for v in both if v[hi] > 0]
        if not factors:
            return ""
        return (f"Every configuration decreases, by a factor of "
                f"${min(factors):.1f}$ to ${max(factors):.1f}$.")
    deltas = [v[hi] - v[lo] for v in both]
    down = sum(1 for d in deltas if d < 0)
    direction = (f"{down} of {len(deltas)} decrease" if 0 < down < len(deltas)
                 else ("every configuration decreases" if down
                       else "every configuration increases"))
    return (f"From $D = {lo}$ to $D = {hi}$, {direction}, by "
            f"${fmt.format(min(deltas))}$ to ${fmt.format(max(deltas))}$.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render thesis-ready figures for the Fock-cutoff sweep.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--sweep-dir", action="append",
                        help="sweep dir holding a cutoff_summary.csv (repeatable; "
                             "defaults to the three model-variant sweeps)")
    parser.add_argument("--out-dir", help="override the output directory")
    parser.add_argument("--metric", action="append", choices=list(METRICS),
                        help="render just this metric (repeatable)")
    args = parser.parse_args()

    sweep_dirs = ([Path(d) for d in args.sweep_dir] if args.sweep_dir
                  else DEFAULT_SWEEP_DIRS)
    out_dir = Path(args.out_dir) if args.out_dir else DEFAULT_OUT_DIR
    metrics = args.metric or list(METRICS)

    missing = [d for d in sweep_dirs if not Path(d).is_dir()]
    if missing:
        raise SystemExit(f"sweep dir(s) not found: {[str(m) for m in missing]}")

    try:
        rows = load_cutoff_rows([Path(d) for d in sweep_dirs])
    except MissingCutoffData as exc:
        raise SystemExit(f"error: {exc}") from exc

    print(f"Cutoff figures -> {show_path(out_dir)}\n")
    book = CaptionBook(out_dir, filename="captions_cutoff.tex")
    failures = []
    for metric in metrics:
        print(f"[{metric}]")
        try:
            plot_slope(rows, out_dir, metric, book)
        except MissingCutoffData as exc:
            print(f"  ✗ {exc}")
            failures.append(metric)
    book.write()
    if failures:
        print(f"\n{len(failures)} figure(s) not rendered.")
        sys.exit(1)
    print("\nDone.")


if __name__ == "__main__":
    main()

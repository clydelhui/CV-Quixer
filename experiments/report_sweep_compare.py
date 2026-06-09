"""Compare two or more CV-Quixer sweeps side by side.

Where `report_sweep.py` aggregates a *single* sweep, this script overlays
several — typically one sweep per model variant (e.g. an already-finished
`quantum` sweep vs a new `quantum_shared` sweep) — into one combined table and a
pair of cross-sweep comparison figures. It reads only JSON (no torch / model
rebuild) and reuses `report_sweep`'s run loader and aggregation helpers, so it
never reimplements run parsing and stays fast on partial/in-progress sweeps.

The series are keyed on (sweep_label, model, observables, scaling_knob): leading
with the sweep label means two sweeps that share a model/observable/knob — e.g.
two `quantum` sweeps, or a re-run vs the original — stay separate instead of being
averaged into one point. Within a sweep, the same key also keeps distinct scaling
knobs apart (e.g. the quantum knobsweep has both `cnn_channels_2` and `num_heads`
runs at 12,800, genuinely different architectures). Colour encodes the sweep
label (one per --label); marker/linestyle encode the knob.

Outputs (under --out-dir, default results/sweeps/compare_<ts>/):

    comparison.csv / comparison.md      every run across all sweeps, one row each
    figures/acc_vs_params_compare.png   best test acc vs achieved param count
    figures/acc_by_params_compare.png   grouped bars: model/knob at each budget

Usage:
    uv run python experiments/report_sweep_compare.py \\
        --sweep-dir results/sweeps/full_xpxsps_knobsweep_L2_<ts>/ \\
        --sweep-dir results/sweeps/shared_xpxsps_paramsweep_L2_<ts>/ \\
        --label quantum --label quantum_shared
"""

from __future__ import annotations

import argparse
import csv
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Reuse the single-sweep loader + aggregation helpers (JSON only, no torch).
from report_sweep import _aggregate_by, _mean_value_by, load_sweep

# Series identity: never collapse different sweeps, models, or scaling knobs into
# one point. Including sweep_label keeps same-model runs from different sweeps
# separate (one series per --label). (Seeds are still averaged within a key by
# _aggregate_by.)
SERIES_FIELDS = ("sweep_label", "model", "observables", "scaling_knob", "target_params")

# Columns for the combined table, in display order.
COMPARISON_COLUMNS = [
    "sweep_label", "run_name", "model", "observables", "scaling_knob",
    "target_params", "achieved_params", "num_layers", "trunc_lambda", "seed",
    "best_test_acc", "best_epoch", "final_test_acc", "final_train_acc",
    "n_epochs", "total_runtime_sec", "device",
]

# Plot markers cycled per scaling knob (linestyle paired so b/w prints differ).
_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
_LINESTYLES = ["-", "--", "-.", ":"]


def _fmt(v: object) -> str:
    if isinstance(v, float):
        return f"{v:.4f}"
    return "" if v is None else str(v)


def load_sweeps(sweep_dirs: list[Path], labels: list[str]) -> list[dict]:
    """Load every run across all sweeps, each row tagged with its sweep label."""
    rows: list[dict] = []
    for sweep_dir, label in zip(sweep_dirs, labels):
        sweep_rows = load_sweep(sweep_dir)
        if not sweep_rows:
            warnings.warn(
                f"no runs with history.json under {sweep_dir} — skipping",
                RuntimeWarning,
            )
            continue
        for r in sweep_rows:
            r["sweep_label"] = label
            rows.append(r)
    return rows


def write_table(rows: list[dict], out_dir: Path) -> None:
    """Write comparison.csv and comparison.md (one row per run, all sweeps)."""
    csv_path = out_dir / "comparison.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COMPARISON_COLUMNS)
        writer.writeheader()
        writer.writerows({k: r.get(k) for k in COMPARISON_COLUMNS} for r in rows)
    print(f"  ✓ {csv_path}")

    md_path = out_dir / "comparison.md"
    with open(md_path, "w") as f:
        f.write("| " + " | ".join(COMPARISON_COLUMNS) + " |\n")
        f.write("|" + "|".join(["---"] * len(COMPARISON_COLUMNS)) + "|\n")
        for r in rows:
            f.write("| " + " | ".join(_fmt(r.get(k)) for k in COMPARISON_COLUMNS) + " |\n")
    print(f"  ✓ {md_path}")


def _series_style(rows: list[dict]) -> tuple[dict[str, object], dict[str, tuple]]:
    """Stable colour-per-sweep-label and marker/linestyle-per-knob maps."""
    labels = sorted({str(r.get("sweep_label")) for r in rows})
    knobs = sorted({str(r.get("scaling_knob")) for r in rows})
    cmap = plt.get_cmap("tab10")
    color_by_label = {lbl: cmap(i % 10) for i, lbl in enumerate(labels)}
    style_by_knob = {
        k: (_MARKERS[i % len(_MARKERS)], _LINESTYLES[i % len(_LINESTYLES)])
        for i, k in enumerate(knobs)
    }
    return color_by_label, style_by_knob


def plot_acc_vs_params(rows: list[dict], fig_dir: Path) -> None:
    """Best test acc vs achieved param count, one line per (sweep, model, obs, knob)."""
    agg = _aggregate_by(rows, SERIES_FIELDS)
    if not agg:
        print("  (no runs with best_test_acc — skipping acc_vs_params_compare)")
        return
    x_by_key = _mean_value_by(rows, SERIES_FIELDS, "achieved_params")
    color_by_label, style_by_knob = _series_style(rows)

    series = sorted({(lbl, m, o, k) for (lbl, m, o, k, _tp) in agg})
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for label, model, obs, knob in series:
        pts = sorted(
            (x_by_key[(lbl, m, o, k, tp)], mean, std)
            for (lbl, m, o, k, tp), (mean, std) in agg.items()
            if (lbl, m, o, k) == (label, model, obs, knob)
            and (lbl, m, o, k, tp) in x_by_key
        )
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        es = [p[2] for p in pts]
        marker, linestyle = style_by_knob[str(knob)]
        ax.errorbar(
            xs, ys, yerr=es, marker=marker, linestyle=linestyle, capsize=3,
            color=color_by_label[str(label)], label=f"{label}/{model}/{obs}/{knob}",
        )
    ax.set_xlabel("Achieved parameter count")
    ax.set_ylabel("Best test accuracy")
    ax.set_title("Accuracy vs achieved params (colour = sweep label, marker = scaling knob)")
    ax.grid(alpha=0.3)
    ax.legend(title="sweep / model / observable / knob", fontsize=8)
    fig.tight_layout()
    out = fig_dir / "acc_vs_params_compare.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  ✓ {out}")


def plot_acc_by_params(rows: list[dict], fig_dir: Path) -> None:
    """Grouped bars: best test acc by (sweep, model, obs, knob), grouped by budget."""
    agg = _aggregate_by(rows, SERIES_FIELDS)
    if not agg:
        print("  (no runs with best_test_acc — skipping acc_by_params_compare)")
        return
    achieved = _mean_value_by(rows, SERIES_FIELDS, "achieved_params")
    color_by_label, _ = _series_style(rows)

    cats = sorted({(lbl, m, o, k) for (lbl, m, o, k, _tp) in agg})
    budgets = sorted({tp for (_lbl, _m, _o, _k, tp) in agg})
    x = np.arange(len(budgets))
    width = 0.8 / max(len(cats), 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (label, model, obs, knob) in enumerate(cats):
        means = [agg.get((label, model, obs, knob, tp), (np.nan, 0.0))[0] for tp in budgets]
        errs = [agg.get((label, model, obs, knob, tp), (np.nan, 0.0))[1] for tp in budgets]
        bars = ax.bar(
            x + i * width, means, width, yerr=errs, capsize=3,
            color=color_by_label[str(label)], label=f"{label}/{model}/{obs}/{knob}",
        )
        for tp, bar in zip(budgets, bars):
            n = achieved.get((label, model, obs, knob, tp))
            if n is not None and not np.isnan(bar.get_height()):
                ax.annotate(
                    f"{int(round(n)):,}",
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center", va="bottom", fontsize=6, rotation=90,
                )
    ax.set_xticks(x + width * (len(cats) - 1) / 2)
    ax.set_xticklabels([f"{tp:,}" for tp in budgets])
    ax.set_xlabel("Target parameter budget")
    ax.set_ylabel("Best test accuracy")
    ax.set_title("Accuracy by sweep / model / knob at each budget (bar labels = achieved params)")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(title="sweep / model / observable / knob", fontsize=8)
    fig.tight_layout()
    out = fig_dir / "acc_by_params_compare.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  ✓ {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweep-dir", type=str, action="append", required=True, dest="sweep_dirs",
        help="a sweep directory written by experiments/sweep.py (repeat ≥2 times)",
    )
    parser.add_argument(
        "--label", type=str, action="append", dest="labels", default=None,
        help="display label for the matching --sweep-dir (repeat; defaults to "
        "each sweep dir's basename)",
    )
    parser.add_argument(
        "--out-dir", type=str, default=None,
        help="output directory (default: results/sweeps/compare_<ts>/)",
    )
    args = parser.parse_args()

    sweep_dirs = [Path(d) for d in args.sweep_dirs]
    if len(sweep_dirs) < 2:
        parser.error("pass --sweep-dir at least twice (one per sweep to compare)")
    for d in sweep_dirs:
        if not d.is_dir():
            parser.error(f"--sweep-dir does not exist: {d}")

    labels = args.labels or [d.name for d in sweep_dirs]
    if len(labels) != len(sweep_dirs):
        parser.error(
            f"got {len(labels)} --label(s) for {len(sweep_dirs)} --sweep-dir(s); "
            "pass one --label per --sweep-dir (or none to default to basenames)"
        )

    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else Path("results/sweeps") / f"compare_{datetime.now():%Y-%m-%d_%H-%M-%S}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_sweeps(sweep_dirs, labels)
    if not rows:
        print("No runs found across the given sweeps (need per-run history.json).")
        return

    print(f"Comparing {len(sweep_dirs)} sweep(s), {len(rows)} run(s) → {out_dir}")
    write_table(rows, out_dir)

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for fn in (plot_acc_vs_params, plot_acc_by_params):
        try:
            fn(rows, fig_dir)
        except Exception as e:  # one bad figure must not abort the rest
            warnings.warn(f"{fn.__name__} failed: {type(e).__name__}: {e}", RuntimeWarning)


if __name__ == "__main__":
    main()

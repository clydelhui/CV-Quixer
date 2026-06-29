"""Compare two or more CV-Quixer sweeps side by side.

Where `report_sweep.py` aggregates a *single* sweep, this script overlays
several — typically one sweep per model variant (e.g. an already-finished
`quantum` sweep vs a new `quantum_shared` sweep) — into one combined table and a
pair of cross-sweep comparison figures. It reads only JSON (no torch / model
rebuild) and reuses `report_sweep`'s run loader and configuration-identity
grouping, so it never reimplements run parsing and stays fast on
partial/in-progress sweeps.

Points are seed-averaged per (sweep label, *configuration identity*) — every
recorded sweep coordinate except the training seed — so manual-mode sweeps get
one point per architecture instead of collapsing into a single average, and two
sweeps that contain the same configuration (e.g. a re-run vs the original) stay
separate. Series (one line each) are the sweep label plus the --series-by
fields (default: model observables); colour encodes the sweep label, marker the
series fields. As in `report_sweep.py`, a series' points are connected only
when exactly one identity field varies across them.

Outputs (under --out-dir, default results/sweeps/compare_<ts>/):

    comparison.csv / comparison.md      every run across all sweeps, one row each
    figures/acc_vs_params_compare.png   best test acc vs achieved param count
    figures/acc_by_params_compare.png   grouped bars: series at each budget
                                        (or per configuration for manual sweeps)

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
from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime
from functools import partial
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Reuse the single-sweep loader + configuration-identity helpers (JSON only).
from _run_selection import read_run_names_file

from report_sweep import (
    CONFIG_IDENTITY_FIELDS,
    _FIELD_INDEX,
    _check_epoch_heterogeneity,
    _check_identity_drift,
    _config_groups,
    _series_key,
    _varying_fields,
    load_sweep,
)

# Columns for the combined table, in display order.
COMPARISON_COLUMNS = [
    "sweep_label", "run_name", "model", "observables", "scaling_knob",
    "target_params", "achieved_params", "num_layers", "trunc_lambda", "seed",
    "best_test_acc", "best_epoch", "final_test_acc", "final_train_acc",
    "n_epochs", "total_runtime_sec", "device",
]

# Plot markers cycled per --series-by group (colour already encodes the sweep).
_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]


def _fmt(v: object) -> str:
    if isinstance(v, float):
        return f"{v:.4f}"
    return "" if v is None else str(v)


def load_sweeps(
    sweep_dirs: list[Path], labels: list[str], max_epoch: int | None = None,
    include: set[str] | None = None,
) -> list[dict]:
    """Load every run across all sweeps, each row tagged with its sweep label.

    When ``include`` is given, only runs whose ``run_name`` is in the set are
    kept (applied per sweep, before the drift check) — the way to restrict the
    comparison to a curated subset (e.g. a degree ablation vs only its source
    configs, not the rest of the baseline sweep).
    """
    rows: list[dict] = []
    for sweep_dir, label in zip(sweep_dirs, labels):
        sweep_rows = load_sweep(sweep_dir, max_epoch=max_epoch)
        if include is not None:
            sweep_rows = [r for r in sweep_rows if r["run_name"] in include]
        if not sweep_rows:
            warnings.warn(
                f"no runs with history.json under {sweep_dir}"
                + (" matching --include-*" if include is not None else "")
                + " — skipping",
                RuntimeWarning,
            )
            continue
        # Per sweep, not across sweeps: two sweeps may legitimately contain the
        # same configuration (that is the point of comparing them).
        _check_identity_drift(sweep_rows)
        for r in sweep_rows:
            r["sweep_label"] = label
            rows.append(r)
    # Across sweeps: the whole point is comparing them, so unequal training
    # horizons anywhere in the combined set make the comparison unfair.
    _check_epoch_heterogeneity(rows)
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


def _per_label_groups(rows: list[dict]) -> dict[tuple, dict]:
    """{(sweep_label, config_key): seed-averaged group}, grouped per sweep."""
    out: dict[tuple, dict] = {}
    for label in sorted({r["sweep_label"] for r in rows}):
        label_rows = [r for r in rows if r["sweep_label"] == label]
        for key, g in _config_groups(label_rows).items():
            out[(label, key)] = g
    return out


def _label_colors(labels: set) -> dict:
    """Stable colour per sweep label."""
    cmap = plt.get_cmap("tab10")
    return {lbl: cmap(i % 10) for i, lbl in enumerate(sorted(labels))}


def _suffix_markers(suffixes: set) -> dict:
    """Stable marker per --series-by value tuple (shared across sweep labels)."""
    return {
        s: _MARKERS[i % len(_MARKERS)]
        for i, s in enumerate(sorted(suffixes, key=str))
    }


def _series_name(label: str, suffix: tuple) -> str:
    return "/".join([str(label), *(str(v) for v in suffix)])


def plot_acc_vs_params(
    rows: list[dict], fig_dir: Path, series_by: Sequence[str]
) -> None:
    """Best test acc vs achieved params, one point per (sweep, configuration)."""
    groups = {
        lk: g for lk, g in _per_label_groups(rows).items() if g["x"] is not None
    }
    if len(groups) < 2:
        print("  (need ≥2 distinct configurations — skipping acc_vs_params_compare)")
        return
    by_series: dict[tuple, dict[tuple, dict]] = defaultdict(dict)
    for (label, key), g in groups.items():
        by_series[(label, _series_key(key, series_by))][key] = g
    colors = _label_colors({lbl for (lbl, _s) in by_series})
    markers = _suffix_markers({s for (_lbl, s) in by_series})

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for label, suffix in sorted(by_series, key=str):
        sgroups = by_series[(label, suffix)]
        pts = sorted((g["x"], g["acc"][0], g["acc"][1]) for g in sgroups.values())
        linestyle = "-" if len(_varying_fields(sgroups)) == 1 else "none"
        ax.errorbar(
            [p[0] for p in pts], [p[1] for p in pts], yerr=[p[2] for p in pts],
            marker=markers[suffix], linestyle=linestyle, capsize=3,
            color=colors[label], label=_series_name(label, suffix),
        )
    ax.set_xlabel("Achieved parameter count")
    ax.set_ylabel("Best test accuracy")
    ax.set_title("Accuracy vs achieved params (colour = sweep label)")
    ax.grid(alpha=0.3)
    ax.legend(title=" / ".join(["sweep", *series_by]), fontsize=8)
    fig.tight_layout()
    out = fig_dir / "acc_vs_params_compare.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  ✓ {out}")


def plot_acc_by_params(
    rows: list[dict], fig_dir: Path, series_by: Sequence[str]
) -> None:
    """Grouped bars: one bar per (sweep, series) at each target budget.

    Budget sweeps keep the historic budgets-on-x layout (skipped unless ≥2
    budgets); when several configurations land in one bar cell — i.e. an
    identity field outside --series-by varies — they are averaged with a
    RuntimeWarning suggesting the fields to add. Manual sweeps (no positive
    budgets anywhere) instead draw one bar per configuration under its
    (sweep, series) category, annotated with achieved params.
    """
    per_cfg = _per_label_groups(rows)
    if not per_cfg:
        print("  (no runs with best_test_acc — skipping acc_by_params_compare)")
        return
    tpi = _FIELD_INDEX["target_params"]

    if any((key[tpi] or 0) > 0 for (_lbl, key) in per_cfg):
        # Budget mode: cells keyed (sweep, series, budget).
        cells: dict[tuple, list[tuple]] = defaultdict(list)
        for (label, key), g in per_cfg.items():
            cells[(label, _series_key(key, series_by), key[tpi])].append((key, g))
        budgets = sorted({tp for (_l, _s, tp) in cells})
        if len(budgets) < 2:
            print("  (need ≥2 target budgets — skipping acc_by_params_compare)")
            return
        merged = {c: v for c, v in cells.items() if len(v) > 1}
        if merged:
            fields = sorted({
                f for v in merged.values()
                for f in _varying_fields([k for k, _g in v])
            })
            warnings.warn(
                f"{len(merged)} bar cell(s) average >1 configuration (varying "
                f"fields: {fields}); add them to --series-by to split the bars.",
                RuntimeWarning,
            )
        cats = sorted({(lbl, s) for (lbl, s, _tp) in cells}, key=str)
        colors = _label_colors({lbl for (lbl, _s) in cats})
        x = np.arange(len(budgets))
        width = 0.8 / max(len(cats), 1)

        fig, ax = plt.subplots(figsize=(8, 5))
        for i, (label, suffix) in enumerate(cats):
            means, errs, achs = [], [], []
            for tp in budgets:
                v = cells.get((label, suffix, tp))
                if not v:
                    means.append(np.nan)
                    errs.append(0.0)
                    achs.append(None)
                    continue
                ms = [g["acc"][0] for _k, g in v]
                means.append(float(np.mean(ms)))
                errs.append(
                    v[0][1]["acc"][1] if len(v) == 1 else float(np.std(ms))
                )
                xs_ = [g["x"] for _k, g in v if g["x"] is not None]
                achs.append(float(np.mean(xs_)) if xs_ else None)
            bars = ax.bar(
                x + i * width, means, width, yerr=errs, capsize=3,
                color=colors[label], label=_series_name(label, suffix),
            )
            for n, bar in zip(achs, bars):
                if n is not None and not np.isnan(bar.get_height()):
                    ax.annotate(
                        f"{int(round(n)):,}",
                        (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        ha="center", va="bottom", fontsize=6, rotation=90,
                    )
        ax.set_xticks(x + width * (len(cats) - 1) / 2)
        ax.set_xticklabels([f"{tp:,}" for tp in budgets])
        ax.set_xlabel("Target parameter budget")
        ax.legend(title=" / ".join(["sweep", *series_by]), fontsize=8)
    else:
        # Manual mode: one bar per configuration under its (sweep, series).
        by_cat: dict[tuple, list[dict]] = defaultdict(list)
        for (label, key), g in per_cfg.items():
            by_cat[(label, _series_key(key, series_by))].append(g)
        if sum(len(v) for v in by_cat.values()) < 2:
            print("  (need ≥2 configurations — skipping acc_by_params_compare)")
            return
        cats = sorted(by_cat, key=str)
        colors = _label_colors({lbl for (lbl, _s) in cats})
        n_bars = max(len(v) for v in by_cat.values())
        width = 0.8 / n_bars
        x = np.arange(len(cats))

        fig, ax = plt.subplots(figsize=(8, 5))
        seen_labels: set = set()
        for ci, (label, suffix) in enumerate(cats):
            cfgs = sorted(by_cat[(label, suffix)], key=lambda g: (g["x"] is None, g["x"]))
            for bi, g in enumerate(cfgs):
                bar = ax.bar(
                    ci + bi * width, g["acc"][0], width, yerr=g["acc"][1],
                    capsize=3, color=colors[label],
                    label=str(label) if label not in seen_labels else None,
                )[0]
                seen_labels.add(label)
                if g["x"] is not None:
                    ax.annotate(
                        f"{int(round(g['x'])):,}",
                        (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        ha="center", va="bottom", fontsize=6, rotation=90,
                    )
        ax.set_xticks(x + width * (n_bars - 1) / 2)
        ax.set_xticklabels(
            [_series_name(lbl, s) for (lbl, s) in cats], rotation=15, ha="right"
        )
        ax.set_xlabel("Sweep / series (one bar per configuration)")
        ax.legend(title="sweep", fontsize=8)
    ax.set_ylabel("Best test accuracy")
    ax.set_title("Accuracy by series at each budget (bar labels = achieved params)")
    ax.grid(alpha=0.3, axis="y")
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
        "--series-by", nargs="+", default=["model", "observables"],
        choices=CONFIG_IDENTITY_FIELDS, metavar="FIELD",
        help="meta fields defining the per-sweep legend/series "
             "(default: model observables — e.g. add scaling_knob to recover "
             "per-knob lines in a multi-knob budget sweep; choices: "
             + ", ".join(CONFIG_IDENTITY_FIELDS) + ")",
    )
    parser.add_argument(
        "--max-epoch", type=int, default=None, metavar="N",
        help="derive best/final accuracy from each run's first N epochs only "
             "(fair comparison across sweeps trained to different lengths)",
    )
    parser.add_argument(
        "--include-file", type=str, action="append", default=[], metavar="PATH",
        help="restrict the comparison to the run names listed in this file "
             "(low_accuracy_runs.txt format: one name per line, # comments ok); "
             "repeatable. Use to compare an ablation vs only its source configs",
    )
    parser.add_argument(
        "--include-run", type=str, action="append", default=[], metavar="NAME",
        help="restrict the comparison to this run name (repeatable; unions with "
             "--include-file)",
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

    include: set[str] | None = None
    if args.include_file or args.include_run:
        include = set(args.include_run)
        for fpath in args.include_file:
            if not Path(fpath).is_file():
                parser.error(f"--include-file does not exist: {fpath}")
            include |= read_run_names_file(Path(fpath))
        print(f"Restricting comparison to {len(include)} included run name(s)")

    rows = load_sweeps(sweep_dirs, labels, max_epoch=args.max_epoch, include=include)
    if not rows:
        print("No runs found across the given sweeps (need per-run history.json).")
        return

    print(f"Comparing {len(sweep_dirs)} sweep(s), {len(rows)} run(s) → {out_dir}")
    write_table(rows, out_dir)

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plotters = (
        partial(plot_acc_vs_params, series_by=args.series_by),
        partial(plot_acc_by_params, series_by=args.series_by),
    )
    for fn in plotters:
        try:
            fn(rows, fig_dir)
        except Exception as e:  # one bad figure must not abort the rest
            name = getattr(fn, "func", fn).__name__
            warnings.warn(f"{name} failed: {type(e).__name__}: {e}", RuntimeWarning)


if __name__ == "__main__":
    main()

"""Aggregate a CV-Quixer hyperparameter sweep into thesis-ready outputs.

Scans a sweep directory (`results/sweeps/<sweep>_<ts>/`) produced by
`experiments/sweep.py`, reads each run's `history.json` (and `config.json` for
the resolved architecture), and emits a comparison table + plots into the sweep
directory:

    summary.csv / summary.md         one row per run (drop-in thesis table)
    figures/acc_vs_params.png        best test acc vs parameter budget,
                                     one line per observable/scaling-knob (± std)
    figures/acc_by_observable.png    grouped bars: acc by observable/knob,
                                     grouped by parameter budget
    figures/acc_by_scaling_knob.png  grouped bars: acc by scaling knob, grouped by
                                     budget (only when ≥2 knobs swept)
    figures/acc_vs_trunc_lambda.png  acc vs truncation penalty weight λ
                                     (only when trunc_lambda is swept)

This reads only JSON — no torch / model rebuild — so it is fast and runs on a
partial or in-progress sweep (re-run any time to refresh).

Usage:
    uv run python experiments/report_sweep.py --sweep-dir results/sweeps/<sweep>_<ts>/
"""

from __future__ import annotations

import argparse
import csv
import json
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Columns for the summary table, in display order.
SUMMARY_COLUMNS = [
    "run_name", "observables", "target_params", "achieved_params",
    "scaling_knob", "num_layers", "trunc_lambda", "seed",
    "best_test_acc", "best_epoch", "final_test_acc",
    "final_train_acc", "n_epochs", "total_runtime_sec", "device",
]


def _load_run(run_dir: Path) -> dict | None:
    """Read one run's history/config into a flat summary row, or None to skip."""
    history_path = run_dir / "history.json"
    if not history_path.is_file():
        return None
    with open(history_path) as f:
        history = json.load(f)

    meta = history.get("meta", {})
    epoch = history.get("epoch", {})
    test_acc = epoch.get("test_acc") or []
    train_acc = epoch.get("train_acc") or []

    best = meta.get("best_test_acc")
    if best is None and test_acc:
        best = max(test_acc)

    return {
        "run_name": run_dir.name,
        "observables": meta.get("observables_name"),
        "target_params": meta.get("target_params"),
        "achieved_params": meta.get("achieved_params") or meta.get("n_params"),
        # Default to the historic knob for runs predating the scaling_knob axis,
        # so old sweeps still group cleanly instead of being dropped as None.
        "scaling_knob": meta.get("scaling_knob") or "cnn_channels_2",
        "num_layers": meta.get("num_layers"),
        "trunc_lambda": meta.get("trunc_lambda"),
        "seed": meta.get("seed"),
        "best_test_acc": best,
        "best_epoch": meta.get("best_epoch"),
        "final_test_acc": test_acc[-1] if test_acc else None,
        "final_train_acc": train_acc[-1] if train_acc else None,
        "n_epochs": len(test_acc),
        "total_runtime_sec": meta.get("total_runtime_sec"),
        "device": meta.get("device"),
    }


def load_sweep(sweep_dir: Path) -> list[dict]:
    """Load every run row under `sweep_dir`, sorted for stable output."""
    rows: list[dict] = []
    for run_dir in sorted(p for p in sweep_dir.iterdir() if p.is_dir()):
        row = _load_run(run_dir)
        if row is not None:
            rows.append(row)
    rows.sort(
        key=lambda r: (
            str(r["observables"]),
            r["target_params"] if r["target_params"] is not None else -1,
            r["seed"] if r["seed"] is not None else -1,
        )
    )
    return rows


def write_table(rows: list[dict], sweep_dir: Path) -> None:
    """Write summary.csv and summary.md."""
    csv_path = sweep_dir / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows({k: r.get(k) for k in SUMMARY_COLUMNS} for r in rows)
    print(f"  ✓ {csv_path}")

    def _fmt(v: object) -> str:
        if isinstance(v, float):
            return f"{v:.4f}"
        return "" if v is None else str(v)

    md_path = sweep_dir / "summary.md"
    with open(md_path, "w") as f:
        f.write("| " + " | ".join(SUMMARY_COLUMNS) + " |\n")
        f.write("|" + "|".join(["---"] * len(SUMMARY_COLUMNS)) + "|\n")
        for r in rows:
            f.write("| " + " | ".join(_fmt(r.get(k)) for k in SUMMARY_COLUMNS) + " |\n")
    print(f"  ✓ {md_path}")


def _aggregate_by(
    rows: list[dict], fields: tuple[str, ...]
) -> dict[tuple, tuple[float, float]]:
    """Mean ± std of best_test_acc over seeds, keyed by the given meta fields.

    Runs missing best_test_acc, or any of the requested key fields, are skipped.
    (Skipping on a None key is what restricts the λ figure to λ-swept runs:
    non-λ runs carry trunc_lambda=None and drop out automatically.)
    """
    buckets: dict[tuple, list[float]] = defaultdict(list)
    for r in rows:
        acc = r.get("best_test_acc")
        if acc is None:
            continue
        key = tuple(r.get(f) for f in fields)
        if any(k is None for k in key):
            continue
        buckets[key].append(float(acc))
    return {k: (float(np.mean(v)), float(np.std(v))) for k, v in buckets.items()}


def plot_acc_vs_params(rows: list[dict], fig_dir: Path) -> None:
    """Best test acc vs parameter budget, one line per (observable, scaling knob).

    Keying by scaling_knob (not just observable) keeps multi-knob sweeps from
    collapsing two knobs into one averaged point per budget.
    """
    agg = _aggregate_by(rows, ("observables", "scaling_knob", "target_params"))
    if not agg:
        print("  (no completed runs with best_test_acc — skipping acc_vs_params)")
        return
    series = sorted({(o, k) for (o, k, _tp) in agg})
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for obs, knob in series:
        pts = sorted(
            (tp, m, s) for (o, k, tp), (m, s) in agg.items() if o == obs and k == knob
        )
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        es = [p[2] for p in pts]
        ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3, label=f"{obs}/{knob}")
    ax.set_xlabel("Target parameter budget")
    ax.set_ylabel("Best test accuracy")
    ax.set_title("Accuracy vs parameter budget by observable / scaling knob")
    ax.grid(alpha=0.3)
    ax.legend(title="observable / knob")
    fig.tight_layout()
    out = fig_dir / "acc_vs_params.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  ✓ {out}")


def plot_acc_by_observable(rows: list[dict], fig_dir: Path) -> None:
    """Grouped bars: best test acc by (observable, scaling knob), grouped by budget."""
    agg = _aggregate_by(rows, ("observables", "scaling_knob", "target_params"))
    if not agg:
        print("  (no completed runs with best_test_acc — skipping acc_by_observable)")
        return
    cats = sorted({(o, k) for (o, k, _tp) in agg})
    budgets = sorted({tp for (_o, _k, tp) in agg})
    x = np.arange(len(cats))
    width = 0.8 / max(len(budgets), 1)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, tp in enumerate(budgets):
        means = [agg.get((o, k, tp), (np.nan, 0.0))[0] for (o, k) in cats]
        errs = [agg.get((o, k, tp), (np.nan, 0.0))[1] for (o, k) in cats]
        ax.bar(x + i * width, means, width, yerr=errs, capsize=3, label=f"{tp:,}")
    ax.set_xticks(x + width * (len(budgets) - 1) / 2)
    ax.set_xticklabels([f"{o}/{k}" for (o, k) in cats], rotation=15, ha="right")
    ax.set_xlabel("Observable preset / scaling knob")
    ax.set_ylabel("Best test accuracy")
    ax.set_title("Accuracy by observable / scaling knob")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(title="target params")
    fig.tight_layout()
    out = fig_dir / "acc_by_observable.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  ✓ {out}")


def plot_acc_by_scaling_knob(rows: list[dict], fig_dir: Path) -> None:
    """Grouped bars: best test acc by scaling knob, grouped by parameter budget.

    The headline figure for a scaling-knob sweep — quantum-width (num_heads) vs
    classical (cnn_channels_2) at each budget. Skipped unless ≥2 knobs are
    present. Intended for single-observable sweeps; observables are averaged over.
    """
    agg = _aggregate_by(rows, ("scaling_knob", "target_params"))
    knobs = sorted({k for (k, _tp) in agg})
    if len(knobs) < 2:
        print("  (need ≥2 scaling_knob values — skipping acc_by_scaling_knob)")
        return
    budgets = sorted({tp for (_k, tp) in agg})
    x = np.arange(len(knobs))
    width = 0.8 / max(len(budgets), 1)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, tp in enumerate(budgets):
        means = [agg.get((k, tp), (np.nan, 0.0))[0] for k in knobs]
        errs = [agg.get((k, tp), (np.nan, 0.0))[1] for k in knobs]
        ax.bar(x + i * width, means, width, yerr=errs, capsize=3, label=f"{tp:,}")
    ax.set_xticks(x + width * (len(budgets) - 1) / 2)
    ax.set_xticklabels(knobs)
    ax.set_xlabel("Scaling knob")
    ax.set_ylabel("Best test accuracy")
    ax.set_title("Accuracy by scaling knob (grouped by parameter budget)")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(title="target params")
    fig.tight_layout()
    out = fig_dir / "acc_by_scaling_knob.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  ✓ {out}")


def plot_acc_vs_trunc_lambda(rows: list[dict], fig_dir: Path) -> None:
    """Best test acc vs truncation penalty weight λ, one line per (obs, knob, budget).

    Only λ-swept runs (trunc_lambda not None) contribute; skipped when none do.
    """
    agg = _aggregate_by(
        rows, ("observables", "scaling_knob", "target_params", "trunc_lambda")
    )
    if not agg:
        print("  (no runs with a trunc_lambda axis — skipping acc_vs_trunc_lambda)")
        return
    series = sorted({(o, k, tp) for (o, k, tp, _l) in agg})
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for obs, knob, tp in series:
        pts = sorted(
            (lam, m, s)
            for (o, k, t, lam), (m, s) in agg.items()
            if (o, k, t) == (obs, knob, tp)
        )
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        es = [p[2] for p in pts]
        ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3, label=f"{obs}/{knob}/{tp:,}")
    ax.set_xlabel("Truncation penalty weight λ (trunc_lambda)")
    ax.set_ylabel("Best test accuracy")
    ax.set_title("Accuracy vs truncation penalty weight")
    ax.grid(alpha=0.3)
    ax.legend(title="observable / knob / params")
    fig.tight_layout()
    out = fig_dir / "acc_vs_trunc_lambda.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  ✓ {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweep-dir", type=str, required=True,
        help="sweep directory written by experiments/sweep.py",
    )
    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)
    if not sweep_dir.is_dir():
        parser.error(f"--sweep-dir does not exist: {sweep_dir}")

    rows = load_sweep(sweep_dir)
    if not rows:
        print(f"No runs found under {sweep_dir} (need per-run history.json).")
        return

    print(f"Aggregating {len(rows)} run(s) under {sweep_dir}")
    write_table(rows, sweep_dir)

    fig_dir = sweep_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for fn in (
        plot_acc_vs_params,
        plot_acc_by_observable,
        plot_acc_by_scaling_knob,
        plot_acc_vs_trunc_lambda,
    ):
        try:
            fn(rows, fig_dir)
        except Exception as e:  # one bad figure must not abort the rest
            warnings.warn(f"{fn.__name__} failed: {type(e).__name__}: {e}", RuntimeWarning)


if __name__ == "__main__":
    main()

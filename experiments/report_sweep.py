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

The cross-run aggregation reads only JSON (no torch / model rebuild), so it is
fast and runs on a partial or in-progress sweep. By default it then also renders
the full `report_diagnostics.py` figure suite into each run's own `figures/` dir
(one subprocess per run; report_diagnostics' default path is npz/JSON-based, so
heavy torch imports stay deferred). Pass --skip-per-run-figures for the fast
cross-run-only pass.

Usage:
    uv run python experiments/report_sweep.py --sweep-dir results/sweeps/<sweep>_<ts>/
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Repo root (experiments/ -> repo root); used to invoke report_diagnostics.py.
REPO_ROOT = Path(__file__).resolve().parent.parent
REPORT_DIAGNOSTICS = "experiments/report_diagnostics.py"

# Resolved architecture knobs recorded in history["meta"] by full_experiment.py.
# Surfaced per-run in the table and used as the candidate axes for the
# acc-vs-hyperparam figures (any that vary across runs). num_layers is loaded
# separately (it predates this list) but is included here as a figure candidate.
ARCH_META_FIELDS = [
    "num_modes", "num_heads", "cutoff_dim", "poly_degree",
    "cnn_channels_1", "cnn_channels_2", "cnn_kernel_size", "decoder_hidden_dim",
    "cnn_num_conv_layers", "hypernet_num_linear_layers", "decoder_num_layers",
    "decoder_hidden_mult", "cvqnn_num_layers",
    # Stacked-model axes (ADR-0002); absent on older runs → skipped.
    "num_seq2seq_blocks", "pooling", "block_residual",
]

# Columns for the summary table, in display order.
SUMMARY_COLUMNS = [
    "run_name", "model", "observables", "target_params", "achieved_params",
    "scaling_knob", "num_layers", "trunc_lambda", "seed",
    *ARCH_META_FIELDS,
    "best_test_acc", "best_epoch", "final_test_acc",
    "final_train_acc", "n_epochs", "total_runtime_sec", "device",
]


def _resolve_model(run_dir: Path, meta: dict) -> str:
    """Model variant for a run: meta → config.json fallback → "quantum".

    `meta["model"]` only exists on runs trained after that key was added; older
    runs (e.g. the finished quantum sweeps) carry it only in config.json, so fall
    back to the saved config and finally to the historic default "quantum".
    """
    model = meta.get("model")
    if model:
        return str(model)
    config_path = run_dir / "config.json"
    if config_path.is_file():
        try:
            with open(config_path) as f:
                cfg_model = json.load(f).get("model")
            if cfg_model:
                return str(cfg_model)
        except (json.JSONDecodeError, OSError):
            pass
    return "quantum"


def _load_run(run_dir: Path) -> dict | None:
    """Read one run's history/config into a flat summary row, or None to skip."""
    history_path = run_dir / "history.json"
    if not history_path.is_file():
        # A dir with config.json but no history.json is a run that started but
        # never finished an epoch (e.g. OOM / wall-time kill). Warn so it isn't
        # silently dropped from the table and figures.
        if (run_dir / "config.json").is_file():
            warnings.warn(
                f"skipping {run_dir.name}: no history.json — run did not complete "
                "an epoch (check its logs/slurm .err; re-run to populate).",
                RuntimeWarning, stacklevel=2,
            )
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

    row = {
        "run_name": run_dir.name,
        "model": _resolve_model(run_dir, meta),
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
    # Resolved architecture knobs (present for runs from the manual-sweep-aware
    # full_experiment.py; None for older runs — safely omitted from figures).
    for field in ARCH_META_FIELDS:
        row[field] = meta.get(field)
    return row


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


def _mean_value_by(
    rows: list[dict], fields: tuple[str, ...], value_field: str
) -> dict[tuple, float]:
    """Mean of ``value_field`` over rows, keyed by the given meta fields.

    Uses the same keying/skip rules as ``_aggregate_by`` (rows missing
    best_test_acc or any key field drop out), so the returned map lines up
    one-to-one with ``_aggregate_by``'s keys. Used to place points at the mean
    *achieved* parameter count of each seed-group rather than the target budget.
    """
    buckets: dict[tuple, list[float]] = defaultdict(list)
    for r in rows:
        if r.get("best_test_acc") is None:
            continue
        key = tuple(r.get(f) for f in fields)
        if any(k is None for k in key):
            continue
        val = r.get(value_field)
        if val is None:
            continue
        buckets[key].append(float(val))
    return {k: float(np.mean(v)) for k, v in buckets.items()}


def plot_acc_vs_params(rows: list[dict], fig_dir: Path) -> None:
    """Best test acc vs *achieved* parameter count, one line per (observable, knob).

    Keying by scaling_knob (not just observable) keeps multi-knob sweeps from
    collapsing two knobs into one averaged point per budget. Accuracy is still
    seed-averaged within each (observable, knob, target budget) group, but each
    point's x is the group's mean *achieved* parameter count — so two knobs that
    overshoot the same budget differently (e.g. 13,530 vs 12,722 at a 12,800
    target) are drawn at their true sizes rather than stacked at one x.
    """
    agg = _aggregate_by(rows, ("observables", "scaling_knob", "target_params"))
    if not agg:
        print("  (no completed runs with best_test_acc — skipping acc_vs_params)")
        return
    x_by_key = _mean_value_by(
        rows, ("observables", "scaling_knob", "target_params"), "achieved_params"
    )
    series = sorted({(o, k) for (o, k, _tp) in agg})
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for obs, knob in series:
        # Sort points by achieved param count (x) for a sensible line order.
        pts = sorted(
            (x_by_key[(o, k, tp)], m, s)
            for (o, k, tp), (m, s) in agg.items()
            if o == obs and k == knob and (o, k, tp) in x_by_key
        )
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        es = [p[2] for p in pts]
        ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3, label=f"{obs}/{knob}")
    ax.set_xlabel("Achieved parameter count")
    ax.set_ylabel("Best test accuracy")
    ax.set_title("Accuracy vs achieved parameter count by observable / scaling knob")
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
    # Achieved param count per group, annotated on each bar (the budget grouping
    # is categorical, but a group's two knobs can have different achieved sizes).
    achieved = _mean_value_by(
        rows, ("observables", "scaling_knob", "target_params"), "achieved_params"
    )
    cats = sorted({(o, k) for (o, k, _tp) in agg})
    budgets = sorted({tp for (_o, _k, tp) in agg})
    x = np.arange(len(cats))
    width = 0.8 / max(len(budgets), 1)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, tp in enumerate(budgets):
        means = [agg.get((o, k, tp), (np.nan, 0.0))[0] for (o, k) in cats]
        errs = [agg.get((o, k, tp), (np.nan, 0.0))[1] for (o, k) in cats]
        bars = ax.bar(x + i * width, means, width, yerr=errs, capsize=3, label=f"{tp:,}")
        for (o, k), bar in zip(cats, bars):
            n = achieved.get((o, k, tp))
            if n is not None and not np.isnan(bar.get_height()):
                ax.annotate(
                    f"{int(round(n)):,}",
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center", va="bottom", fontsize=6, rotation=90,
                )
    ax.set_xticks(x + width * (len(budgets) - 1) / 2)
    ax.set_xticklabels([f"{o}/{k}" for (o, k) in cats], rotation=15, ha="right")
    ax.set_xlabel("Observable preset / scaling knob")
    ax.set_ylabel("Best test accuracy")
    ax.set_title("Accuracy by observable / scaling knob (bar labels = achieved params)")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(title="target param budget")
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
    # Achieved param count per (knob, budget), averaged over observables, for bar labels.
    achieved = _mean_value_by(rows, ("scaling_knob", "target_params"), "achieved_params")
    budgets = sorted({tp for (_k, tp) in agg})
    x = np.arange(len(knobs))
    width = 0.8 / max(len(budgets), 1)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, tp in enumerate(budgets):
        means = [agg.get((k, tp), (np.nan, 0.0))[0] for k in knobs]
        errs = [agg.get((k, tp), (np.nan, 0.0))[1] for k in knobs]
        bars = ax.bar(x + i * width, means, width, yerr=errs, capsize=3, label=f"{tp:,}")
        for k, bar in zip(knobs, bars):
            n = achieved.get((k, tp))
            if n is not None and not np.isnan(bar.get_height()):
                ax.annotate(
                    f"{int(round(n)):,}",
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center", va="bottom", fontsize=6, rotation=90,
                )
    ax.set_xticks(x + width * (len(budgets) - 1) / 2)
    ax.set_xticklabels(knobs)
    ax.set_xlabel("Scaling knob")
    ax.set_ylabel("Best test accuracy")
    ax.set_title("Accuracy by scaling knob (bar labels = achieved params)")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(title="target param budget")
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


def plot_acc_vs_hyperparam(rows: list[dict], fig_dir: Path) -> None:
    """One ``acc_vs_<field>.png`` per architecture field that varies across runs.

    The figure for manual-hyperparameter sweeps: for every candidate field
    (``ARCH_META_FIELDS`` + ``num_layers``) that takes ≥2 distinct non-None values,
    plot seed-averaged best test accuracy (± std) against that field's value, one
    line per (observable, scaling_knob). Fields constant across the sweep, or
    absent (older runs), are skipped. When several arch axes vary at once, the
    others are averaged into each point (widening the ± band) — read alongside the
    per-run ``summary.csv`` for the exact grid.
    """
    candidates = ARCH_META_FIELDS + ["num_layers"]
    for field in candidates:
        distinct = {
            r.get(field) for r in rows
            if r.get(field) is not None and r.get("best_test_acc") is not None
        }
        if len(distinct) < 2:
            continue
        agg = _aggregate_by(rows, ("observables", "scaling_knob", field))
        if not agg:
            continue
        series = sorted({(o, k) for (o, k, _v) in agg})
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for obs, knob in series:
            pts = sorted(
                (v, m, s) for (o, k, v), (m, s) in agg.items() if (o, k) == (obs, knob)
            )
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            es = [p[2] for p in pts]
            ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3, label=f"{obs}/{knob}")
        ax.set_xlabel(field)
        ax.set_ylabel("Best test accuracy")
        ax.set_title(f"Accuracy vs {field}")
        ax.grid(alpha=0.3)
        ax.legend(title="observable / knob")
        fig.tight_layout()
        out = fig_dir / f"acc_vs_{field}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  ✓ {out}")


def render_per_run_figures(sweep_dir: Path, rows: list[dict]) -> None:
    """Render the full report_diagnostics figure suite for every run in the sweep.

    One `report_diagnostics.py --run-dir <run>` subprocess per run (its default
    path is npz/JSON-based, so heavy torch imports stay deferred). Each is wrapped
    so one failure doesn't abort the rest.
    """
    print(f"\nRendering per-run figure suites for {len(rows)} run(s)...")
    for r in rows:
        run_dir = sweep_dir / r["run_name"]
        if not run_dir.is_dir():
            warnings.warn(f"run dir missing, skipping: {run_dir}", RuntimeWarning)
            continue
        cmd = [sys.executable, "-u", REPORT_DIAGNOSTICS, "--run-dir", str(run_dir)]
        try:
            subprocess.run(cmd, cwd=REPO_ROOT, check=True)
        except subprocess.CalledProcessError as e:
            warnings.warn(
                f"report_diagnostics failed for {r['run_name']}: exit {e.returncode}",
                RuntimeWarning,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweep-dir", type=str, required=True,
        help="sweep directory written by experiments/sweep.py",
    )
    parser.add_argument(
        "--skip-per-run-figures", action="store_true",
        help="skip rendering report_diagnostics for each run "
             "(cross-run figures + tables only — the fast JSON-only pass)",
    )
    args = parser.parse_args()

    # Line-buffer stdout so progress streams live even when piped / redirected /
    # captured (non-TTY block-buffers by default, making the run look hung).
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

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
        plot_acc_vs_hyperparam,
    ):
        try:
            fn(rows, fig_dir)
        except Exception as e:  # one bad figure must not abort the rest
            warnings.warn(f"{fn.__name__} failed: {type(e).__name__}: {e}", RuntimeWarning)

    if not args.skip_per_run_figures:
        render_per_run_figures(sweep_dir, rows)


if __name__ == "__main__":
    main()

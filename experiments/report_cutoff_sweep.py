"""Aggregate a whole-sweep cutoff-dim sweep into cross-run tables + figures.

Consumes the per-run `eval_cutoff_sweep.py` outputs produced by
`experiments/eval_cutoff_sweep_all.py` (each at `<run>/eval/<eval_name>/`) and
emits cross-run artefacts into the sweep directory:

    cutoff_summary.csv / .md              one row per (run, split, cutoff)
    figures/cutoff/acc_vs_cutoff_all.png        test acc vs D, one line per run
    figures/cutoff/trunc_loss_vs_cutoff_all.png trunc loss vs D, one line per run
    figures/cutoff/acc_recovery_vs_params.png   Δacc(maxD−trainD) vs params, by knob
    figures/cutoff/acc_vs_params_by_cutoff.png  acc vs achieved params, one line per D

It reads only JSON (no torch / model rebuild), so it is fast and runs on a partial
or in-progress whole-sweep eval. By default it then also renders the **full**
report_diagnostics figure suite into every per-cutoff `D{NN}/` dir (one
subprocess per cutoff per run); pass --skip-per-run-figures for the fast
cross-run-only pass.

Usage:
    uv run python experiments/report_cutoff_sweep.py --sweep-dir results/sweeps/<sweep>_<ts>/
"""

from __future__ import annotations

import argparse
import csv
import glob
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

# Sibling script (experiments/ is sys.path[0] when run as a script).
from report_sweep import _load_run

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORT_DIAGNOSTICS = "experiments/report_diagnostics.py"

SUMMARY_COLUMNS = [
    "run_name", "scaling_knob", "observables", "target_params", "achieved_params",
    "training_cutoff", "split", "cutoff_dim", "acc", "ce_loss", "trunc_loss",
    "mean_state_norm", "mean_photon", "n_samples", "elapsed_sec",
]


def resolve_eval_name(sweep_dir: Path, eval_name: str | None) -> str:
    """Pick the eval folder name shared across runs.

    Priority: explicit --eval-name, then the manifest's `eval_name`, then the
    lexicographically-latest `eval/cutoff_sweep_*` present under any run (the
    name carries a sortable timestamp).
    """
    if eval_name is not None:
        return eval_name
    manifest = sweep_dir / "cutoff_sweep_manifest.json"
    if manifest.is_file():
        with open(manifest) as f:
            name = json.load(f).get("eval_name")
        if name:
            return name
    candidates = {
        Path(p).name
        for p in glob.glob(str(sweep_dir / "*" / "eval" / "cutoff_sweep_*"))
    }
    if not candidates:
        raise FileNotFoundError(
            f"No cutoff_sweep_manifest.json and no eval/cutoff_sweep_* dirs under "
            f"{sweep_dir}. Run experiments/eval_cutoff_sweep_all.py first."
        )
    return sorted(candidates)[-1]


def load_rows(sweep_dir: Path, eval_name: str) -> tuple[list[dict], list[str]]:
    """Load one row per (run, split, cutoff) and collect per-cutoff sub-run dirs.

    Returns (rows, sub_run_dirs). Runs without an eval `results.json` for
    `eval_name` are skipped with a warning.
    """
    rows: list[dict] = []
    sub_run_dirs: list[str] = []
    for run_dir in sorted(p for p in sweep_dir.iterdir() if p.is_dir()):
        results_path = run_dir / "eval" / eval_name / "results.json"
        if not results_path.is_file():
            continue
        with open(results_path) as f:
            res = json.load(f)
        coords = _load_run(run_dir) or {}
        for pc in res.get("per_cutoff", []):
            rows.append({
                "run_name": run_dir.name,
                "scaling_knob": coords.get("scaling_knob"),
                "observables": coords.get("observables"),
                "target_params": coords.get("target_params"),
                "achieved_params": coords.get("achieved_params"),
                "training_cutoff": res.get("training_cutoff"),
                "split": pc.get("split"),
                "cutoff_dim": pc.get("cutoff_dim"),
                "acc": pc.get("acc"),
                "ce_loss": pc.get("ce_loss"),
                "trunc_loss": pc.get("trunc_loss"),
                "mean_state_norm": pc.get("mean_state_norm"),
                "mean_photon": pc.get("mean_photon"),
                "n_samples": pc.get("n_samples"),
                "elapsed_sec": pc.get("elapsed_sec"),
            })
        sub_run_dirs.extend(res.get("sub_run_dirs", []))
    if not rows:
        warnings.warn(
            f"No results.json found under {sweep_dir}/*/eval/{eval_name}/ — "
            "has the whole-sweep eval finished?",
            RuntimeWarning, stacklevel=2,
        )
    return rows, sub_run_dirs


def write_table(rows: list[dict], sweep_dir: Path) -> None:
    """Write cutoff_summary.csv and cutoff_summary.md (sorted for stable output)."""
    rows = sorted(rows, key=lambda r: (
        str(r["scaling_knob"]), r["achieved_params"] or -1,
        str(r["split"]), r["cutoff_dim"] or -1,
    ))
    csv_path = sweep_dir / "cutoff_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows({k: r.get(k) for k in SUMMARY_COLUMNS} for r in rows)
    print(f"  ✓ {csv_path}")

    def _fmt(v: object) -> str:
        if isinstance(v, float):
            return f"{v:.4f}"
        return "" if v is None else str(v)

    md_path = sweep_dir / "cutoff_summary.md"
    with open(md_path, "w") as f:
        f.write("| " + " | ".join(SUMMARY_COLUMNS) + " |\n")
        f.write("|" + "|".join(["---"] * len(SUMMARY_COLUMNS)) + "|\n")
        for r in rows:
            f.write("| " + " | ".join(_fmt(r.get(k)) for k in SUMMARY_COLUMNS) + " |\n")
    print(f"  ✓ {md_path}")


def _test_rows(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r["split"] == "test"]


def _run_label(run: list[dict]) -> str:
    r = run[0]
    knob = r["scaling_knob"]
    ap = r["achieved_params"]
    return f"{knob}/{ap:,}" if ap is not None else str(knob)


def _by_run(rows: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r["run_name"]].append(r)
    return groups


def plot_metric_vs_cutoff(rows: list[dict], fig_dir: Path, metric: str,
                          ylabel: str, title: str, fname: str) -> None:
    """One line per run: `metric` (test split) vs cutoff_dim, training-D marked."""
    test = [r for r in _test_rows(rows) if r.get(metric) is not None]
    if not test:
        print(f"  (no test rows with {metric} — skipping {fname})")
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    training_cutoffs = set()
    # Order runs by achieved params for a sensible legend.
    runs = sorted(_by_run(test).items(),
                  key=lambda kv: kv[1][0]["achieved_params"] or -1)
    for _name, run in runs:
        pts = sorted((r["cutoff_dim"], r[metric]) for r in run)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, marker="o", lw=1.8, label=_run_label(run))
        if run[0]["training_cutoff"] is not None:
            training_cutoffs.add(run[0]["training_cutoff"])
    for tc in sorted(training_cutoffs):
        ax.axvline(tc, color="gray", ls="--", alpha=0.5)
    ax.set_xlabel("cutoff_dim (D)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}  (dashed = training D)")
    ax.grid(alpha=0.3)
    ax.legend(title="knob / achieved params", fontsize=8)
    fig.tight_layout()
    out = fig_dir / fname
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  ✓ {out}")


def plot_acc_recovery_vs_params(rows: list[dict], fig_dir: Path) -> None:
    """Δacc = acc(maxD) − acc(trainD) vs achieved params, one line per knob."""
    test = _test_rows(rows)
    pts_by_knob: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for _name, run in _by_run(test).items():
        accs = {r["cutoff_dim"]: r["acc"] for r in run
                if r["cutoff_dim"] is not None and r["acc"] is not None}
        if not accs:
            continue
        train_d = run[0]["training_cutoff"]
        base_d = train_d if train_d in accs else min(accs)
        max_d = max(accs)
        delta = accs[max_d] - accs[base_d]
        ap = run[0]["achieved_params"]
        if ap is not None:
            pts_by_knob[run[0]["scaling_knob"]].append((ap, delta))
    if not pts_by_knob:
        print("  (no data — skipping acc_recovery_vs_params)")
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for knob, pts in sorted(pts_by_knob.items()):
        pts = sorted(pts)
        ax.plot([p[0] for p in pts], [p[1] for p in pts],
                marker="o", lw=1.8, label=knob)
    ax.axhline(0.0, color="gray", ls="--", alpha=0.5)
    ax.set_xlabel("Achieved parameter count")
    ax.set_ylabel("Δ test accuracy (max D − training D)")
    ax.set_title("Accuracy recovery from relaxing the Fock cutoff")
    ax.grid(alpha=0.3)
    ax.legend(title="scaling knob")
    fig.tight_layout()
    out = fig_dir / "acc_recovery_vs_params.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  ✓ {out}")


def plot_acc_vs_params_by_cutoff(rows: list[dict], fig_dir: Path) -> None:
    """Test acc vs achieved params, one line per cutoff_dim."""
    test = [r for r in _test_rows(rows)
            if r["achieved_params"] is not None and r["acc"] is not None]
    if not test:
        print("  (no data — skipping acc_vs_params_by_cutoff)")
        return
    by_cut: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for r in test:
        by_cut[r["cutoff_dim"]].append((r["achieved_params"], r["acc"]))
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for D in sorted(by_cut):
        pts = sorted(by_cut[D])
        ax.plot([p[0] for p in pts], [p[1] for p in pts],
                marker="o", lw=1.8, label=f"D={D}")
    ax.set_xlabel("Achieved parameter count")
    ax.set_ylabel("Best test accuracy")
    ax.set_title("Accuracy vs params, by Fock cutoff")
    ax.grid(alpha=0.3)
    ax.legend(title="cutoff_dim")
    fig.tight_layout()
    out = fig_dir / "acc_vs_params_by_cutoff.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  ✓ {out}")


def render_per_cutoff_figures(sub_run_dirs: list[str]) -> None:
    """Render the full report_diagnostics figure suite for each D{NN}/ dir."""
    print(f"\nRendering per-cutoff figure suites for {len(sub_run_dirs)} cutoff dir(s)...")
    for sub in sub_run_dirs:
        if not Path(sub).is_dir():
            warnings.warn(f"sub-run dir missing, skipping: {sub}", RuntimeWarning)
            continue
        cmd = [sys.executable, REPORT_DIAGNOSTICS, "--run-dir", sub]
        try:
            subprocess.run(cmd, cwd=REPO_ROOT, check=True)
        except subprocess.CalledProcessError as e:
            warnings.warn(
                f"report_diagnostics failed for {sub}: exit {e.returncode}",
                RuntimeWarning,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-dir", type=Path, required=True,
                        help="sweep directory previously eval'd by "
                             "experiments/eval_cutoff_sweep_all.py")
    parser.add_argument("--eval-name", type=str, default=None,
                        help="eval folder name (default: from manifest, else "
                             "the latest eval/cutoff_sweep_* across runs)")
    parser.add_argument("--skip-per-run-figures", action="store_true",
                        help="skip rendering the full report_diagnostics suite "
                             "per cutoff (cross-run figures only)")
    args = parser.parse_args()

    if not args.sweep_dir.is_dir():
        parser.error(f"--sweep-dir does not exist: {args.sweep_dir}")

    eval_name = resolve_eval_name(args.sweep_dir, args.eval_name)
    print(f"Aggregating cutoff sweep '{eval_name}' under {args.sweep_dir}")

    rows, sub_run_dirs = load_rows(args.sweep_dir, eval_name)
    if not rows:
        return

    write_table(rows, args.sweep_dir)

    fig_dir = args.sweep_dir / "figures" / "cutoff"
    fig_dir.mkdir(parents=True, exist_ok=True)
    figures = [
        lambda: plot_metric_vs_cutoff(
            rows, fig_dir, "acc", "Test accuracy",
            "Accuracy vs cutoff_dim", "acc_vs_cutoff_all.png"),
        lambda: plot_metric_vs_cutoff(
            rows, fig_dir, "trunc_loss", "Mean truncation loss",
            "Truncation loss vs cutoff_dim", "trunc_loss_vs_cutoff_all.png"),
        lambda: plot_acc_recovery_vs_params(rows, fig_dir),
        lambda: plot_acc_vs_params_by_cutoff(rows, fig_dir),
    ]
    for fn in figures:
        try:
            fn()
        except Exception as e:  # one bad figure must not abort the rest
            warnings.warn(f"figure failed: {type(e).__name__}: {e}", RuntimeWarning)

    if not args.skip_per_run_figures:
        render_per_cutoff_figures(sub_run_dirs)


if __name__ == "__main__":
    main()

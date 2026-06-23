"""Select the representative runs to extend from 10 -> 25 epochs (thesis record).

Motivation
----------
At epoch 10 the high_epoch_* sweeps were *not* saturated: across the healthy
runs the test-accuracy slope over the final 5 epochs was positive in ~all runs
(quantum/stacked 100%, shared ~85%), averaging +0.002..+0.004 per epoch. Rather
than re-extend all ~120 runs, a representative subset per model was topped up to
25 epochs to locate the saturation point cheaply.

Candidate pool
--------------
Every run directory under each ``high_epoch_<model>_*`` sweep whose
``history.json`` records a full ``test_acc`` series of >= MIN_EPOCHS epochs,
EXCLUDING re-roll directories (``reroll__`` prefix). Collapsed runs are not
filtered explicitly: a uniform-predictor collapse sits at acc 0.10 for all
epochs, so it can never enter the top-2 of any selection metric.

Selection metrics (per run, from the test-set accuracy series ``a`` of length E)
--------------------------------------------------------------------------------
  best  = max_e a[e]                         peak accuracy over training
  last3 = mean(a[E-3 : E])                    denoised endpoint (final 3 epochs)
  slope = OLS slope of a vs epoch over the
          last WINDOW epochs (a[E-WINDOW:E]),  "is it still improving?"
          per-epoch units

Indices are 0-based; with E=10 and WINDOW=5, ``slope`` is fit over epochs 6..10
(1-based) and ``last3`` averages epochs 8..10 (1-based).

Selection rule
--------------
Per model, take the UNION of the top ``TOP_K`` runs by each of the three metrics,
de-duplicated. A run may be picked by more than one metric (its tags record
which). This deliberately spans three different "good run" definitions:
peak (best), robust endpoint (last3), and still-climbing (slope).

Usage
-----
    uv run python experiments/select_extension_runs.py            # print audit
    uv run python experiments/select_extension_runs.py --write    # + rewrite the txt

This script is read-only by default and is the provenance record for
``results/extended_runs_25ep.txt``.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

SWEEPS = {
    "quantum": "results/sweeps/high_epoch_quantum_2026-06-15_03-47-33",
    "shared": "results/sweeps/high_epoch_shared_2026-06-15_03-52-41",
    "stacked": "results/sweeps/high_epoch_stacked_2026-06-15_04-02-35",
}
MIN_EPOCHS = 10      # candidate pool: completed runs only
WINDOW = 5           # slope fit window (last WINDOW epochs)
TOP_K = 2            # top-K per metric, unioned
METRICS = ("best", "last3", "slope")

# The default sweep GPU (a100-40) tops out near 40 GB; a run whose measured peak
# exceeds that needs a bigger card. This reproduces the hand-assigned GPU map in
# results/extended_runs_25ep.txt (stacked nm3 at ~86-89 GB -> h100-96, everything
# else -> a100-40). The emitted GPU column is the contract experiments/
# build_pe_ablation.py reads (parse_source_runs_file expects gpu in column 3).
A100_40_CAPACITY_MB = 40_000


def gpu_for_peak_mb(peak_mb: float) -> str:
    """Target GPU for a run given its measured peak memory (see A100_40_CAPACITY_MB)."""
    return "h100-96" if peak_mb > A100_40_CAPACITY_MB else "a100-40"


def run_metrics(history_path: str) -> dict | None:
    """Compute (best, last3, slope, peak) for one run, or None if not eligible."""
    with open(history_path) as f:
        h = json.load(f)
    ep = h.get("epoch", {})
    acc = ep.get("test_acc", [])
    if len(acc) < MIN_EPOCHS:
        return None
    a = np.asarray(acc, dtype=float)
    e = np.arange(len(a) - WINDOW, len(a))           # last WINDOW epoch indices
    slope = float(np.polyfit(e, a[-WINDOW:], 1)[0])
    pm = ep.get("peak_mem_mb", [])
    return {
        "best": float(a.max()),
        "last3": float(a[-3:].mean()),
        "slope": slope,
        "peak_mb": float(max(pm)) if pm else 0.0,
        "n_epochs": len(a),
    }


def collect(sweep_dir: str) -> dict[str, dict]:
    """All eligible (non-reroll, >=MIN_EPOCHS) runs in a sweep -> their metrics."""
    out = {}
    for hp in sorted(glob.glob(f"{sweep_dir}/*/history.json")):
        name = os.path.basename(os.path.dirname(hp))
        if name.startswith("reroll__"):
            continue
        m = run_metrics(hp)
        if m is not None:
            out[name] = m
    return out


def select(runs: dict[str, dict]) -> dict[str, list[str]]:
    """run_name -> sorted list of metric tags that selected it (union of top-K)."""
    picked: dict[str, list[str]] = {}
    for metric in METRICS:
        ranked = sorted(runs, key=lambda n: -runs[n][metric])
        for name in ranked[:TOP_K]:
            picked.setdefault(name, []).append(metric)
    return {n: sorted(tags) for n, tags in picked.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true",
                    help="rewrite results/extended_runs_25ep.txt from this selection")
    args = ap.parse_args()

    total = 0
    selected_lines = []
    for model, sweep in SWEEPS.items():
        runs = collect(sweep)
        picked = select(runs)
        total += len(picked)
        print(f"\n{'='*92}\n{model}  ({sweep})  —  {len(runs)} eligible runs, "
              f"{len(picked)} selected\n{'='*92}")
        for metric in METRICS:
            ranked = sorted(runs, key=lambda n: -runs[n][metric])
            cut = runs[ranked[TOP_K - 1]][metric]
            print(f"  top-{TOP_K} by {metric:<5} (cutoff {cut:.4f}):")
            for i, name in enumerate(ranked[:TOP_K + 3]):
                star = "*" if i < TOP_K else " "
                short = name.split("__seed42__")[-1]
                print(f"    {star} {metric}={runs[name][metric]:+.4f}  {short}")
        print(f"\n  --> SELECTED ({len(picked)}):")
        print(f"    {'run (short)':<46} {'why':<18} {'best':>6} {'last3':>6} "
              f"{'slope':>8} {'peak_MB':>8}")
        for name in sorted(picked, key=lambda n: -runs[n]["best"]):
            m = runs[name]
            short = name.split("__seed42__")[-1]
            print(f"    {short:<46} {'+'.join(picked[name]):<18} "
                  f"{m['best']:6.3f} {m['last3']:6.3f} {m['slope']:+8.4f} "
                  f"{m['peak_mb']:8.0f}")
            selected_lines.append((model, sweep, name, picked[name], m))

    print(f"\nTOTAL selected across models: {total}")

    if args.write:
        path = "results/extended_runs_25ep.txt"
        with open(path, "w") as f:
            f.write("# Auto-generated by experiments/select_extension_runs.py\n")
            # Column 3 (gpu) is the contract build_pe_ablation.py reads; keep it third.
            f.write("# run_name  sweep_dir  gpu  peak_MB  why  best  last3  slope\n")
            for model, sweep, name, tags, m in selected_lines:
                f.write(f"{name}  {sweep}  {gpu_for_peak_mb(m['peak_mb'])}  "
                        f"{m['peak_mb']:.0f}  {'+'.join(tags)}  "
                        f"{m['best']:.3f}  {m['last3']:.3f}  {m['slope']:+.4f}\n")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()

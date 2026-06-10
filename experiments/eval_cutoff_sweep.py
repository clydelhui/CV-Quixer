"""Cutoff-dim sweep evaluation for a trained CV-Quixer checkpoint.

Loads a checkpoint trained at one `cutoff_dim` D and re-evaluates the same
weights at a list of larger D values to study how Fock truncation affects
test accuracy and truncation loss. Safe because every trainable parameter is
D-independent (gate matrices and observables are rebuilt from learnable
scalars at each forward pass).

The sweep is compatible with the artefact contract that
`experiments/full_experiment.py` writes and that
`experiments/report_diagnostics.py` consumes:

- The run's saved `subset_indices.npz` is reused (mandatory) so the test and
  diagnostic populations exactly match the trained run — "accuracy at cutoff
  D" is then apples-to-apples with the run's reported accuracy.
- Per cutoff, the sweep synthesises a self-contained single-epoch run dir at
  `<run_dir>/eval/cutoff_sweep_<ts>/D{NN}/` that
  `report_diagnostics.py --run-dir <...>/D{NN}` consumes directly to render
  the full figure suite at that cutoff.

D-invariance note: `lcu_coefficients_heatmap`,
`polynomial_coefficients_trajectory`, and `hypernet_gate_param_histograms`
are weight/input-only and identical across all `D*/`; the D-varying figures
are `photon_number_per_mode`, `state_norm_histogram`, and the
prediction-derived ones (truncation shifts the quantum readouts).

Run:
    uv run python experiments/eval_cutoff_sweep.py \\
        --checkpoint results/runs/full_fashionmnist_*/checkpoints/final_model.pt \\
        [--cutoffs 6 8 10 12] \\
        [--batch-size 64] \\
        [--eval-splits test train] \\        # default: just test
        [--test-fraction 0.5] \\             # override: re-subset; warns
        [--test-limit 64] \\                 # override: re-subset; warns
        [--train-fraction 0.1] \\            # override: re-subset; warns
        [--train-limit 64] \\                # override: re-subset; warns
        [--subset-seed 42] \\
        [--dtype complex64|complex128] \\
        [--output-name <name>]

Outputs go to <run_dir>/eval/cutoff_sweep_<timestamp>/.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import os
import sys
import time
import warnings
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from cv_quixer.config.schema import ExperimentConfig
from cv_quixer.config.utils import experiment_config_from_dict
from cv_quixer.data.mnist import PatchedDataset
from cv_quixer.evaluation.diagnostics import (
    evaluate,
    load_subset_indices,
    new_history,
    quantum_diagnostics,
    save_test_images_once,
    snapshot_coefficients,
)
from cv_quixer.models import build_model
from cv_quixer.utils import print_parameter_table


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True,
                    help="path to checkpoint .pt file (e.g. final_model.pt)")
parser.add_argument("--cutoffs", type=int, nargs="+", default=[6, 8, 10, 12],
                    help="list of cutoff_dim values to evaluate at")
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--eval-splits", type=str, nargs="+",
                    default=["test"], choices=["train", "test"],
                    help="which dataset splits to evaluate on; supply both to "
                         "compare train vs test at each cutoff (default: test only)")
parser.add_argument("--test-fraction", type=float, default=None,
                    help="OVERRIDE: re-subset the test set (0 < x <= 1). "
                         "Default path reuses the run's saved test subset. "
                         "Mutually exclusive with --test-limit.")
parser.add_argument("--test-limit", type=int, default=None,
                    help="OVERRIDE: first N samples from the reused test "
                         "subset (deterministic). Mutually exclusive with "
                         "--test-fraction.")
parser.add_argument("--train-fraction", type=float, default=None,
                    help="OVERRIDE: re-subset the train set (0 < x <= 1). "
                         "Default path reuses the run's saved train subset. "
                         "Mutually exclusive with --train-limit.")
parser.add_argument("--train-limit", type=int, default=None,
                    help="OVERRIDE: first N samples from the reused train "
                         "subset (deterministic). Mutually exclusive with "
                         "--train-fraction.")
parser.add_argument("--subset-seed", type=int, default=42,
                    help="seed for the --{train,test}-fraction override "
                         "random subsets (ignored in the default reuse path)")
parser.add_argument("--dtype", type=str, default=None,
                    choices=["complex64", "complex128"],
                    help="override the quantum dtype (default: match training)")
parser.add_argument("--output-name", type=str, default=None,
                    help="override eval folder name (default: cutoff_sweep_<timestamp>)")
args = parser.parse_args()

# de-dup while preserving order (argparse allows repeated values with nargs="+")
args.eval_splits = list(dict.fromkeys(args.eval_splits))

if args.test_fraction is not None and args.test_limit is not None:
    parser.error("--test-fraction and --test-limit are mutually exclusive")
if args.test_fraction is not None and not (0.0 < args.test_fraction <= 1.0):
    parser.error("--test-fraction must be in (0, 1]")
if args.train_fraction is not None and args.train_limit is not None:
    parser.error("--train-fraction and --train-limit are mutually exclusive")
if args.train_fraction is not None and not (0.0 < args.train_fraction <= 1.0):
    parser.error("--train-fraction must be in (0, 1]")


# ---------------------------------------------------------------------------
# Resolve run directory and load resolved config
# ---------------------------------------------------------------------------

ckpt_path = Path(args.checkpoint).resolve()
if not ckpt_path.is_file():
    raise FileNotFoundError(f"--checkpoint path does not exist: {ckpt_path}")

# checkpoint lives at <run_dir>/checkpoints/<name>.pt
run_dir = ckpt_path.parent.parent
config_json = run_dir / "config.json"
if not config_json.is_file():
    raise FileNotFoundError(
        f"Could not find config.json at {config_json}. "
        "The eval script needs the resolved training config "
        "(with auto-scaled cnn_channels_2) to rebuild the model exactly."
    )

with open(config_json) as f:
    cfg_raw = json.load(f)

config: ExperimentConfig = experiment_config_from_dict(cfg_raw)

# Pin the resolved cnn_channels_2 by disabling further auto-scaling.
quantum_cfg_base = replace(config.quantum, target_params=-1)
training_cutoff = int(quantum_cfg_base.cutoff_dim)

if args.dtype is not None:
    quantum_cfg_base = replace(quantum_cfg_base, dtype=args.dtype)

# Mandatory subset_indices.npz — raises FileNotFoundError with a helpful message
# if the run was not produced by a recent full_experiment.py.
subset_indices = load_subset_indices(run_dir)


# ---------------------------------------------------------------------------
# Output directory (inside the trained run's directory)
# ---------------------------------------------------------------------------

eval_root = run_dir / "eval"
eval_root.mkdir(parents=True, exist_ok=True)

eval_name = args.output_name or f"cutoff_sweep_{datetime.now():%Y-%m-%d_%H-%M-%S}"
eval_dir = eval_root / eval_name
fig_dir  = eval_dir / "figures"
for d in (eval_dir, fig_dir):
    d.mkdir(parents=True, exist_ok=True)


class _Tee:
    """Duplicates writes to multiple streams (stdout → also a log file)."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, s):
        for st in self._streams:
            st.write(s)
        return len(s)

    def flush(self):
        for st in self._streams:
            st.flush()


_log_fh = open(eval_dir / "eval.log", "w", buffering=1)
sys.stdout = _Tee(sys.__stdout__, _log_fh)

print(f"Checkpoint:        {ckpt_path}")
print(f"Run directory:     {run_dir}")
print(f"Training cutoff:   {training_cutoff}")
print(f"Sweep cutoffs:     {args.cutoffs}")
print(f"Eval splits:       {args.eval_splits}")
print(f"Eval output dir:   {eval_dir}\n")


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}\n")


# ---------------------------------------------------------------------------
# Per-split loaders (reused across cutoffs for an apples-to-apples comparison).
# Default path reuses subset_indices.npz from the trained run; --*-fraction /
# --*-limit are explicit overrides that warn loudly because their results
# will not match the trained run's reported numbers.
# ---------------------------------------------------------------------------


def _apply_override(indices: np.ndarray, fraction: float | None,
                    limit: int | None, label: str) -> np.ndarray:
    """Apply --fraction / --limit on top of the reused subset. Warns loudly."""
    if limit is not None:
        n = min(limit, len(indices))
        out = indices[:n]
        warnings.warn(
            f"[{label}] --{label.lower()}-limit overrides subset reuse: "
            f"first {n}/{len(indices)} of the trained run's {label} subset. "
            "Results will NOT match the trained run.",
            RuntimeWarning, stacklevel=2,
        )
        return out
    if fraction is not None and fraction < 1.0:
        n = int(fraction * len(indices))
        g = np.random.default_rng(args.subset_seed)
        perm = g.permutation(len(indices))[:n]
        out = indices[np.sort(perm)]
        warnings.warn(
            f"[{label}] --{label.lower()}-fraction overrides subset reuse: "
            f"random {n}/{len(indices)} of the trained run's {label} subset "
            f"(subset-seed {args.subset_seed}). Results will NOT match the "
            "trained run.",
            RuntimeWarning, stacklevel=2,
        )
        return out
    return indices


def _build_split_loader(split: str) -> tuple[DataLoader, int, int]:
    """Build (loader, subset_size, parent_run_size) for `split`, reusing
    `subset_indices.npz` from the parent run (mandatory). Overrides warn.
    """
    is_train = split == "train"
    ds_full = PatchedDataset(config.data, train=is_train)
    parent_indices = subset_indices["train_indices" if is_train else "test_indices"]
    fraction = args.train_fraction if is_train else args.test_fraction
    limit    = args.train_limit    if is_train else args.test_limit
    indices = _apply_override(parent_indices, fraction, limit, split.capitalize())
    ds = Subset(ds_full, indices=indices.tolist())
    label = split.capitalize()
    if fraction is None and limit is None:
        print(f"{label} subset:       {len(ds):,} samples "
              f"(reused from subset_indices.npz)")
    print(f"{label} batches:      {(len(ds) + args.batch_size - 1) // args.batch_size}")
    return (
        DataLoader(ds, batch_size=args.batch_size, shuffle=False),
        len(ds),
        int(parent_indices.size),
    )


loaders: dict[str, DataLoader] = {}
split_sizes: dict[str, int] = {}
split_full_sizes: dict[str, int] = {}

# Build in a stable order so the printed log is deterministic.
for split in ("train", "test"):
    if split not in args.eval_splits:
        continue
    loader, n, n_full = _build_split_loader(split)
    loaders[split] = loader
    split_sizes[split] = n
    split_full_sizes[split] = n_full
    print()

# Diag loader — ALWAYS the original 512 from diag_indices, even under override.
# Resampling it would defeat cross-cutoff comparability of the state-norm and
# photon-number diagnostics. Absolute indices, so always against the full test
# PatchedDataset.
_test_ds_full = PatchedDataset(config.data, train=False)
diag_indices = subset_indices["diag_indices"]
diag_loader = DataLoader(
    Subset(_test_ds_full, indices=diag_indices.tolist()),
    batch_size=args.batch_size, shuffle=False,
)
print(f"Diag subset:       {len(diag_indices):,} samples (reused diag_indices)\n")


# ---------------------------------------------------------------------------
# Load checkpoint state dict once
# ---------------------------------------------------------------------------

ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
state_dict = ckpt["model_state_dict"]


# ---------------------------------------------------------------------------
# Per-cutoff run-dir synthesis helpers
# ---------------------------------------------------------------------------


def _relative_symlink(target: Path, link: Path) -> None:
    """Create a relative symlink at `link` pointing to `target`. Idempotent
    (skips if `link` already exists). Linux/SLURM target.
    """
    if link.exists() or link.is_symlink():
        return
    link.parent.mkdir(parents=True, exist_ok=True)
    rel = os.path.relpath(target, start=link.parent)
    os.symlink(rel, link)


def _synthesize_cutoff_rundir(
    D_new: int,
    cfg_eval: ExperimentConfig,
    test_eval: dict,
    train_eval: dict | None,
    quantum_stats: dict,
    mean_photon: np.ndarray,
    diag_raw: dict,
    lcu_snap: np.ndarray,
    poly_snap: np.ndarray,
    n_params: int,
) -> Path:
    """Write a self-contained, report_diagnostics-runnable single-epoch run
    dir at `eval_dir/D{D:02d}/`. Returns the path.
    """
    sub = eval_dir / f"D{D_new:02d}"
    (sub / "predictions").mkdir(parents=True, exist_ok=True)
    (sub / "diagnostics").mkdir(parents=True, exist_ok=True)
    (sub / "checkpoints").mkdir(parents=True, exist_ok=True)
    # report_diagnostics writes into `<run_dir>/figures/` but doesn't create
    # it (full_experiment.py does so at startup); create it here so the
    # synthesized dir matches the artefact contract.
    (sub / "figures").mkdir(parents=True, exist_ok=True)

    # config.json — parent config with overridden cutoff_dim and pinned cnn_channels_2
    cfg_dict = asdict(cfg_eval)
    with open(sub / "config.json", "w") as f:
        json.dump(cfg_dict, f, indent=2)

    # history.json — single-epoch training-log entries only. Derived metrics
    # (per-class acc, confusion, lcu/poly coeffs, mean photon, hypernet stats)
    # are read from the npz artefacts by report_diagnostics.
    src = train_eval if train_eval is not None else test_eval
    h = new_history(int(n_params), str(device))
    h["epoch"]["train_loss"]      = [float(src["loss"])]
    h["epoch"]["train_acc"]       = [float(src["acc"])]
    h["epoch"]["test_loss"]       = [float(test_eval["loss"])]
    h["epoch"]["test_acc"]        = [float(test_eval["acc"])]
    h["epoch"]["trunc_loss"]      = [float(src["trunc_loss"])]
    h["epoch"]["test_trunc_loss"] = [float(test_eval["trunc_loss"])]
    h["epoch"]["elapsed_sec"]     = [0.0]
    h["meta"]["best_test_acc"]    = float(test_eval["acc"])
    h["meta"]["best_epoch"]       = 1
    h["meta"]["total_runtime_sec"] = 0.0
    h["meta"]["completed_at"]     = datetime.now().isoformat(timespec="seconds")
    with open(sub / "history.json", "w") as f:
        json.dump(h, f, indent=2)

    # predictions/epoch_0001.npz (test) + optionally _train counterpart.
    # success_probs is absent for models without LCU post-selection.
    test_pred_kwargs = dict(
        y_true=test_eval["y_true"],
        y_pred=test_eval["y_pred"],
        y_probs=test_eval["y_probs"],
        readouts=test_eval["readouts"],
    )
    if test_eval.get("success_probs") is not None:
        test_pred_kwargs["success_probs"] = test_eval["success_probs"]
    np.savez_compressed(sub / "predictions" / "epoch_0001.npz",
                        **test_pred_kwargs)
    if train_eval is not None:
        train_pred_kwargs = dict(
            y_true=train_eval["y_true"],
            y_pred=train_eval["y_pred"],
            y_probs=train_eval["y_probs"],
            readouts=train_eval["readouts"],
        )
        if train_eval.get("success_probs") is not None:
            train_pred_kwargs["success_probs"] = train_eval["success_probs"]
        np.savez_compressed(sub / "predictions" / "epoch_0001_train.npz",
                            **train_pred_kwargs)

    # diagnostics/epoch_0001.npz — raw quantum-diagnostic arrays plus the
    # lcu/poly coefficient snapshots (so the post-hoc report_diagnostics
    # figures don't need to rebuild the model).
    diag_raw_with_coeffs = dict(diag_raw)
    diag_raw_with_coeffs["lcu_coeffs"] = lcu_snap
    diag_raw_with_coeffs["poly_coeffs"] = poly_snap
    np.savez_compressed(sub / "diagnostics" / "epoch_0001.npz", **diag_raw_with_coeffs)

    # Shared heavy artefacts — relative symlinks back to the parent run.
    parent_test_images = run_dir / "predictions" / "test_images.npz"
    if parent_test_images.is_file():
        _relative_symlink(parent_test_images,
                          sub / "predictions" / "test_images.npz")
    else:
        # Fallback: regenerate from the reused test loader. Only hit on runs
        # that pre-date the artefact contract.
        save_test_images_once(
            loaders["test"] if "test" in loaders else DataLoader(
                Subset(_test_ds_full, indices=subset_indices["test_indices"].tolist()),
                batch_size=args.batch_size, shuffle=False,
            ),
            config.data.image_size, config.data.patch_size,
            sub / "predictions" / "test_images.npz",
            progress="reassembling test images",
        )

    _relative_symlink(run_dir / "subset_indices.npz",
                      sub / "subset_indices.npz")
    _relative_symlink(ckpt_path,
                      sub / "checkpoints" / "final_model.pt")
    return sub


# ---------------------------------------------------------------------------
# Per-cutoff evaluation
# ---------------------------------------------------------------------------


_param_table_printed = False


def evaluate_at_cutoff(D_new: int) -> tuple[list[dict], Path | None]:
    """Build a fresh model at cutoff_dim=D_new, load weights, run evaluation
    on each requested split, capture quantum diagnostics (test only), and
    synthesize a report_diagnostics-runnable run dir (test only).

    Returns (rows, sub_run_dir): one row per evaluated split for aggregation,
    and the synthesized D{NN}/ dir (or None if test was not evaluated).
    """
    global _param_table_printed

    quantum_cfg_eval = replace(quantum_cfg_base, cutoff_dim=D_new)
    cfg_eval = replace(config, quantum=quantum_cfg_eval)

    model = build_model(cfg_eval).to(device)

    # Strict load — at D ≠ training_cutoff, mismatches mean an unexpected
    # D-dependent parameter slipped in and the result would be garbage.
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    assert not missing and not unexpected, (
        f"state_dict load mismatch at D={D_new}: "
        f"missing={missing}, unexpected={unexpected}"
    )

    if not _param_table_printed:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            print_parameter_table(model)
        print(buf.getvalue())
        _param_table_printed = True

    n_params = model.get_num_parameters()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()

    evals: dict[str, dict] = {}
    rows: list[dict] = []
    for split in args.eval_splits:
        t_split = time.time()
        evals[split] = evaluate(model, loaders[split], device,
                                progress=f"D={D_new} {split}")
        elapsed_split = time.time() - t_split
        rows.append({
            "split":        split,
            "cutoff_dim":   D_new,
            "acc":          float(evals[split]["acc"]),
            "ce_loss":      float(evals[split]["loss"]),
            "trunc_loss":   float(evals[split]["trunc_loss"]),
            "elapsed_sec":  float(elapsed_split),
            "peak_mem_mb":  None,   # populated once, after the full cutoff
            "n_samples":    int(evals[split]["y_true"].shape[0]),
            "mean_state_norm": None,
            "mean_photon":     None,
        })

    sub_dir: Path | None = None
    if "test" in evals:
        # quantum diagnostics on the reused diag subset (test only)
        stats_summary, mean_photon, diag_raw = quantum_diagnostics(
            model, diag_loader, device,
            progress=f"D={D_new} diagnostics",
        )
        lcu_snap, poly_snap = snapshot_coefficients(model)

        sub_dir = _synthesize_cutoff_rundir(
            D_new, cfg_eval,
            test_eval=evals["test"],
            train_eval=evals.get("train"),
            quantum_stats=stats_summary,
            mean_photon=mean_photon,
            diag_raw=diag_raw,
            lcu_snap=lcu_snap,
            poly_snap=poly_snap,
            n_params=int(n_params),
        )

        # Aggregate diag-derived columns on the test row
        norm_arrays = [diag_raw[k] for k in diag_raw if k.endswith("_state_norms")]
        mean_state_norm = float(np.concatenate(norm_arrays).mean()) if norm_arrays else None
        mean_photon_val = float(mean_photon.mean())
        for r in rows:
            if r["split"] == "test":
                r["mean_state_norm"] = mean_state_norm
                r["mean_photon"]     = mean_photon_val

    elapsed = time.time() - t0
    peak_mem_mb = (torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                   if device.type == "cuda" else None)
    # Stamp the same peak_mem onto each row from this cutoff (cumulative)
    for r in rows:
        r["peak_mem_mb"] = peak_mem_mb
    # Stamp end-to-end elapsed onto the test row (includes diagnostics)
    for r in rows:
        if r["split"] == "test":
            r["elapsed_sec"] = float(elapsed)

    return rows, sub_dir


print(f"{'Split':<7} {'Cutoff':<8} {'Acc':<10} {'CE loss':<10} {'Trunc loss':<12} "
      f"{'Elapsed':<10} {'Peak mem':<10}")
print("─" * 72)

per_cutoff: list[dict] = []
sub_run_dirs: list[Path] = []
for D_new in tqdm(args.cutoffs, desc="cutoffs", unit="D"):
    rows, sub = evaluate_at_cutoff(D_new)
    per_cutoff.extend(rows)
    if sub is not None:
        sub_run_dirs.append(sub)
    for r in rows:
        mem_str = (f"{r['peak_mem_mb']:.0f} MB"
                   if r["peak_mem_mb"] is not None else "n/a")
        print(f"{r['split']:<7} {r['cutoff_dim']:<8} {r['acc']:<10.4f} "
              f"{r['ce_loss']:<10.4f} {r['trunc_loss']:<12.4f} "
              f"{r['elapsed_sec']:<10.1f} {mem_str:<10}")


# ---------------------------------------------------------------------------
# Save results.json, results.csv, meta.json
# ---------------------------------------------------------------------------

results_payload = {
    "checkpoint": str(ckpt_path),
    "run_dir": str(run_dir),
    "training_cutoff": training_cutoff,
    "splits": list(args.eval_splits),
    "split_sizes": split_sizes,
    "per_cutoff": per_cutoff,
    "sub_run_dirs": [str(p) for p in sub_run_dirs],
}
with open(eval_dir / "results.json", "w") as f:
    json.dump(results_payload, f, indent=2)

with open(eval_dir / "results.csv", "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["split", "cutoff_dim", "acc", "ce_loss", "trunc_loss",
                    "elapsed_sec", "peak_mem_mb", "n_samples",
                    "mean_state_norm", "mean_photon"],
    )
    writer.writeheader()
    writer.writerows(per_cutoff)

meta = {
    "checkpoint": str(ckpt_path),
    "run_dir": str(run_dir),
    "training_cutoff": training_cutoff,
    "sweep_cutoffs": list(args.cutoffs),
    "eval_splits": list(args.eval_splits),
    "subset_source": "subset_indices.npz (reused from parent run)",
    "test_fraction": args.test_fraction,
    "test_limit": args.test_limit,
    "train_fraction": args.train_fraction,
    "train_limit": args.train_limit,
    "subset_seed": args.subset_seed,
    "split_sizes": split_sizes,
    "split_full_sizes": split_full_sizes,
    "diag_size": int(diag_indices.size),
    "dtype": quantum_cfg_base.dtype,
    "device": str(device),
    "batch_size": args.batch_size,
    "started_at": datetime.fromtimestamp(
        Path(eval_dir / "eval.log").stat().st_mtime
    ).isoformat(timespec="seconds"),
    "completed_at": datetime.now().isoformat(timespec="seconds"),
    "config_source": str(config_json),
    "resolved_quantum_config": asdict(quantum_cfg_base),
}
with open(eval_dir / "meta.json", "w") as f:
    json.dump(meta, f, indent=2)


# ---------------------------------------------------------------------------
# Aggregate trend plots
# ---------------------------------------------------------------------------


_SPLIT_STYLE = {
    "test":  {"color": "tab:blue",   "marker": "o"},
    "train": {"color": "tab:orange", "marker": "s"},
}


def _plot(metric: str, ylabel: str, title: str, fname: str,
          log_y: bool = False, splits: tuple[str, ...] | None = None) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    plot_splits = splits if splits is not None else tuple(args.eval_splits)
    has_data = False
    for split in plot_splits:
        rows = [r for r in per_cutoff if r["split"] == split
                and r.get(metric) is not None]
        if not rows:
            continue
        has_data = True
        xs = [r["cutoff_dim"] for r in rows]
        ys = [r[metric] for r in rows]
        style = _SPLIT_STYLE.get(split, {})
        ax.plot(xs, ys, lw=1.8, label=split, **style)
    if not has_data:
        plt.close(fig)
        return
    ax.axvline(training_cutoff, color="gray", ls="--", alpha=0.6,
               label=f"training D={training_cutoff}")
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel("cutoff_dim (D)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / fname, dpi=150)
    plt.close(fig)


_plot("acc",        "Accuracy",        "Accuracy vs cutoff_dim",
      "acc_vs_cutoff.png")
_plot("trunc_loss", "Mean truncation loss",
      "Truncation loss vs cutoff_dim", "trunc_loss_vs_cutoff.png")
_plot("ce_loss",    "Cross-entropy loss",
      "CE loss vs cutoff_dim", "ce_loss_vs_cutoff.png")
_plot("elapsed_sec", "Elapsed (s)",
      "Evaluation wall-time vs cutoff_dim", "elapsed_vs_cutoff.png",
      log_y=True)
# Quantum-diagnostic trends (test split only — diag is test-anchored)
_plot("mean_state_norm", r"Mean output state norm $\|\psi\|^2$",
      "State norm vs cutoff_dim", "state_norm_vs_cutoff.png",
      splits=("test",))
_plot("mean_photon", r"Mean $\langle \hat n \rangle$ over heads/modes",
      "Mean photon number vs cutoff_dim", "photon_number_vs_cutoff.png",
      splits=("test",))


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print(f"\nEval directory:        {eval_dir}/")
print(f"  results.json         (per-(split, cutoff) metrics + sub_run_dirs)")
print(f"  results.csv          (same data, pandas-friendly)")
print(f"  meta.json            (sweep configuration)")
print(f"  figures/             (6 PNGs: acc, trunc_loss, ce_loss, elapsed, "
      f"state_norm, photon_number)")
print(f"  D{{NN}}/              (one report_diagnostics-runnable sub-run dir per cutoff)")

if sub_run_dirs:
    print(f"\nTo render the full figure suite for each cutoff:")
    for sub in sub_run_dirs:
        print(f"  uv run python experiments/report_diagnostics.py --run-dir {sub}")

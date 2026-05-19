"""Cutoff-dim sweep evaluation for a trained CV-Quixer checkpoint.

Loads a checkpoint trained at one `cutoff_dim` D and re-evaluates the same
weights at a list of larger D values to study how Fock truncation affects
test accuracy and truncation loss. Safe because every trainable parameter is
D-independent in the quadrature_x readout path (gate matrices and observables
are rebuilt from learnable scalars at each forward pass).

Run:
    uv run python experiments/eval_cutoff_sweep.py \\
        --checkpoint results/runs/full_fashionmnist_*/checkpoints/final_model.pt \\
        [--cutoffs 6 8 10 12] \\
        [--batch-size 64] \\
        [--eval-splits test train] \\        # default: just test
        [--test-fraction 0.5] \\             # random subset of test set
        [--test-limit 64] \\                 # mutex with --test-fraction; smoke only
        [--train-fraction 0.1] \\            # random subset of train set
        [--train-limit 64] \\                # mutex with --train-fraction; smoke only
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
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

import dacite
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from cv_quixer.config.schema import ExperimentConfig
from cv_quixer.data.mnist import PatchedDataset
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
                    help="random subset of test set (0 < x <= 1). "
                         "Mutually exclusive with --test-limit.")
parser.add_argument("--test-limit", type=int, default=None,
                    help="take first N test samples (deterministic, "
                         "smoke-test only). Mutually exclusive with --test-fraction.")
parser.add_argument("--train-fraction", type=float, default=None,
                    help="random subset of train set (0 < x <= 1). "
                         "Mutually exclusive with --train-limit.")
parser.add_argument("--train-limit", type=int, default=None,
                    help="take first N train samples (deterministic, "
                         "smoke-test only). Mutually exclusive with --train-fraction.")
parser.add_argument("--subset-seed", type=int, default=42,
                    help="seed for the --train-fraction / --test-fraction random subsets "
                         "(applied independently to each split via dedicated generators)")
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

config: ExperimentConfig = dacite.from_dict(
    data_class=ExperimentConfig,
    data=cfg_raw,
    config=dacite.Config(strict=False),
)

# Pin the resolved cnn_channels_2 by disabling further auto-scaling.
quantum_cfg_base = replace(config.quantum, target_params=-1)
training_cutoff = int(quantum_cfg_base.cutoff_dim)

if args.dtype is not None:
    quantum_cfg_base = replace(quantum_cfg_base, dtype=args.dtype)


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
# Per-split loaders (selected once, reused across cutoffs for an apples-to-
# apples comparison)
# ---------------------------------------------------------------------------

def _build_split_loader(split: str, fraction: float | None,
                        limit: int | None) -> tuple[DataLoader, int, int]:
    """Build a (DataLoader, subset_size, full_size) for `split`.

    Honours `--{split}-fraction` / `--{split}-limit`; both random subset
    selections use a local `torch.Generator` seeded with `--subset-seed`, so
    runs with the same seed are reproducible and do not perturb global RNG.
    """
    is_train = split == "train"
    ds_full = PatchedDataset(config.data, train=is_train)
    label = split.capitalize()

    if limit is not None:
        indices = list(range(min(limit, len(ds_full))))
        ds = Subset(ds_full, indices=indices)
        print(f"{label} subset:       first {len(ds):,} samples (--{split}-limit)")
    elif fraction is not None and fraction < 1.0:
        n = int(fraction * len(ds_full))
        g = torch.Generator().manual_seed(args.subset_seed)
        perm = torch.randperm(len(ds_full), generator=g)[:n].tolist()
        ds = Subset(ds_full, indices=perm)
        print(f"{label} subset:       random {len(ds):,} / {len(ds_full):,} "
              f"samples (--{split}-fraction {fraction}, subset-seed {args.subset_seed})")
    else:
        ds = ds_full
        print(f"{label} set:          full {len(ds):,} samples")

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    print(f"{label} batches:      {len(loader)}")
    return loader, len(ds), len(ds_full)


loaders: dict[str, DataLoader] = {}
split_sizes: dict[str, int] = {}
split_full_sizes: dict[str, int] = {}

# Build in a stable order so the printed log is deterministic.
for split in ("train", "test"):
    if split not in args.eval_splits:
        continue
    fraction = args.train_fraction if split == "train" else args.test_fraction
    limit    = args.train_limit    if split == "train" else args.test_limit
    loader, n, n_full = _build_split_loader(split, fraction, limit)
    loaders[split] = loader
    split_sizes[split] = n
    split_full_sizes[split] = n_full
    print()


# ---------------------------------------------------------------------------
# Load checkpoint state dict once
# ---------------------------------------------------------------------------

ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
state_dict = ckpt["model_state_dict"]


# ---------------------------------------------------------------------------
# Per-cutoff evaluation
# ---------------------------------------------------------------------------


_param_table_printed = False


@torch.no_grad()
def evaluate_at_cutoff(D_new: int, split: str, loader: DataLoader) -> dict:
    """Build a fresh model at cutoff_dim=D_new, load weights, evaluate `split`."""
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
        # One-time parameter sanity check on the very first (split, cutoff)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            print_parameter_table(model)
        print(buf.getvalue())
        _param_table_printed = True

    model.eval()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    ce_sum = 0.0
    trunc_sum = 0.0
    correct = 0
    total = 0
    logits_chunks: list[torch.Tensor] = []
    labels_chunks: list[torch.Tensor] = []

    t0 = time.time()
    for patches, labels in tqdm(
        loader, desc=f"{split:<5} D={D_new:>2}", unit="batch", leave=False
    ):
        patches = patches.to(device)
        labels = labels.to(device)
        out = model(patches, return_trunc_loss=True)
        logits, trunc_loss = out.logits, out.trunc_loss
        n = labels.size(0)
        ce_sum    += F.cross_entropy(logits, labels, reduction="sum").item()
        trunc_sum += float(trunc_loss.item()) * n
        correct   += (logits.argmax(dim=-1) == labels).sum().item()
        total     += n
        logits_chunks.append(logits.detach().to("cpu"))
        labels_chunks.append(labels.detach().to("cpu"))
    elapsed = time.time() - t0

    peak_mem_mb = (torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                   if device.type == "cuda" else None)

    logits_all = torch.cat(logits_chunks, dim=0)
    labels_all = torch.cat(labels_chunks, dim=0)
    torch.save(
        {"logits": logits_all, "labels": labels_all,
         "cutoff_dim": D_new, "split": split, "set_size": total},
        eval_dir / f"logits_{split}_D{D_new:02d}.pt",
    )

    return {
        "split":        split,
        "cutoff_dim":   D_new,
        "acc":          correct / total,
        "ce_loss":      ce_sum / total,
        "trunc_loss":   trunc_sum / total,
        "elapsed_sec":  elapsed,
        "peak_mem_mb":  peak_mem_mb,
        "n_samples":    total,
    }


print(f"{'Split':<7} {'Cutoff':<8} {'Acc':<10} {'CE loss':<10} {'Trunc loss':<12} "
      f"{'Elapsed':<10} {'Peak mem':<10}")
print("─" * 72)

per_cutoff: list[dict] = []
for split in args.eval_splits:
    loader = loaders[split]
    for D_new in args.cutoffs:
        res = evaluate_at_cutoff(D_new, split, loader)
        per_cutoff.append(res)
        mem_str = (f"{res['peak_mem_mb']:.0f} MB"
                   if res["peak_mem_mb"] is not None else "n/a")
        print(f"{res['split']:<7} {res['cutoff_dim']:<8} {res['acc']:<10.4f} "
              f"{res['ce_loss']:<10.4f} {res['trunc_loss']:<12.4f} "
              f"{res['elapsed_sec']:<10.1f} {mem_str:<10}")


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
}
with open(eval_dir / "results.json", "w") as f:
    json.dump(results_payload, f, indent=2)

with open(eval_dir / "results.csv", "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["split", "cutoff_dim", "acc", "ce_loss", "trunc_loss",
                    "elapsed_sec", "peak_mem_mb", "n_samples"],
    )
    writer.writeheader()
    writer.writerows(per_cutoff)

meta = {
    "checkpoint": str(ckpt_path),
    "run_dir": str(run_dir),
    "training_cutoff": training_cutoff,
    "sweep_cutoffs": list(args.cutoffs),
    "eval_splits": list(args.eval_splits),
    "test_fraction": args.test_fraction,
    "test_limit": args.test_limit,
    "train_fraction": args.train_fraction,
    "train_limit": args.train_limit,
    "subset_seed": args.subset_seed,
    "split_sizes": split_sizes,
    "split_full_sizes": split_full_sizes,
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
# Plots
# ---------------------------------------------------------------------------


_SPLIT_STYLE = {
    "test":  {"color": "tab:blue",   "marker": "o"},
    "train": {"color": "tab:orange", "marker": "s"},
}


def _plot(metric: str, ylabel: str, title: str, fname: str,
          log_y: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for split in args.eval_splits:
        rows = [r for r in per_cutoff if r["split"] == split]
        xs = [r["cutoff_dim"] for r in rows]
        ys = [r[metric] for r in rows]
        style = _SPLIT_STYLE.get(split, {})
        ax.plot(xs, ys, lw=1.8, label=split, **style)
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


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print(f"\nEval directory:        {eval_dir}/")
print(f"  results.json         (per-(split, cutoff) metrics)")
print(f"  results.csv          (same data, pandas-friendly)")
print(f"  meta.json            (sweep configuration)")
print(f"  logits_<split>_D*.pt (per-(split, cutoff) logits + labels for offline analysis)")
print(f"  figures/             (4 PNGs: acc, trunc_loss, ce_loss, elapsed)")

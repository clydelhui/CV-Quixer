"""Full FashionMNIST experiment for CV-Quixer.

Same quantum config as `experiments/mini_experiment.py` (num_modes=2, cutoff_dim=6,
num_heads=4, poly_degree=2, ~13.7k params via target_params auto-scaling), but
trained on the **full** 60k train / 10k test FashionMNIST split. Default 3 epochs;
~75-90 min/epoch on V100, ~30-45 min/epoch on A100.

Per-batch metrics (CE loss, trunc loss, total loss, batch accuracy, gradient L2 norm)
are logged alongside per-epoch metrics so the loss/trunc/accuracy/gradient curves
have ~1800-2800 data points instead of 2-3.

Run directory layout (one self-contained folder per run):

    results/runs/full_fashionmnist_YYYY-MM-DD_HH-MM-SS/
    ├── config.json                  # full resolved ExperimentConfig
    ├── history.json                 # epoch + batch + meta metrics (plot source of truth)
    ├── parameter_table.txt          # snapshot of print_parameter_table()
    ├── checkpoints/
    │   ├── latest.pt                # overwritten every epoch (resume safety)
    │   ├── best.pt                  # best test-acc snapshot
    │   ├── final_model.pt           # written once at end of training
    │   └── epoch_NNNN.pt            # versioned per-epoch checkpoint
    ├── figures/                     # all PNGs re-rendered after every epoch
    └── logs/

Run (fresh):
    uv run python experiments/full_experiment.py

Resume from a checkpoint (continues writing into the same run directory):
    uv run python experiments/full_experiment.py \\
        --resume results/runs/full_fashionmnist_2026-05-15_14-30-00/checkpoints/latest.pt

Local smoke test (reduced scale):
    uv run python experiments/full_experiment.py \\
        --epochs 1 --train-limit 256 --test-limit 64
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from cv_quixer.config.schema import (
    DataConfig,
    ExperimentConfig,
    QuantumConfig,
    TrainingConfig,
)
from cv_quixer.data.mnist import PatchedDataset
from cv_quixer.models import build_model
from cv_quixer.utils import print_parameter_table

# ---------------------------------------------------------------------------
# Defaults (CLI-overridable below)
# ---------------------------------------------------------------------------

EPOCHS = 3
BATCH_SIZE = 64
TARGET_PARAMS = 13_760
CHECKPOINT_INTERVAL = 1   # versioned epoch_NNNN.pt every N epochs
MA_WINDOW = 50            # moving-average window for per-batch plots


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--resume", type=str, default=None,
                    help="path to checkpoint .pt file to resume from. The "
                         "parent run directory is reused so the run continues "
                         "writing into the same folder.")
parser.add_argument("--epochs", type=int, default=None,
                    help=f"override the default EPOCHS={EPOCHS}")
parser.add_argument("--train-limit", type=int, default=None,
                    help="cap training set to first N samples (smoke test)")
parser.add_argument("--test-limit", type=int, default=None,
                    help="cap test set to first N samples (smoke test)")
args = parser.parse_args()

if args.epochs is not None:
    EPOCHS = args.epochs


# ---------------------------------------------------------------------------
# Config (quantum config matches mini_experiment.py — do not change)
# ---------------------------------------------------------------------------

data_cfg = DataConfig(
    dataset="fashionmnist",
    normalize=True,
    patch_size=7,
    batch_size=BATCH_SIZE,
    num_workers=0,
    data_root="data/",
)
quantum_cfg = QuantumConfig(
    num_modes=2,
    cutoff_dim=6,
    num_heads=4,
    cnn_channels_1=8,
    cnn_channels_2=16,          # overridden by target_params auto-scaling
    cnn_kernel_size=3,
    decoder_hidden_dim=32,
    poly_degree=2,
    dtype="complex64",
    trunc_penalty="norm",
    trunc_lambda=0.01,
    target_params=TARGET_PARAMS,
)
config = ExperimentConfig(
    name="full_fashionmnist",
    model="quantum",
    data=data_cfg,
    quantum=quantum_cfg,
    training=TrainingConfig(lr=1e-3, epochs=EPOCHS, seed=42),
    use_wandb=False,
)


# ---------------------------------------------------------------------------
# Run directory: fresh datetime or recovered from --resume path
# ---------------------------------------------------------------------------

if args.resume:
    resume_path = Path(args.resume).resolve()
    if not resume_path.is_file():
        raise FileNotFoundError(f"--resume path does not exist: {resume_path}")
    # checkpoints/<name>.pt → parent is <run_dir>/checkpoints → grandparent is run_dir
    run_dir = resume_path.parent.parent
    print(f"Resuming into existing run directory: {run_dir}")
else:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path("results/runs") / f"full_fashionmnist_{timestamp}"
    print(f"Fresh run directory: {run_dir}")

ckpt_dir = run_dir / "checkpoints"
fig_dir  = run_dir / "figures"
log_dir  = run_dir / "logs"
for d in (run_dir, ckpt_dir, fig_dir, log_dir):
    d.mkdir(parents=True, exist_ok=True)


class _Tee:
    """Duplicates writes to multiple streams (used to tee stdout to a log file)."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, s):
        for st in self._streams:
            st.write(s)
        return len(s)

    def flush(self):
        for st in self._streams:
            st.flush()


# Tee stdout to logs/train.log (append on resume, fresh otherwise). tqdm writes to
# stderr by default, so progress bars do not pollute the log file.
_log_fh = open(log_dir / "train.log", "a" if args.resume else "w", buffering=1)
sys.stdout = _Tee(sys.__stdout__, _log_fh)


# ---------------------------------------------------------------------------
# Device + seed
# ---------------------------------------------------------------------------

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}\n")

torch.manual_seed(config.training.seed)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

train_ds_full = PatchedDataset(data_cfg, train=True)
test_ds_full  = PatchedDataset(data_cfg, train=False)

if args.train_limit is not None:
    print(f"Limiting train set to first {args.train_limit:,} samples (smoke test)")
    train_ds = Subset(train_ds_full, indices=list(range(args.train_limit)))
else:
    train_ds = train_ds_full

if args.test_limit is not None:
    print(f"Limiting test set to first {args.test_limit:,} samples (smoke test)")
    test_ds = Subset(test_ds_full, indices=list(range(args.test_limit)))
else:
    test_ds = test_ds_full

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)
print(f"Train samples: {len(train_ds):,} | test samples: {len(test_ds):,}")
print(f"Train batches/epoch: {len(train_loader):,} | test batches: {len(test_loader):,}\n")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

model = build_model(config).to(device)

# Capture parameter table to both stdout and a text file
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    print_parameter_table(model)
param_table_str = buf.getvalue()
print(param_table_str)
(run_dir / "parameter_table.txt").write_text(param_table_str)

n_params = model.get_num_parameters()
print(f"Trainable parameters: {n_params:,}  (target: {TARGET_PARAMS:,})\n")

# Shape check on one batch
_patches, _ = next(iter(train_loader))
_logits = model(_patches.to(device))
assert _logits.shape[-1] == 10, f"Expected logits dim 10, got {_logits.shape}"
print(f"Forward shape OK — logits: {tuple(_logits.shape)}\n")


# ---------------------------------------------------------------------------
# Optimizer + state
# ---------------------------------------------------------------------------

optimizer = torch.optim.Adam(model.parameters(), lr=config.training.lr)

history: dict = {
    "epoch": {
        "train_loss": [], "train_acc": [],
        "test_loss":  [], "test_acc":  [],
        "trunc_loss": [], "elapsed_sec": [],
    },
    "batch": {
        "step":       [], "epoch":      [],
        "train_loss": [], "trunc_loss": [], "total_loss": [],
        "batch_acc":  [], "grad_norm":  [],
    },
    "meta": {
        "best_test_acc": None, "best_epoch": None,
        "total_runtime_sec": None, "n_params": int(n_params),
        "device": str(device),
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "completed_at": None,
    },
}

start_epoch = 1
global_step = 0
if args.resume:
    ckpt = torch.load(args.resume, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    history = ckpt["history"]
    start_epoch = ckpt["epoch"] + 1
    global_step = len(history["batch"]["step"])
    print(f"Resumed from {args.resume}")
    print(f"  Continuing from epoch {start_epoch} (global_step={global_step})\n")


# ---------------------------------------------------------------------------
# Save resolved config (after auto-scaling has chosen cnn_channels_2)
# ---------------------------------------------------------------------------

# Reflect the auto-scaled cnn_channels_2 (chosen by binary search inside
# CVQuixer.__init__) in the saved config so the JSON matches the model actually built.
config_to_save = asdict(config)
if hasattr(model, "config") and hasattr(model.config, "cnn_channels_2"):
    config_to_save["quantum"]["cnn_channels_2"] = int(model.config.cnn_channels_2)

with open(run_dir / "config.json", "w") as f:
    json.dump(config_to_save, f, indent=2)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------


def _grad_l2_norm(parameters) -> float:
    """Global L2 norm of gradients (diagnostic only, no clipping)."""
    norms = [p.grad.detach().norm(2) for p in parameters if p.grad is not None]
    if not norms:
        return 0.0
    return float(torch.norm(torch.stack(norms), 2).item())


def train_epoch(epoch: int) -> tuple[float, float, float]:
    """One epoch of training. Returns (avg_ce, avg_trunc, train_acc).

    Appends a record per batch into history["batch"].
    """
    global global_step
    model.train()
    total_ce, total_trunc, correct, total = 0.0, 0.0, 0, 0
    for patches, labels in tqdm(
        train_loader, desc=f"Epoch {epoch:>3}/{EPOCHS}", leave=False, unit="batch"
    ):
        patches, labels = patches.to(device), labels.to(device)
        optimizer.zero_grad()
        logits, trunc_loss = model(patches, return_trunc_loss=True)
        ce_loss = F.cross_entropy(logits, labels)
        loss = ce_loss + quantum_cfg.trunc_lambda * trunc_loss
        loss.backward()

        grad_norm = _grad_l2_norm(model.parameters())
        optimizer.step()

        n = labels.size(0)
        preds = logits.argmax(dim=-1)
        batch_correct = (preds == labels).sum().item()
        batch_acc = batch_correct / n

        total_ce    += ce_loss.item() * n
        total_trunc += trunc_loss.item() * n
        correct     += batch_correct
        total       += n

        global_step += 1
        history["batch"]["step"].append(global_step)
        history["batch"]["epoch"].append(epoch)
        history["batch"]["train_loss"].append(float(ce_loss.item()))
        history["batch"]["trunc_loss"].append(float(trunc_loss.item()))
        history["batch"]["total_loss"].append(float(loss.item()))
        history["batch"]["batch_acc"].append(float(batch_acc))
        history["batch"]["grad_norm"].append(float(grad_norm))

    return total_ce / total, total_trunc / total, correct / total


@torch.no_grad()
def evaluate() -> tuple[float, float]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for patches, labels in test_loader:
        patches, labels = patches.to(device), labels.to(device)
        logits = model(patches)
        total_loss += F.cross_entropy(logits, labels).item() * labels.size(0)
        correct    += (logits.argmax(dim=-1) == labels).sum().item()
        total      += labels.size(0)
    return total_loss / total, correct / total


# ---------------------------------------------------------------------------
# Plotting (re-rendered after every epoch so a SLURM kill mid-run still
# leaves usable figures)
# ---------------------------------------------------------------------------


def _moving_average(x: list[float], window: int) -> np.ndarray:
    if len(x) < window:
        return np.array(x, dtype=float)
    cumsum = np.cumsum(np.insert(np.asarray(x, dtype=float), 0, 0.0))
    ma = (cumsum[window:] - cumsum[:-window]) / float(window)
    # left-pad with NaN so x-axis lengths match
    return np.concatenate([np.full(window - 1, np.nan), ma])


def _epoch_boundary_steps() -> list[int]:
    """Global step at which each epoch ended (used for vertical markers)."""
    epochs_arr = np.asarray(history["batch"]["epoch"])
    steps_arr  = np.asarray(history["batch"]["step"])
    boundaries = []
    for e in sorted(set(epochs_arr.tolist())):
        mask = epochs_arr == e
        if mask.any():
            boundaries.append(int(steps_arr[mask].max()))
    return boundaries[:-1]  # don't draw a line at the final step


def save_figures() -> None:
    epoch_x = list(range(1, len(history["epoch"]["train_loss"]) + 1))
    eh = history["epoch"]
    bh = history["batch"]

    # Per-epoch loss
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(epoch_x, eh["train_loss"], label="train loss", marker="o")
    ax.plot(epoch_x, eh["test_loss"],  label="test loss",  marker="s")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Cross-entropy loss")
    ax.set_title("Loss (per epoch)"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(fig_dir / "loss_curve.png", dpi=150); plt.close(fig)

    # Per-epoch accuracy
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(epoch_x, eh["train_acc"], label="train acc", marker="o")
    ax.plot(epoch_x, eh["test_acc"],  label="test acc",  marker="s")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy (per epoch)"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(fig_dir / "accuracy_curve.png", dpi=150); plt.close(fig)

    # Per-epoch trunc loss
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(epoch_x, eh["trunc_loss"], label="trunc loss",
            color="tab:orange", marker="o")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Mean per-patch truncation loss")
    ax.set_title("Truncation loss (per epoch)"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(fig_dir / "trunc_loss_curve.png", dpi=150); plt.close(fig)

    if not bh["step"]:
        return

    steps = bh["step"]
    boundaries = _epoch_boundary_steps()

    def _per_batch_plot(values: list[float], ylabel: str, title: str,
                        fname: str, log_y: bool = False) -> None:
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(steps, values, alpha=0.25, lw=0.7, label="per batch")
        ma = _moving_average(values, MA_WINDOW)
        if not np.all(np.isnan(ma)):
            ax.plot(steps, ma, lw=1.8, label=f"moving avg (w={MA_WINDOW})")
        for b in boundaries:
            ax.axvline(b, color="gray", ls="--", alpha=0.4, lw=0.8)
        if log_y:
            ax.set_yscale("log")
        ax.set_xlabel("Global batch step"); ax.set_ylabel(ylabel)
        ax.set_title(title); ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(fig_dir / fname, dpi=150); plt.close(fig)

    _per_batch_plot(bh["train_loss"], "Cross-entropy loss",
                    "Train CE loss (per batch)", "per_batch_train_loss.png")
    _per_batch_plot(bh["trunc_loss"], "Truncation loss",
                    "Truncation loss (per batch)", "per_batch_trunc_loss.png")
    _per_batch_plot(bh["batch_acc"], "Batch accuracy",
                    "Train accuracy (per batch)", "per_batch_train_accuracy.png")
    _per_batch_plot(bh["grad_norm"], "Gradient L2 norm (log scale)",
                    "Gradient norm (per batch)", "per_batch_grad_norm.png",
                    log_y=True)


def save_history() -> None:
    with open(run_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

print(f"{'Epoch':<7} {'Train loss':<12} {'Train acc':<11} "
      f"{'Test loss':<11} {'Test acc':<11} {'Trunc loss':<12} {'Time'}")
print("─" * 76)

run_start = time.time()
best_test_acc = max(history["epoch"]["test_acc"]) if history["epoch"]["test_acc"] else -1.0
best_epoch    = (history["epoch"]["test_acc"].index(best_test_acc) + 1
                 if history["epoch"]["test_acc"] else None)

for epoch in range(start_epoch, EPOCHS + 1):
    t0 = time.time()
    train_loss, trunc_loss, train_acc = train_epoch(epoch)
    test_loss, test_acc = evaluate()
    elapsed = time.time() - t0

    history["epoch"]["train_loss"].append(train_loss)
    history["epoch"]["train_acc"].append(train_acc)
    history["epoch"]["test_loss"].append(test_loss)
    history["epoch"]["test_acc"].append(test_acc)
    history["epoch"]["trunc_loss"].append(trunc_loss)
    history["epoch"]["elapsed_sec"].append(elapsed)

    print(f"{epoch:<7} {train_loss:<12.4f} {train_acc:<11.3f} "
          f"{test_loss:<11.4f} {test_acc:<11.3f} {trunc_loss:<12.4f} {elapsed:.1f}s")

    # Checkpoints
    ckpt_payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": history,
    }
    torch.save(ckpt_payload, ckpt_dir / "latest.pt")
    if epoch % CHECKPOINT_INTERVAL == 0:
        torch.save(ckpt_payload, ckpt_dir / f"epoch_{epoch:04d}.pt")
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        best_epoch = epoch
        torch.save(ckpt_payload, ckpt_dir / "best.pt")

    # Persist metadata + figures every epoch so a SLURM kill is non-destructive
    history["meta"]["best_test_acc"] = float(best_test_acc)
    history["meta"]["best_epoch"]    = int(best_epoch) if best_epoch is not None else None
    history["meta"]["total_runtime_sec"] = float(time.time() - run_start)
    save_history()
    save_figures()

total_runtime = time.time() - run_start

# ---------------------------------------------------------------------------
# Final saves
# ---------------------------------------------------------------------------

history["meta"]["completed_at"] = datetime.now().isoformat(timespec="seconds")
history["meta"]["total_runtime_sec"] = float(total_runtime)
save_history()
save_figures()

torch.save(
    {"model_state_dict": model.state_dict(), "history": history},
    ckpt_dir / "final_model.pt",
)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print(f"\nTotal runtime:    {total_runtime / 60:.1f} min")
print(f"Best test acc:    {best_test_acc:.4f} (epoch {best_epoch})")
print(f"Run directory:    {run_dir}/")
print(f"  config         → config.json")
print(f"  history        → history.json")
print(f"  figures        → figures/")
print(f"  checkpoints    → checkpoints/  (latest.pt, best.pt, final_model.pt, epoch_NNNN.pt)")

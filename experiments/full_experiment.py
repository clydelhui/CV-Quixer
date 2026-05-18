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
        --epochs 1 --train-fraction 0.1 --test-fraction 0.1
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
                    help="cap training set to first N samples (deterministic, "
                         "smoke-test only). Mutually exclusive with --train-fraction.")
parser.add_argument("--test-limit", type=int, default=None,
                    help="cap test set to first N samples (deterministic, "
                         "smoke-test only). Mutually exclusive with --test-fraction.")
parser.add_argument("--train-fraction", type=float, default=None,
                    help="random subset of the train set (0 < x <= 1). "
                         "Mutually exclusive with --train-limit.")
parser.add_argument("--test-fraction", type=float, default=None,
                    help="random subset of the test set (0 < x <= 1). "
                         "Mutually exclusive with --test-limit.")
parser.add_argument("--subset-seed", type=int, default=42,
                    help="seed shared by --train-fraction and --test-fraction "
                         "random subsets")
args = parser.parse_args()

if args.train_fraction is not None and args.train_limit is not None:
    parser.error("--train-fraction and --train-limit are mutually exclusive")
if args.test_fraction is not None and args.test_limit is not None:
    parser.error("--test-fraction and --test-limit are mutually exclusive")
if args.train_fraction is not None and not (0.0 < args.train_fraction <= 1.0):
    parser.error("--train-fraction must be in (0, 1]")
if args.test_fraction is not None and not (0.0 < args.test_fraction <= 1.0):
    parser.error("--test-fraction must be in (0, 1]")

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
preds_dir = run_dir / "predictions"
diag_dir  = run_dir / "diagnostics"
for d in (run_dir, ckpt_dir, fig_dir, log_dir, preds_dir, diag_dir):
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
    n = min(args.train_limit, len(train_ds_full))
    train_ds = Subset(train_ds_full, indices=list(range(n)))
    print(f"Train subset:  first {len(train_ds):,} samples (--train-limit)")
elif args.train_fraction is not None and args.train_fraction < 1.0:
    n = int(args.train_fraction * len(train_ds_full))
    g = torch.Generator().manual_seed(args.subset_seed)
    perm = torch.randperm(len(train_ds_full), generator=g)[:n].tolist()
    train_ds = Subset(train_ds_full, indices=perm)
    print(f"Train subset:  random {len(train_ds):,} / {len(train_ds_full):,} "
          f"samples (--train-fraction {args.train_fraction}, seed {args.subset_seed})")
else:
    train_ds = train_ds_full
    print(f"Train set:     full {len(train_ds):,} samples")

if args.test_limit is not None:
    n = min(args.test_limit, len(test_ds_full))
    test_ds = Subset(test_ds_full, indices=list(range(n)))
    print(f"Test subset:   first {len(test_ds):,} samples (--test-limit)")
elif args.test_fraction is not None and args.test_fraction < 1.0:
    n = int(args.test_fraction * len(test_ds_full))
    g = torch.Generator().manual_seed(args.subset_seed)
    perm = torch.randperm(len(test_ds_full), generator=g)[:n].tolist()
    test_ds = Subset(test_ds_full, indices=perm)
    print(f"Test subset:   random {len(test_ds):,} / {len(test_ds_full):,} "
          f"samples (--test-fraction {args.test_fraction}, seed {args.subset_seed})")
else:
    test_ds = test_ds_full
    print(f"Test set:      full {len(test_ds):,} samples")

train_loader      = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
train_eval_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader       = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)


def _save_test_images_once(loader, image_size: int, patch_size: int,
                           out_path: Path) -> None:
    """Reassemble the patches of every test sample into a (N, H, W) array and
    write it once. Idempotent: skip the work if the file already exists with
    a matching N. Used by experiments/report_diagnostics.py's default
    file-only mode to draw the misclassification gallery without re-running
    the model.
    """
    n_total = len(loader.dataset)
    if out_path.is_file():
        with np.load(out_path) as existing:
            if existing["images"].shape[0] == n_total:
                return
    grid = image_size // patch_size
    images = np.zeros((n_total, image_size, image_size), dtype=np.float32)
    labels = np.zeros(n_total, dtype=np.int64)
    idx = 0
    for patches, lbls in loader:
        # patches: (B, N_patches, patch_dim)
        bs = patches.shape[0]
        p_np = patches.numpy().astype(np.float32)
        for b in range(bs):
            for k in range(p_np.shape[1]):
                r, c = divmod(k, grid)
                images[idx + b,
                       r*patch_size:(r+1)*patch_size,
                       c*patch_size:(c+1)*patch_size] = (
                    p_np[b, k].reshape(patch_size, patch_size)
                )
        labels[idx:idx + bs] = lbls.numpy().astype(np.int64)
        idx += bs
    np.savez_compressed(out_path, images=images, labels=labels)


_save_test_images_once(
    test_loader, data_cfg.image_size, data_cfg.patch_size,
    preds_dir / "test_images.npz",
)

# Fixed diagnostic subset (sampled once at startup, stable across epochs and
# across resumes thanks to the fixed seed).
DIAG_SIZE = min(512, len(test_ds))
_diag_g = torch.Generator().manual_seed(args.subset_seed + 1)
diag_indices_in_test = torch.randperm(len(test_ds), generator=_diag_g)[:DIAG_SIZE].tolist()
diag_ds = Subset(test_ds, indices=diag_indices_in_test)
diag_loader = DataLoader(diag_ds, batch_size=BATCH_SIZE, shuffle=False)

print(f"Train samples: {len(train_ds):,} | test samples: {len(test_ds):,} "
      f"| diagnostic subset: {len(diag_ds):,}")
print(f"Train batches/epoch: {len(train_loader):,} | test batches: {len(test_loader):,}\n")


def _absolute_indices(ds, full_len: int) -> np.ndarray:
    """Indices into the underlying full PatchedDataset. `Subset.indices` are
    already absolute. For an un-subset dataset we return arange(full_len)."""
    if isinstance(ds, Subset):
        return np.asarray(ds.indices, dtype=np.int64)
    return np.arange(full_len, dtype=np.int64)


def _save_subset_indices_once(out_path: Path) -> None:
    """Record train/test/diag subset indices (absolute, into the full
    PatchedDataset) so consumers like experiments/report_diagnostics.py's
    --full-inference path can reconstruct the exact same Subset.

    Idempotent: skip if the file already exists with matching sizes.
    """
    train_abs = _absolute_indices(train_ds, len(train_ds_full))
    test_abs  = _absolute_indices(test_ds,  len(test_ds_full))
    diag_abs  = test_abs[diag_indices_in_test]
    if out_path.is_file():
        with np.load(out_path) as existing:
            if (existing["train_indices"].size == train_abs.size
                and existing["test_indices"].size == test_abs.size
                and existing["diag_indices"].size == diag_abs.size):
                return
    np.savez_compressed(
        out_path,
        train_indices=train_abs,
        test_indices=test_abs,
        diag_indices=diag_abs,
    )


_save_subset_indices_once(run_dir / "subset_indices.npz")


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

_EPOCH_KEYS_NEW: tuple[str, ...] = (
    "train_per_class_acc", "test_per_class_acc",
    "train_confusion",     "test_confusion",
    "test_trunc_loss",
    "lcu_coeffs", "poly_coeffs",
    "hypernet_stats", "mean_photon_number",
)


def _ensure_history_schema(h: dict) -> None:
    """Idempotently add any missing diagnostic fields to a (possibly resumed) history dict."""
    h.setdefault("epoch", {})
    for k in _EPOCH_KEYS_NEW:
        h["epoch"].setdefault(k, [])


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
_ensure_history_schema(history)

start_epoch = 1
global_step = 0
if args.resume:
    ckpt = torch.load(args.resume, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    history = ckpt["history"]
    _ensure_history_schema(history)
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


def _gate_param_layout(num_modes: int, n_bs: int) -> list[tuple[str, int, int]]:
    """Slice layout for hypernetwork gate-parameter outputs.

    Derived from cv_attention._GATE_SEQUENCE — single source of truth for
    the gate set. Each op contributes one (name, start, count) entry per
    parameter name; slice keys are f"{op.name}_{param_name}" to match the
    legacy names (squeeze_r, squeeze_phi, bs_theta, bs_phi, rot_phi,
    disp_re, disp_im, kerr_kappa).

    Returns list of (name, start, count); count==0 means the gate is absent.
    """
    from cv_quixer.models.quantum.cv_attention import _GATE_SEQUENCE

    layout: list[tuple[str, int, int]] = []
    offset = 0
    for op in _GATE_SEQUENCE:
        n_sites = num_modes if op.site_kind == "mode" else n_bs
        for p in op.param_names:
            layout.append((f"{op.name}_{p}", offset, n_sites))
            offset += n_sites
    return layout


@torch.no_grad()
def _snapshot_coefficients(model) -> tuple[np.ndarray, np.ndarray]:
    """LCU (real, imag) and polynomial coefficients per head — JSON-friendly arrays.

    Returns:
        lcu:  (num_heads, num_patches, 2) float32 — last dim = [real, imag]
        poly: (num_heads, poly_degree+1) float32
    """
    lcu_chunks, poly_chunks = [], []
    for head in model.cv_attention.heads:
        b = head.lcu_coeffs().detach().cpu()
        lcu_chunks.append(torch.stack([b.real, b.imag], dim=-1))
        poly_chunks.append(head.poly_coeffs().detach().cpu())
    lcu = torch.stack(lcu_chunks).numpy().astype(np.float32)
    poly = torch.stack(poly_chunks).numpy().astype(np.float32)
    return lcu, poly


@torch.no_grad()
def _quantum_diagnostics(model, loader) -> tuple[dict, np.ndarray, dict]:
    """Capture hypernetwork gate-parameter distributions, state norms, and
    per-mode photon numbers on a fixed diagnostic subset.

    Returns:
        stats_summary:  JSON-friendly dict (mean/std/min/max per head per gate type).
        mean_photon:    (num_heads, num_modes) float32 — ⟨n̂_k⟩.
        raw:            dict of large numpy arrays (gate-param samples + state norms)
                        to be saved as an .npz alongside the run directory.
    """
    from cv_quixer.quantum import FockState  # local import to avoid top-level cycles

    model.eval()
    attn = model.cv_attention
    num_heads = len(attn.heads)
    num_modes = attn.num_modes
    cutoff = attn.cutoff_dim
    head0 = attn.heads[0]
    n_bs = len(head0._bs_pairs)
    layout = _gate_param_layout(num_modes, n_bs)

    gate_outs: list[list[torch.Tensor]] = [[] for _ in range(num_heads)]
    state_norm_chunks: list[list[float]] = [[] for _ in range(num_heads)]
    photon_sums = np.zeros((num_heads, num_modes), dtype=np.float64)
    n_processed = 0

    for patches, _ in loader:
        patches = patches.to(device)
        B, N, _ = patches.shape
        n_processed += B

        for h_idx, head in enumerate(attn.heads):
            outs_bn: list[torch.Tensor] = []
            for b in range(B):
                for i in range(N):
                    outs_bn.append(head.hypernetwork(patches[b, i], i).detach().cpu())
            gate_outs[h_idx].append(torch.stack(outs_bn).reshape(B, N, -1))

        out = model(patches, return_states=True)
        states_list = out[1]
        for h_idx, state_batch in enumerate(states_list):
            circuit = attn.heads[h_idx].circuit
            for b in range(B):
                fs = FockState(state_batch[b], num_modes, cutoff)
                for k in range(num_modes):
                    photon_sums[h_idx, k] += float(
                        circuit.measure_photon_number(k, fs).item()
                    )
                state_norm_chunks[h_idx].append(float(fs.norm().item()))

    mean_photon = (photon_sums / max(n_processed, 1)).astype(np.float32)

    stats_summary: dict = {"per_head": [], "gate_layout": layout}
    raw: dict = {}
    for h_idx in range(num_heads):
        flat = torch.cat(gate_outs[h_idx], dim=0).numpy()  # (S, N, P)
        flat2d = flat.reshape(-1, flat.shape[-1])           # (S*N, P)
        gate_stats: dict = {}
        for name, start, count in layout:
            if count == 0:
                gate_stats[name] = {"mean": [], "std": [], "min": [], "max": []}
                continue
            sl = flat2d[:, start:start + count]
            gate_stats[name] = {
                "mean": sl.mean(axis=0).astype(float).tolist(),
                "std":  sl.std(axis=0).astype(float).tolist(),
                "min":  sl.min(axis=0).astype(float).tolist(),
                "max":  sl.max(axis=0).astype(float).tolist(),
            }
            raw[f"head{h_idx}_{name}"] = sl.astype(np.float32)
        stats_summary["per_head"].append(gate_stats)
        raw[f"head{h_idx}_state_norms"] = np.asarray(
            state_norm_chunks[h_idx], dtype=np.float32
        )
    raw["mean_photon_number"] = mean_photon
    return stats_summary, mean_photon, raw


def train_epoch(epoch: int) -> float:
    """One epoch of training. Returns the mean per-sample truncation loss
    measured across training batches (cheap in-epoch summary, no extra pass).

    The epoch-level CE loss and accuracy are NOT computed here — they are
    produced after this function returns by running the post-epoch model on
    the full training set in eval mode, which avoids the running-average
    bias of mixing early-batch (under-trained) and late-batch predictions.

    Per-batch CE / trunc / total loss / batch acc / grad norm are still
    appended to history["batch"] so the per-batch diagnostic plots are
    unchanged.
    """
    global global_step
    model.train()
    total_trunc, total = 0.0, 0
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
        batch_acc = (preds == labels).sum().item() / n

        total_trunc += trunc_loss.item() * n
        total       += n

        global_step += 1
        history["batch"]["step"].append(global_step)
        history["batch"]["epoch"].append(epoch)
        history["batch"]["train_loss"].append(float(ce_loss.item()))
        history["batch"]["trunc_loss"].append(float(trunc_loss.item()))
        history["batch"]["total_loss"].append(float(loss.item()))
        history["batch"]["batch_acc"].append(float(batch_acc))
        history["batch"]["grad_norm"].append(float(grad_norm))

    return total_trunc / total


@torch.no_grad()
def evaluate(loader, num_classes: int = 10) -> dict:
    """Run one evaluation pass; return loss / accuracy plus per-sample and
    per-class diagnostics.

    Returns a dict with keys:
        loss           — mean cross-entropy
        acc            — overall accuracy
        trunc_loss     — mean per-sample truncation loss (eval-time)
        y_true         — int64 (N,) ground-truth labels
        y_pred         — int64 (N,) argmax predictions
        y_probs        — float32 (N, num_classes) softmax probabilities
        per_class_acc  — float32 (num_classes,) recall per class
        confusion      — int64 (num_classes, num_classes) — rows true, cols pred
        readouts       — float32 (N, num_heads * readout_per_head) pre-decoder
                         activations (used for the post-hoc t-SNE plot)
    """
    model.eval()
    total_loss, total_trunc, correct, total = 0.0, 0.0, 0, 0
    y_true_chunks: list[torch.Tensor] = []
    y_pred_chunks: list[torch.Tensor] = []
    y_prob_chunks: list[torch.Tensor] = []
    readout_chunks: list[torch.Tensor] = []
    for patches, labels in loader:
        patches, labels = patches.to(device), labels.to(device)
        out = model(patches, return_trunc_loss=True, return_readouts=True)
        # Forward returns (logits, [trunc_loss], readouts); trunc_loss is omitted
        # when quantum_cfg.trunc_penalty == "none", so unpack defensively.
        if len(out) == 3:
            logits, trunc_loss, readouts = out
            total_trunc += trunc_loss.item() * labels.size(0)
        else:
            logits, readouts = out
        total_loss += F.cross_entropy(logits, labels).item() * labels.size(0)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
        y_true_chunks.append(labels.detach().cpu())
        y_pred_chunks.append(preds.detach().cpu())
        y_prob_chunks.append(F.softmax(logits, dim=-1).detach().cpu().float())
        readout_chunks.append(readouts.detach().cpu().float())

    y_true = torch.cat(y_true_chunks).numpy().astype(np.int64)
    y_pred = torch.cat(y_pred_chunks).numpy().astype(np.int64)
    y_probs = torch.cat(y_prob_chunks).numpy().astype(np.float32)
    readouts_np = torch.cat(readout_chunks).numpy().astype(np.float32)

    flat = y_true.astype(np.int64) * num_classes + y_pred.astype(np.int64)
    confusion = np.bincount(flat, minlength=num_classes * num_classes).reshape(
        num_classes, num_classes
    ).astype(np.int64)
    class_totals = confusion.sum(axis=1)
    per_class_acc = np.where(
        class_totals > 0,
        confusion.diagonal() / np.maximum(class_totals, 1),
        0.0,
    ).astype(np.float32)

    return {
        "loss":          total_loss / total,
        "acc":           correct / total,
        "trunc_loss":    total_trunc / total,
        "y_true":        y_true,
        "y_pred":        y_pred,
        "y_probs":       y_probs,
        "per_class_acc": per_class_acc,
        "confusion":     confusion,
        "readouts":      readouts_np,
    }


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


_FASHIONMNIST_CLASSES = (
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
)
_MNIST_CLASSES = tuple(str(d) for d in range(10))


def _class_names() -> tuple[str, ...]:
    return (_MNIST_CLASSES if data_cfg.dataset == "mnist"
            else _FASHIONMNIST_CLASSES)


def save_figures() -> None:
    epoch_x = list(range(1, len(history["epoch"]["train_loss"]) + 1))
    eh = history["epoch"]
    bh = history["batch"]
    classes = _class_names()
    num_classes = len(classes)

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

    # Per-epoch trunc loss (train + test side-by-side if test is available)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(epoch_x, eh["trunc_loss"], label="train trunc loss",
            color="tab:orange", marker="o")
    if eh.get("test_trunc_loss"):
        ax.plot(epoch_x[:len(eh["test_trunc_loss"])], eh["test_trunc_loss"],
                label="test trunc loss", color="tab:blue", marker="s")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Mean per-patch truncation loss")
    ax.set_title("Truncation loss (per epoch)"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(fig_dir / "trunc_loss_curve.png", dpi=150); plt.close(fig)

    # Per-class accuracy curve (test)
    if eh.get("test_per_class_acc"):
        fig, ax = plt.subplots(figsize=(9, 5.5))
        per_class = np.asarray(eh["test_per_class_acc"])    # (n_epochs, C)
        ep_x = list(range(1, per_class.shape[0] + 1))
        cmap = matplotlib.colormaps.get_cmap("tab10").resampled(num_classes)
        for c in range(per_class.shape[1]):
            ax.plot(ep_x, per_class[:, c], marker="o", color=cmap(c),
                    label=f"{c}: {classes[c]}")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Test accuracy (recall)")
        ax.set_title("Per-class test accuracy across epochs")
        ax.set_ylim(-0.02, 1.02); ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=8, ncol=2)
        fig.tight_layout()
        fig.savefig(fig_dir / "per_class_accuracy_curve.png", dpi=150)
        plt.close(fig)

    # Latest-epoch test confusion matrix (counts + row-normalised)
    if eh.get("test_confusion"):
        cm = np.asarray(eh["test_confusion"][-1], dtype=np.int64)
        cm_row = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
        for ax_, mat, title, fmt, cmap_name in (
            (axes[0], cm,     "Counts",          "d",   "Blues"),
            (axes[1], cm_row, "Row-normalised",  ".2f", "Blues"),
        ):
            im = ax_.imshow(mat, cmap=cmap_name)
            ax_.set_title(f"{title} (epoch {len(eh['test_confusion'])})")
            ax_.set_xticks(range(num_classes))
            ax_.set_yticks(range(num_classes))
            ax_.set_xticklabels(classes, rotation=45, ha="right", fontsize=8)
            ax_.set_yticklabels(classes, fontsize=8)
            ax_.set_xlabel("Predicted"); ax_.set_ylabel("True")
            for i in range(num_classes):
                for j in range(num_classes):
                    val = mat[i, j]
                    txt = format(val, fmt)
                    colour = "white" if (val > mat.max() * 0.6) else "black"
                    ax_.text(j, i, txt, ha="center", va="center",
                             color=colour, fontsize=7)
            fig.colorbar(im, ax=ax_, fraction=0.046, pad=0.04)
        fig.suptitle("Test confusion matrix")
        fig.tight_layout()
        fig.savefig(fig_dir / "confusion_matrix_test.png", dpi=150)
        plt.close(fig)

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
    trunc_loss = train_epoch(epoch)
    train_eval = evaluate(train_eval_loader)
    test_eval  = evaluate(test_loader)
    elapsed_train_eval = time.time() - t0

    train_loss, train_acc = train_eval["loss"], train_eval["acc"]
    test_loss,  test_acc  = test_eval["loss"],  test_eval["acc"]

    history["epoch"]["train_loss"].append(train_loss)
    history["epoch"]["train_acc"].append(train_acc)
    history["epoch"]["test_loss"].append(test_loss)
    history["epoch"]["test_acc"].append(test_acc)
    history["epoch"]["trunc_loss"].append(trunc_loss)
    history["epoch"]["test_trunc_loss"].append(float(test_eval["trunc_loss"]))
    history["epoch"]["train_per_class_acc"].append(
        train_eval["per_class_acc"].tolist()
    )
    history["epoch"]["test_per_class_acc"].append(
        test_eval["per_class_acc"].tolist()
    )
    history["epoch"]["train_confusion"].append(train_eval["confusion"].tolist())
    history["epoch"]["test_confusion"].append(test_eval["confusion"].tolist())

    np.savez_compressed(
        preds_dir / f"epoch_{epoch:04d}.npz",
        y_true=test_eval["y_true"],
        y_pred=test_eval["y_pred"],
        y_probs=test_eval["y_probs"],
        readouts=test_eval["readouts"],
    )

    lcu_snap, poly_snap = _snapshot_coefficients(model)
    history["epoch"]["lcu_coeffs"].append(lcu_snap.tolist())
    history["epoch"]["poly_coeffs"].append(poly_snap.tolist())

    t_diag = time.time()
    try:
        stats_summary, mean_photon, diag_raw = _quantum_diagnostics(model, diag_loader)
        history["epoch"]["hypernet_stats"].append(stats_summary)
        history["epoch"]["mean_photon_number"].append(mean_photon.tolist())
        np.savez_compressed(diag_dir / f"epoch_{epoch:04d}.npz", **diag_raw)
        diag_status = f"  (diag {time.time() - t_diag:.1f}s)"
    except Exception as e:
        # Don't let a diagnostic-pass failure kill a multi-hour training run.
        import warnings
        warnings.warn(
            f"_quantum_diagnostics failed at epoch {epoch}: {type(e).__name__}: {e}. "
            "Skipping diagnostic outputs for this epoch.",
            RuntimeWarning,
        )
        history["epoch"]["hypernet_stats"].append(None)
        history["epoch"]["mean_photon_number"].append(None)
        diag_status = "  (diag skipped)"

    elapsed = time.time() - t0
    print(f"{epoch:<7} {train_loss:<12.4f} {train_acc:<11.3f} "
          f"{test_loss:<11.4f} {test_acc:<11.3f} {trunc_loss:<12.4f} "
          f"{elapsed:.1f}s{diag_status}")

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

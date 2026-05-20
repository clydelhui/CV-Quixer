"""Shared evaluation + diagnostic helpers for the CV-Quixer pipeline.

Single source of truth for the per-pass evaluation and quantum diagnostic
captures consumed by both the training producer (experiments/full_experiment.py)
and the post-hoc cutoff-sweep evaluator (experiments/eval_cutoff_sweep.py).
Outputs are the artefacts that experiments/report_diagnostics.py consumes:
`predictions/epoch_NNNN.npz` (y_true,y_pred,y_probs,readouts) and
`diagnostics/epoch_NNNN.npz` (per-head gate-param samples, state norms,
per-mode mean photon number), plus the history-schema factory.

This module imports torch at top level; report_diagnostics.py must NOT import
from here at module scope (use cv_quixer.evaluation.labels instead).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# History schema
# ---------------------------------------------------------------------------


EPOCH_KEYS: tuple[str, ...] = (
    "train_per_class_acc", "test_per_class_acc",
    "train_confusion",     "test_confusion",
    "test_trunc_loss",
    "lcu_coeffs", "poly_coeffs",
    "hypernet_stats", "mean_photon_number",
)


def ensure_history_schema(h: dict) -> None:
    """Idempotently add any missing diagnostic fields to a (possibly resumed)
    history dict. Safe to call on a fresh `new_history()` result or on a
    resumed checkpoint payload from an older schema."""
    h.setdefault("epoch", {})
    for k in EPOCH_KEYS:
        h["epoch"].setdefault(k, [])


def new_history(n_params: int, device: str) -> dict:
    """Build a fresh history dict with the full `{epoch, batch, meta}` skeleton.

    `started_at` is captured at call time. The returned dict already contains
    both the base epoch keys (train/test loss/acc, trunc_loss, elapsed_sec) and
    the extended `EPOCH_KEYS` (per-class accuracy, confusion, lcu/poly coeffs,
    hypernet stats, mean photon number).
    """
    h = {
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
    ensure_history_schema(h)
    return h


# ---------------------------------------------------------------------------
# Subset-indices reader
# ---------------------------------------------------------------------------


def load_subset_indices(run_dir: Path) -> dict[str, np.ndarray]:
    """Read `<run_dir>/subset_indices.npz` written by
    `experiments/full_experiment._save_subset_indices_once`.

    Returns a dict with `train_indices`, `test_indices`, `diag_indices`
    (all absolute indices into the full PatchedDataset of each split).
    Raises FileNotFoundError with a clear message if the file is absent.
    """
    path = Path(run_dir) / "subset_indices.npz"
    if not path.is_file():
        raise FileNotFoundError(
            f"subset_indices.npz not found at {path}. This run directory was "
            "not produced by a recent experiments/full_experiment.py run; the "
            "post-hoc tools require it to reproduce the train/test/diag "
            "populations the model was evaluated on."
        )
    with np.load(path) as data:
        return {k: data[k].astype(np.int64) for k in data.files}


# ---------------------------------------------------------------------------
# Pre-decoder image reassembly
# ---------------------------------------------------------------------------


def save_test_images_once(loader, image_size: int, patch_size: int,
                          out_path: Path) -> None:
    """Reassemble the patches of every sample in `loader` into a (N, H, W) array
    and write it once. Idempotent: skip the work if the file already exists with
    a matching N. Used by experiments/report_diagnostics.py's default file-only
    mode to draw the misclassification gallery without re-running the model.
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


# ---------------------------------------------------------------------------
# Coefficient + gate-param helpers
# ---------------------------------------------------------------------------


def gate_param_layout(num_modes: int, n_bs: int) -> list[tuple[str, int, int]]:
    """Slice layout for hypernetwork gate-parameter outputs.

    Derived from cv_attention._GATE_SEQUENCE — single source of truth for the
    gate set. Each op contributes one (name, start, count) entry per parameter
    name; slice keys are f"{op.name}_{param_name}" to match the legacy names
    (squeeze_r, squeeze_phi, bs_theta, bs_phi, rot_phi, disp_re, disp_im,
    kerr_kappa).

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
def snapshot_coefficients(model) -> tuple[np.ndarray, np.ndarray]:
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


# ---------------------------------------------------------------------------
# Quantum diagnostics (gate-param distributions, state norms, photon numbers)
# ---------------------------------------------------------------------------


@torch.no_grad()
def quantum_diagnostics(model, loader, device) -> tuple[dict, np.ndarray, dict]:
    """Capture hypernetwork gate-parameter distributions, state norms, and
    per-mode photon numbers on the diagnostic subset carried by `loader`.

    Returns:
        stats_summary:  JSON-friendly dict (mean/std/min/max per head per gate type).
        mean_photon:    (num_heads, num_modes) float32 — ⟨n̂_k⟩.
        raw:            dict of large numpy arrays (gate-param samples + state norms
                        + mean_photon_number) to be saved as an .npz under
                        diagnostics/epoch_NNNN.npz.
    """
    from cv_quixer.quantum import FockState  # local import to avoid top-level cycles

    model.eval()
    attn = model.cv_attention
    num_heads = len(attn.heads)
    num_modes = attn.num_modes
    cutoff = attn.cutoff_dim
    head0 = attn.heads[0]
    n_bs = len(head0._bs_pairs)
    layout = gate_param_layout(num_modes, n_bs)

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
        states_list = out.states
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


# ---------------------------------------------------------------------------
# Per-pass evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate(model, loader, device, num_classes: int = 10) -> dict:
    """Run one evaluation pass; return loss / accuracy plus per-sample and
    per-class diagnostics.

    Returns a dict with keys:
        loss           — mean cross-entropy
        acc            — overall accuracy
        trunc_loss     — mean per-sample truncation loss (eval-time; 0 if the
                         model's trunc_penalty is "none")
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
        # out.trunc_loss is None when the model's trunc_penalty == "none".
        logits, readouts = out.logits, out.readouts
        if out.trunc_loss is not None:
            total_trunc += out.trunc_loss.item() * labels.size(0)
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

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


# Derived per-epoch fields are no longer carried in history.json; report_diagnostics
# reads them from predictions/diagnostics npz instead. EPOCH_KEYS lists the
# training-time-only log fields that ensure_history_schema guarantees.
EPOCH_KEYS: tuple[str, ...] = (
    "train_loss", "train_acc",
    "test_loss",  "test_acc",
    "trunc_loss", "test_trunc_loss",
    "cvqnn_trunc_loss", "test_cvqnn_trunc_loss",
    "query_trunc_loss", "test_query_trunc_loss",
    "elapsed_sec",
)

BATCH_KEYS: tuple[str, ...] = (
    "step", "epoch",
    "train_loss", "trunc_loss", "cvqnn_trunc_loss", "query_trunc_loss",
    "total_loss", "batch_acc", "grad_norm",
)


def ensure_history_schema(h: dict) -> None:
    """Idempotently ensure the training-log schema. Safe to call on a fresh
    `new_history()` result or on a resumed checkpoint payload from an older
    schema."""
    h.setdefault("epoch", {})
    for k in EPOCH_KEYS:
        h["epoch"].setdefault(k, [])
    h.setdefault("batch", {})
    for k in BATCH_KEYS:
        h["batch"].setdefault(k, [])


def new_history(n_params: int, device: str) -> dict:
    """Build a fresh history dict with the `{epoch, batch, meta}` skeleton.

    `started_at` is captured at call time. `history["epoch"]` carries only
    the training-time log fields (loss/acc, trunc_loss, elapsed_sec); all
    derived metrics live in the per-epoch predictions/diagnostics npz files.
    """
    h = {
        "epoch": {k: [] for k in EPOCH_KEYS},
        "batch": {k: [] for k in BATCH_KEYS},
        "meta": {
            "best_test_acc": None, "best_epoch": None,
            "total_runtime_sec": None, "n_params": int(n_params),
            "device": str(device),
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "completed_at": None,
        },
    }
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
                          out_path: Path,
                          *, progress: bool | str = False) -> None:
    """Reassemble the patches of every sample in `loader` into a (N, H, W) array
    and write it once. Idempotent: skip the work if the file already exists with
    a matching N. Used by experiments/report_diagnostics.py's default file-only
    mode to draw the misclassification gallery without re-running the model.

    Pass ``progress=True`` (or a string label) to wrap the loader iteration in
    a tqdm bar — useful when the loader has the full 10k+ test set and the
    function is otherwise silent.
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
    iterator = loader
    if progress:
        from tqdm import tqdm
        desc = progress if isinstance(progress, str) else "reassembling images"
        iterator = tqdm(loader, desc=desc, leave=False, unit="batch",
                        mininterval=5.0)
    for patches, lbls in iterator:
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


def gate_param_layout(
    num_modes: int, n_bs: int, num_layers: int = 1
) -> list[tuple[str, int, int]]:
    """Slice layout for hypernetwork gate-parameter outputs.

    Derived from the per-patch op-plan (cv_attention._GATE_SEQUENCE interleaved
    with _INTERFEROMETER_SEQUENCE — the single source of truth for the gate
    set). Each op contributes one (name, start, count) entry per parameter name.

    Keys preserve the legacy names for the first layer (squeeze_r, squeeze_phi,
    bs_theta, bs_phi, rot_phi, disp_re, disp_im, kerr_kappa), so at
    num_layers == 1 the layout — and therefore the saved npz keys — are
    byte-identical to the single-layer model. For num_layers > 1, later blocks
    are prefixed: ``L{l}_`` for layer l (l>=1) and ``I{l}_`` for the
    interferometer that follows layer l.

    Returns list of (name, start, count); count==0 means the gate is absent.
    """
    from cv_quixer.models.quantum.cv_attention import (
        _GATE_SEQUENCE,
        _INTERFEROMETER_SEQUENCE,
    )

    layout: list[tuple[str, int, int]] = []
    offset = 0

    def _emit(seq, prefix: str) -> None:
        nonlocal offset
        for op in seq:
            n_sites = num_modes if op.site_kind == "mode" else n_bs
            for p in op.param_names:
                layout.append((f"{prefix}{op.name}_{p}", offset, n_sites))
                offset += n_sites

    for layer in range(num_layers):
        _emit(_GATE_SEQUENCE, "" if layer == 0 else f"L{layer}_")
        if layer < num_layers - 1:
            _emit(_INTERFEROMETER_SEQUENCE, f"I{layer}_")
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


def snapshot_cvqnn_params(model) -> np.ndarray | None:
    """CVQNN block (W) gate params per head — JSON-friendly array, or None.

    Returns:
        (num_heads, cvqnn_param_count) float32 — the learned W gate-param vector
        per head. ``None`` when the model has no W block (cvqnn_num_layers == 0,
        each head's ``cvqnn_params`` is None), so callers can skip writing the key.
    """
    chunks = []
    for head in model.cv_attention.heads:
        params = getattr(head, "cvqnn_params", None)
        if params is None:
            return None
        chunks.append(params.detach().cpu())
    return torch.stack(chunks).numpy().astype(np.float32)


def snapshot_stacked_coefficients(model) -> dict[str, np.ndarray]:
    """Block-prefixed LCU/poly/W snapshots for the stacked model (ADR-0003).

    One key set per stage: ``block{b}_lcu_coeffs`` / ``block{b}_poly_coeffs``
    (+ ``block{b}_cvqnn_params`` when W is on) for each seq-to-seq block, and
    the same under the ``agg_`` prefix for the aggregator (pooling="quixer").
    Per-stage array shapes match the canonical ``snapshot_coefficients`` /
    ``snapshot_cvqnn_params`` outputs.
    """
    out: dict[str, np.ndarray] = {}

    def _stage(prefix: str, heads) -> None:
        lcu_chunks, poly_chunks, cvqnn_chunks = [], [], []
        for head in heads:
            b = head.lcu_coeffs().detach().cpu()
            lcu_chunks.append(torch.stack([b.real, b.imag], dim=-1))
            poly_chunks.append(head.poly_coeffs().detach().cpu())
            if head.cvqnn_params is not None:
                cvqnn_chunks.append(head.cvqnn_params.detach().cpu())
        out[f"{prefix}_lcu_coeffs"] = (
            torch.stack(lcu_chunks).numpy().astype(np.float32)
        )
        out[f"{prefix}_poly_coeffs"] = (
            torch.stack(poly_chunks).numpy().astype(np.float32)
        )
        if cvqnn_chunks:
            out[f"{prefix}_cvqnn_params"] = (
                torch.stack(cvqnn_chunks).numpy().astype(np.float32)
            )

    for b_idx, block in enumerate(model.blocks):
        _stage(f"block{b_idx}", block.heads)
    if model.aggregator_heads is not None:
        _stage("agg", model.aggregator_heads)
    return out


# ---------------------------------------------------------------------------
# Quantum diagnostics (gate-param distributions, state norms, photon numbers)
# ---------------------------------------------------------------------------


def _state_stats(state_batch: torch.Tensor, num_modes: int,
                 cutoff: int) -> tuple[torch.Tensor, np.ndarray]:
    """Per-element state norms + per-mode photon-number sums from a batched
    state tensor whose trailing ``num_modes`` axes are the Fock axes. Any
    leading batch-like axes are flattened into the sample population (the
    stacked model's per-position states fold in this way, ADR-0003).

    Returns:
        norms:       (S,) real tensor — ‖ψ‖² per flattened element.
        photon_sums: (num_modes,) float64 — Σ_elements ⟨n̂_k⟩.
    """
    probs = (state_batch.abs() ** 2).reshape(-1, *([cutoff] * num_modes))
    ns = torch.arange(cutoff, device=probs.device, dtype=probs.dtype)
    photon_sums = np.zeros(num_modes, dtype=np.float64)
    for k in range(num_modes):
        other_axes = tuple(
            ax for ax in range(1, num_modes + 1) if ax != k + 1
        )
        p_k = probs.sum(dim=other_axes) if other_axes else probs   # (S, D)
        photon_sums[k] = float((p_k * ns).sum().item())
    norms = probs.flatten(start_dim=1).sum(dim=-1)                 # (S,)
    return norms, photon_sums


@torch.no_grad()
def quantum_diagnostics(model, loader, device,
                        *, progress: bool | str = False) -> tuple[dict, np.ndarray, dict]:
    """Capture hypernetwork gate-parameter distributions, state norms, and
    per-mode photon numbers on the diagnostic subset carried by `loader`.

    Pass ``progress=True`` or a string label to wrap the outer loader loop
    in a tqdm bar.

    Returns:
        stats_summary:  JSON-friendly dict (mean/std/min/max per head per gate type).
        mean_photon:    (num_heads, num_modes) float32 — ⟨n̂_k⟩.
        raw:            dict of large numpy arrays (gate-param samples + state norms
                        + mean_photon_number) to be saved as an .npz under
                        diagnostics/epoch_NNNN.npz.
    """
    # The stacked model has per-block heads and a different artefact schema
    # (block-prefixed keys, ADR-0003) — dispatch to its own collector.
    if getattr(model, "blocks", None) is not None:
        return _stacked_quantum_diagnostics(
            model, loader, device, progress=progress
        )

    model.eval()
    attn = model.cv_attention
    num_heads = len(attn.heads)
    num_modes = attn.num_modes
    cutoff = attn.cutoff_dim
    head0 = attn.heads[0]
    n_bs = len(head0._bs_pairs)
    num_layers = getattr(head0, "num_layers", 1)
    layout = gate_param_layout(num_modes, n_bs, num_layers)

    gate_outs: list[list[torch.Tensor]] = [[] for _ in range(num_heads)]
    state_norm_chunks: list[list[float]] = [[] for _ in range(num_heads)]
    photon_sums = np.zeros((num_heads, num_modes), dtype=np.float64)
    n_processed = 0

    iterator = loader
    if progress:
        from tqdm import tqdm
        desc = progress if isinstance(progress, str) else "diagnostics"
        iterator = tqdm(loader, desc=desc, leave=False, unit="batch",
                        mininterval=5.0)
    for patches, _ in iterator:
        patches = patches.to(device)
        B, N, _ = patches.shape
        n_processed += B

        # Per-head gate-parameter samples over the full (B, N, patch_dim)
        # tensor. gate_params_grid is model-agnostic: the canonical model runs
        # each head's CNN hypernetwork; the shared-CNN model embeds once then
        # applies each head's linear.
        for h_idx, gp in enumerate(attn.gate_params_grid(patches)):
            gate_outs[h_idx].append(gp.detach().cpu())

        out = model(patches, return_states=True)
        states_list = out.states
        # Photon number per mode and per-element state norm, computed
        # directly from the batched state tensor (no per-element FockState
        # construction, no reduced-density-matrix build).
        for h_idx, state_batch in enumerate(states_list):
            norms, photon = _state_stats(state_batch, num_modes, cutoff)
            photon_sums[h_idx] += photon
            state_norm_chunks[h_idx].extend(
                float(v) for v in norms.detach().cpu().tolist()
            )

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


@torch.no_grad()
def _stacked_quantum_diagnostics(model, loader, device,
                                 *, progress: bool | str = False
                                 ) -> tuple[dict, np.ndarray, dict]:
    """Stacked-model collector behind ``quantum_diagnostics`` (ADR-0003).

    Gate-param samples are captured **per stage** with block-prefixed keys —
    ``block{b}_head{h}_{gate}`` for the key slice, ``block{b}_head{h}_q_{gate}``
    for the query slice, and ``agg_head{h}_{gate}`` for the aggregator (no
    query slice). State norms / photon numbers describe the decoder-input
    stage and keep the canonical key names (``head{h}_state_norms``,
    ``mean_photon_number``); under pooling="mean" the per-position states fold
    into the sample population.
    """
    model.eval()
    cfg = model.config
    num_heads, num_modes, cutoff = cfg.num_heads, cfg.num_modes, cfg.cutoff_dim
    head0 = model.blocks[0].heads[0]
    layout = gate_param_layout(num_modes, len(head0._bs_pairs), head0.num_layers)
    gp_width = head0._gate_param_width
    num_blocks = len(model.blocks)
    has_agg = model.aggregator_heads is not None
    stage_prefixes = [f"block{b}" for b in range(num_blocks)] + (
        ["agg"] if has_agg else []
    )

    gate_outs: dict[tuple[int, int], list[torch.Tensor]] = {
        (s, h): [] for s in range(len(stage_prefixes)) for h in range(num_heads)
    }
    state_norm_chunks: list[list[float]] = [[] for _ in range(num_heads)]
    photon_sums = np.zeros((num_heads, num_modes), dtype=np.float64)
    n_processed = 0

    iterator = loader
    if progress:
        from tqdm import tqdm
        desc = progress if isinstance(progress, str) else "diagnostics"
        iterator = tqdm(loader, desc=desc, leave=False, unit="batch",
                        mininterval=5.0)
    for patches, _ in iterator:
        patches = patches.to(device)

        # Per-stage gate-param samples; deeper stages see the actual token
        # sequence their block receives (block_inputs replays the stack once).
        inputs = model.block_inputs(patches)
        for b_idx, block in enumerate(model.blocks):
            for h_idx, gpar in enumerate(block.gate_params_grid(inputs[b_idx])):
                gate_outs[(b_idx, h_idx)].append(gpar.detach().cpu())
        if has_agg:
            for h_idx, head in enumerate(model.aggregator_heads):
                gate_outs[(num_blocks, h_idx)].append(
                    head._features_to_params(inputs[-1]).detach().cpu()
                )

        out = model(patches, return_states=True)
        for h_idx, state_batch in enumerate(out.states):
            norms, photon = _state_stats(state_batch, num_modes, cutoff)
            photon_sums[h_idx] += photon
            state_norm_chunks[h_idx].extend(
                float(v) for v in norms.detach().cpu().tolist()
            )
    # Photon means are over the same flattened population the norms use
    # (B per batch for aggregator states, B×N for per-position states).
    n_processed = len(state_norm_chunks[0])
    mean_photon = (photon_sums / max(n_processed, 1)).astype(np.float32)

    def _slice_stats(flat2d: np.ndarray) -> tuple[dict, dict[str, np.ndarray]]:
        """Per-gate stats + raw slices for one (samples, width) param matrix."""
        gate_stats: dict = {}
        raw_slices: dict[str, np.ndarray] = {}
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
            raw_slices[name] = sl.astype(np.float32)
        return gate_stats, raw_slices

    stats_summary: dict = {"gate_layout": layout, "per_stage": {}}
    raw: dict = {}
    for s_idx, prefix in enumerate(stage_prefixes):
        per_head_stats = []
        for h_idx in range(num_heads):
            flat = torch.cat(gate_outs[(s_idx, h_idx)], dim=0).numpy()
            flat2d = flat.reshape(-1, flat.shape[-1])
            # Seq-to-seq stages emit (key | query) slices; the aggregator emits
            # a single canonical slice.
            key_stats, key_raw = _slice_stats(flat2d[:, :gp_width])
            for name, sl in key_raw.items():
                raw[f"{prefix}_head{h_idx}_{name}"] = sl
            head_stats = {"key": key_stats}
            if flat2d.shape[-1] == 2 * gp_width:
                q_stats, q_raw = _slice_stats(flat2d[:, gp_width:])
                for name, sl in q_raw.items():
                    raw[f"{prefix}_head{h_idx}_q_{name}"] = sl
                head_stats["query"] = q_stats
            per_head_stats.append(head_stats)
        stats_summary["per_stage"][prefix] = per_head_stats

    for h_idx in range(num_heads):
        raw[f"head{h_idx}_state_norms"] = np.asarray(
            state_norm_chunks[h_idx], dtype=np.float32
        )
    raw["mean_photon_number"] = mean_photon
    return stats_summary, mean_photon, raw


# ---------------------------------------------------------------------------
# Per-pass evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate(model, loader, device, num_classes: int = 10,
             *, progress: bool | str = False) -> dict:
    """Run one evaluation pass; return loss / accuracy plus per-sample and
    per-class diagnostics.

    Pass ``progress=True`` or a string label to wrap the loader iteration in
    a tqdm bar.

    Returns a dict with keys:
        loss           — mean cross-entropy
        acc            — overall accuracy
        trunc_loss     — mean per-sample truncation loss (eval-time; 0 if the
                         model's trunc_penalty is "none")
        cvqnn_trunc_loss — mean per-sample CVQNN block (W) truncation leakage
                         (eval-time; 0 if cvqnn_num_layers == 0)
        query_trunc_loss — mean per-sample query-unitary truncation leakage
                         (eval-time; 0 for models without query unitaries —
                         only the seq-to-seq stacked model has the stream)
        y_true         — int64 (N,) ground-truth labels
        y_pred         — int64 (N,) argmax predictions
        y_probs        — float32 (N, num_classes) softmax probabilities
        per_class_acc  — float32 (num_classes,) recall per class
        confusion      — int64 (num_classes, num_classes) — rows true, cols pred
        readouts       — float32 (N, num_heads * readout_per_head) pre-decoder
                         activations (used for the post-hoc t-SNE plot)
        success_probs  — float32 (N, num_heads) raw per-sample LCU/QSVT
                         post-selection norms ‖P(M)|ψ⟩‖² per head. Raw, i.e.
                         not divided by the subnormalisation λ² (derive λ from
                         lcu_coeffs/poly_coeffs; see ADR-0002). Key absent for
                         models without LCU post-selection.
    """
    model.eval()
    total_loss, total_trunc, total_cvqnn_trunc, correct, total = 0.0, 0.0, 0.0, 0, 0
    total_query_trunc = 0.0
    y_true_chunks: list[torch.Tensor] = []
    y_pred_chunks: list[torch.Tensor] = []
    y_prob_chunks: list[torch.Tensor] = []
    readout_chunks: list[torch.Tensor] = []
    sp_chunks: list[torch.Tensor] = []
    iterator = loader
    if progress:
        from tqdm import tqdm
        desc = progress if isinstance(progress, str) else "eval"
        iterator = tqdm(loader, desc=desc, leave=False, unit="batch",
                        mininterval=5.0)
    for patches, labels in iterator:
        patches, labels = patches.to(device), labels.to(device)
        out = model(patches, return_trunc_loss=True, return_readouts=True,
                    return_success_prob=True)
        # out.trunc_loss is None when the model's trunc_penalty == "none".
        logits, readouts = out.logits, out.readouts
        if out.trunc_loss is not None:
            total_trunc += out.trunc_loss.item() * labels.size(0)
        # out.cvqnn_trunc_loss is independent of trunc_penalty; None only on a
        # non-CV model (no return_trunc_loss support) — guard anyway.
        if out.cvqnn_trunc_loss is not None:
            total_cvqnn_trunc += out.cvqnn_trunc_loss.item() * labels.size(0)
        # out.query_trunc_loss is None for models without query unitaries
        # (only the seq-to-seq stacked model carries the stream).
        if out.query_trunc_loss is not None:
            total_query_trunc += out.query_trunc_loss.item() * labels.size(0)
        total_loss += F.cross_entropy(logits, labels).item() * labels.size(0)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
        y_true_chunks.append(labels.detach().cpu())
        y_pred_chunks.append(preds.detach().cpu())
        y_prob_chunks.append(F.softmax(logits, dim=-1).detach().cpu().float())
        readout_chunks.append(readouts.detach().cpu().float())
        # out.success_probs is None on models without LCU post-selection.
        if out.success_probs is not None:
            # list of H × (B,) → (B, H)
            sp_chunks.append(
                torch.stack(out.success_probs, dim=1).detach().cpu().float()
            )

    y_true = torch.cat(y_true_chunks).numpy().astype(np.int64)
    y_pred = torch.cat(y_pred_chunks).numpy().astype(np.int64)
    y_probs = torch.cat(y_prob_chunks).numpy().astype(np.float32)
    readouts_np = torch.cat(readout_chunks).numpy().astype(np.float32)
    success_probs_np = (
        torch.cat(sp_chunks).numpy().astype(np.float32) if sp_chunks else None
    )

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

    result = {
        "loss":          total_loss / total,
        "acc":           correct / total,
        "trunc_loss":    total_trunc / total,
        "cvqnn_trunc_loss": total_cvqnn_trunc / total,
        "query_trunc_loss": total_query_trunc / total,
        "y_true":        y_true,
        "y_pred":        y_pred,
        "y_probs":       y_probs,
        "per_class_acc": per_class_acc,
        "confusion":     confusion,
        "readouts":      readouts_np,
    }
    if success_probs_np is not None:
        result["success_probs"] = success_probs_np
    return result

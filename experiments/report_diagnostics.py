"""Post-hoc diagnostic figure generator for a `full_experiment.py` run.

The single figure-generation tool for the project. `full_experiment.py`
writes only raw data artefacts (per-epoch `predictions/*.npz` for both
train and test, `diagnostics/*.npz`, checkpoints, per-batch arrays in
`history.json`); every derived metric (accuracy, per-class accuracy,
confusion matrices, mean photon number, LCU/polynomial coefficient
snapshots) is recomputed here from those raw artefacts.

`history["epoch"]` is treated as a training-time log only — never as the
canonical source for plotted values. `plot_training_curves` derives
accuracy/loss from predictions npz and cross-checks against the logged
values, emitting a `RuntimeWarning` on mismatch.

Hard-fail policy: when a required per-epoch npz is missing, the
relevant figure function raises with a clear message directing the user
to run `experiments/backfill_artefacts.py --run-dir <run>`. The
top-level `main()` catches the exception and surfaces it as a warning so
the rest of the figures still render.

Usage:
    uv run python experiments/report_diagnostics.py \\
        --run-dir results/runs/full_fashionmnist_<timestamp>/ \\
        [--epoch best|final|<N>]

Multi-run sample-efficiency mode (scans results/runs/full_fashionmnist_*):
    uv run python experiments/report_diagnostics.py --multi-run
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from cv_quixer.config.schema import ExperimentConfig
from cv_quixer.config.utils import experiment_config_from_dict
from cv_quixer.evaluation import artefact_schema as schema
from cv_quixer.evaluation.labels import class_names

# Heavy / torch-dependent imports (build_model, PatchedDataset, DataLoader,
# torch itself) are intentionally moved into _load_model_and_run_inference()
# below so the default --no-flag path stays fast and works on machines without
# a configured PyTorch backend. cv_quixer.evaluation.labels is torch-free so
# importing class_names at module scope is safe.


# Moving-average window for the per-batch curves. Matches what the old
# full_experiment.save_figures used (kept here so that figure styling is
# unchanged between this script and any future producer).
MA_WINDOW = 50


# ---------------------------------------------------------------------------
# Loading + helpers
# ---------------------------------------------------------------------------


class MissingArtefactError(RuntimeError):
    """Raised when a required per-epoch npz file is missing. Carries a
    backfill-hint message so the user knows the canonical fix."""


def _require_predictions(run_dir: Path, epoch: int, side: str) -> dict:
    """Load `predictions/epoch_NNNN.npz` (side='test') or
    `predictions/epoch_NNNN_train.npz` (side='train'). Raises
    `MissingArtefactError` with a backfill hint when absent.
    """
    path = run_dir / "predictions" / schema.prediction_filename(
        epoch, train=(side != "test"))
    if not path.is_file():
        raise MissingArtefactError(
            f"required artefact missing: {path.relative_to(run_dir)} — "
            f"run `uv run python experiments/backfill_artefacts.py "
            f"--run-dir {run_dir}` to produce it."
        )
    return dict(np.load(path))


def _load_all_per_epoch_predictions(run_dir: Path, side: str = "test",
                                    n_epochs: int | None = None) -> dict[int, dict]:
    """Load every available `predictions/epoch_NNNN{,_train}.npz` for the
    given side, returning `{epoch: {y_true, y_pred, y_probs, readouts}}`.

    If `n_epochs` is given, the result must cover every epoch in
    `[1, n_epochs]` — missing epochs raise `MissingArtefactError`. If
    `n_epochs` is None, returns whatever exists.
    """
    suffix = "" if side == "test" else "_train"
    preds_dir = run_dir / "predictions"
    out: dict[int, dict] = {}
    if n_epochs is None:
        for path in sorted(preds_dir.glob(f"epoch_*{suffix}.npz")):
            # Skip test-side files when scanning train-side and vice versa
            # (the glob is ambiguous for side='test' since `_train` files
            # also match `epoch_*.npz`).
            stem = path.stem  # "epoch_0001" or "epoch_0001_train"
            if side == "test" and stem.endswith("_train"):
                continue
            if side == "train" and not stem.endswith("_train"):
                continue
            try:
                epoch = int(stem.split("_")[1])
            except (IndexError, ValueError):
                continue
            out[epoch] = dict(np.load(path))
        return out
    for e in range(1, n_epochs + 1):
        out[e] = _require_predictions(run_dir, e, side)
    return out


def _load_all_diagnostics(run_dir: Path,
                          n_epochs: int | None = None) -> dict[int, dict]:
    """Load every available `diagnostics/epoch_NNNN.npz`, returning a
    `{epoch: {key: array, ...}}` mapping. Behaves like
    `_load_all_per_epoch_predictions` w.r.t. the `n_epochs` flag.
    """
    diag_dir = run_dir / "diagnostics"
    out: dict[int, dict] = {}
    if n_epochs is None:
        for path in sorted(diag_dir.glob("epoch_*.npz")):
            try:
                epoch = int(path.stem.split("_")[1])
            except (IndexError, ValueError):
                continue
            out[epoch] = dict(np.load(path))
        return out
    for e in range(1, n_epochs + 1):
        path = diag_dir / schema.diagnostics_filename(e)
        if not path.is_file():
            raise MissingArtefactError(
                f"required artefact missing: {path.relative_to(run_dir)} — "
                f"run `uv run python experiments/backfill_artefacts.py "
                f"--run-dir {run_dir}` to produce it."
            )
        out[e] = dict(np.load(path))
    return out


# Post-selection failure threshold. Mirrors _SUCCESS_PROB_FLOOR in
# cv_quixer/models/quantum/cv_attention.py — duplicated here as a literal
# rather than imported so this script's default path stays free of torch-heavy
# imports. Keep the two values in sync.
_SUCCESS_PROB_FLOOR = 1e-6


def _derive_lcu_lambda(lcu_coeffs: np.ndarray,
                       poly_coeffs: np.ndarray) -> np.ndarray:
    """Per-head LCU/QSVT subnormalisation λ_h = Σ_j |c_hj| · (Σ_i |b_hi|)^j.

    The nested-LCU block-encoding scale of P(M) = Σ_j c_j M^j built from
    M = Σ_i b_i U_i (see ADR-0002). The physical post-selection success
    probability of a sample with raw norm s = ‖P(M)|ψ⟩‖² is s / λ².

    Args:
        lcu_coeffs:  (H, N, 2) — [real, imag] of b_i per head (raw, unnormalised).
        poly_coeffs: (H, d+1)  — real polynomial coefficients c_j per head.

    Returns:
        (H,) float64 — λ per head.
    """
    lcu = np.asarray(lcu_coeffs, dtype=np.float64)
    b_mag = np.sqrt((lcu ** 2).sum(axis=-1))                       # (H, N)
    alpha = b_mag.sum(axis=-1)                                     # (H,)
    c_abs = np.abs(np.asarray(poly_coeffs, dtype=np.float64))      # (H, d+1)
    powers = alpha[:, None] ** np.arange(c_abs.shape[1])[None, :]  # (H, d+1)
    return (c_abs * powers).sum(axis=-1)                           # (H,)


def _warn_mismatch(name: str, derived: float, logged: float | None,
                   tol: float) -> None:
    """Emit a `RuntimeWarning` if a derived per-epoch metric disagrees
    with the corresponding training-log entry by more than `tol`. A None
    logged value is treated as "no log" and silently ignored.
    """
    if logged is None:
        return
    if not np.isfinite(derived) or not np.isfinite(logged):
        return
    diff = abs(float(derived) - float(logged))
    if diff > tol:
        warnings.warn(
            f"{name}: derived={derived:.6f} vs logged={logged:.6f} "
            f"(|Δ|={diff:.2e} > tol={tol:.0e}). The npz-derived value is "
            "used for the figure; the logged value comes from "
            "history.json and may be stale.",
            RuntimeWarning,
        )


def _diag_stages(diag: dict) -> list[tuple[str, str, str]]:
    """Enumerate the per-stage diagnostics namespaces in one npz dict.

    Data-driven detection of the seq-to-seq stacked artefact schema
    (ADR-0003): stacked runs write block-prefixed keys (``block{b}_…`` per
    seq-to-seq block, ``agg_…`` for the aggregator), canonical runs write
    flat keys. Figure functions iterate the returned stages and render one
    file per stage; the canonical single stage keeps the historic unsuffixed
    filenames, so existing runs are unaffected.

    Returns:
        list of ``(key_prefix, file_suffix, title_label)`` tuples — e.g.
        ``[("", "", "")]`` for a canonical run, or
        ``[("block0_", "_block0", " — block 0"), …, ("agg_", "_agg",
        " — aggregator")]`` for a stacked run.
    """
    block_ids = sorted({
        int(m.group(1))
        for k in diag
        for m in [re.match(r"block(\d+)_", k)]
        if m is not None
    })
    if not block_ids:
        return [("", "", "")]
    stages = [
        (f"block{b}_", f"_block{b}", f" — block {b}") for b in block_ids
    ]
    if any(k.startswith("agg_") for k in diag):
        stages.append(("agg_", "_agg", " — aggregator"))
    return stages


def _accuracy_from(preds: dict) -> float:
    return float((preds["y_pred"] == preds["y_true"]).mean())


def _cross_entropy_from(preds: dict) -> float:
    # -log p_{true_class}; numerically stabilised against probs == 0.
    probs = np.clip(preds["y_probs"], 1e-12, 1.0)
    idx = preds["y_true"].astype(np.int64)
    rows = np.arange(probs.shape[0])
    return float(-np.log(probs[rows, idx]).mean())


def _per_class_acc_from(preds: dict, num_classes: int) -> np.ndarray:
    """Recall per class — `diag(C)/rowsum(C)` with zero-row safety."""
    y_true = preds["y_true"].astype(np.int64)
    y_pred = preds["y_pred"].astype(np.int64)
    flat = y_true * num_classes + y_pred
    cm = np.bincount(flat, minlength=num_classes * num_classes).reshape(
        num_classes, num_classes
    ).astype(np.int64)
    totals = cm.sum(axis=1)
    return np.where(totals > 0, cm.diagonal() / np.maximum(totals, 1), 0.0)


def _confusion_from(preds: dict, num_classes: int) -> np.ndarray:
    y_true = preds["y_true"].astype(np.int64)
    y_pred = preds["y_pred"].astype(np.int64)
    flat = y_true * num_classes + y_pred
    return np.bincount(flat, minlength=num_classes * num_classes).reshape(
        num_classes, num_classes
    ).astype(np.int64)


def _moving_average(x: list[float] | np.ndarray, window: int) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if len(arr) < window:
        return arr.copy()
    cumsum = np.cumsum(np.insert(arr, 0, 0.0))
    ma = (cumsum[window:] - cumsum[:-window]) / float(window)
    return np.concatenate([np.full(window - 1, np.nan), ma])


def _epoch_boundary_steps(bh: dict) -> list[int]:
    """Global step at which each epoch ended (used for vertical markers)."""
    epochs_arr = np.asarray(bh.get("epoch", []))
    steps_arr = np.asarray(bh.get("step", []))
    if epochs_arr.size == 0:
        return []
    boundaries = []
    for e in sorted(set(epochs_arr.tolist())):
        mask = epochs_arr == e
        if mask.any():
            boundaries.append(int(steps_arr[mask].max()))
    return boundaries[:-1]


def _resolve_epoch(history: dict, epoch_arg: str) -> int:
    """Map 'best' / 'final' / '<N>' to a 1-indexed epoch number."""
    n_epochs = len(history["epoch"]["test_acc"])
    if n_epochs == 0:
        raise ValueError("history.json has no epoch entries")
    if epoch_arg == "best":
        best = history["meta"].get("best_epoch")
        if best is None:
            return int(np.argmax(history["epoch"]["test_acc"])) + 1
        return int(best)
    if epoch_arg == "final":
        return n_epochs
    n = int(epoch_arg)
    if not (1 <= n <= n_epochs):
        raise ValueError(f"epoch {n} out of range [1, {n_epochs}]")
    return n


def load_run(run_dir: Path, epoch_arg: str) -> dict:
    """Read all on-disk artefacts for a run. Does **not** load the model or
    build a test loader — that's deferred to `_load_model_and_run_inference`
    when the caller passes `--full-inference`.

    Returns a dict with: config, history, epoch, predictions (npz dict for
    chosen epoch or None), diagnostics (npz dict for chosen epoch or None),
    test_images (one-time npz dict or None), run_dir, fig_dir.
    """
    run_dir = Path(run_dir).resolve()
    config_path = run_dir / "config.json"
    history_path = run_dir / "history.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"config.json not found in {run_dir}")
    if not history_path.is_file():
        raise FileNotFoundError(f"history.json not found in {run_dir}")

    with open(config_path) as f:
        config_dict = json.load(f)
    config: ExperimentConfig = experiment_config_from_dict(config_dict)

    with open(history_path) as f:
        history = json.load(f)

    epoch = _resolve_epoch(history, epoch_arg)

    pred_path = run_dir / "predictions" / f"epoch_{epoch:04d}.npz"
    predictions = dict(np.load(pred_path)) if pred_path.is_file() else None
    diag_path = run_dir / "diagnostics" / f"epoch_{epoch:04d}.npz"
    diagnostics = dict(np.load(diag_path)) if diag_path.is_file() else None
    test_images_path = run_dir / "predictions" / "test_images.npz"
    test_images = (dict(np.load(test_images_path))
                   if test_images_path.is_file() else None)
    subset_path = run_dir / "subset_indices.npz"
    subset_indices = (dict(np.load(subset_path))
                      if subset_path.is_file() else None)

    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    return {
        "run_dir":        run_dir,
        "fig_dir":        fig_dir,
        "config":         config,
        "history":        history,
        "epoch":          epoch,
        "predictions":    predictions,
        "diagnostics":    diagnostics,
        "test_images":    test_images,
        "subset_indices": subset_indices,
    }


def _load_artefacts_recomputed(run: dict) -> dict | None:
    """Synthesise the `recomputed` dict (used by misclassification gallery
    and t-SNE) from saved files only — no model required.

    Returns None if either `predictions/epoch_NNNN.npz` or
    `predictions/test_images.npz` is missing or lacks the required keys.
    """
    preds = run["predictions"]
    imgs = run["test_images"]
    if preds is None or imgs is None:
        return None
    needed_pred_keys = set(schema.PREDICTION_KEYS_REQUIRED)
    if not needed_pred_keys.issubset(preds.keys()):
        warnings.warn(
            "predictions npz is missing one of "
            f"{sorted(needed_pred_keys - set(preds.keys()))} — run "
            "full_experiment.py again to populate it, or pass --full-inference.",
            RuntimeWarning,
        )
        return None
    if "images" not in imgs:
        return None
    return {
        "patches":  imgs["images"],   # shape (N, H, W) — pre-assembled images
        "y_true":   imgs["labels"] if "labels" in imgs else preds["y_true"],
        "y_pred":   preds["y_pred"],
        "y_probs":  preds["y_probs"],
        "readouts": preds["readouts"],
    }


def _load_model_and_run_inference(run: dict) -> dict:
    """Rebuild the trained model, load the chosen checkpoint, and run a fresh
    forward pass over the test set. Reuses the training-time subset (via
    `subset_indices.npz`) so the recomputed predictions line up row-by-row
    with the saved ones. Heavy — only called under --full-inference.
    """
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Subset
    from cv_quixer.data.mnist import PatchedDataset
    from cv_quixer.models import build_model

    config = run["config"]
    run_dir = run["run_dir"]
    epoch = run["epoch"]

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Resolve checkpoint path (best.pt / final_model.pt / epoch_NNNN.pt / latest.pt)
    ckpt_dir = run_dir / "checkpoints"
    for candidate in (
        ckpt_dir / "best.pt",
        ckpt_dir / f"epoch_{epoch:04d}.pt",
        ckpt_dir / "final_model.pt",
        ckpt_dir / "latest.pt",
    ):
        if candidate.is_file():
            ckpt_path = candidate
            break
    else:
        raise FileNotFoundError(f"No checkpoint found under {ckpt_dir}")
    print(f"  Loading checkpoint: {ckpt_path.name}")

    model = build_model(config).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    test_ds_full = PatchedDataset(config.data, train=False)
    subset_path = run_dir / "subset_indices.npz"
    if subset_path.is_file():
        with np.load(subset_path) as si:
            test_indices = si["test_indices"].astype(np.int64).tolist()
        test_ds = Subset(test_ds_full, indices=test_indices)
        print(f"  Reusing training-time test subset: {len(test_ds):,} / "
              f"{len(test_ds_full):,} samples (from subset_indices.npz)")
    else:
        test_ds = test_ds_full
        warnings.warn(
            "subset_indices.npz not found in this run directory — "
            "--full-inference will run on the full test set, which may "
            "disagree with the saved predictions/confusion matrices for "
            "runs that used --train-limit / --test-limit / --*-fraction.",
            RuntimeWarning,
        )
    test_loader = DataLoader(
        test_ds, batch_size=config.data.batch_size, shuffle=False
    )
    print(f"  Running forward pass on {len(test_ds)} test samples…")

    images_chunks, labels_chunks, preds_chunks, probs_chunks, readout_chunks = (
        [], [], [], [], []
    )
    with torch.no_grad():
        for patches, labels in tqdm(test_loader, desc="full inference",
                                    leave=False, unit="batch",
                                    mininterval=5.0):
            patches_d = patches.to(device)
            out = model(patches_d, return_readouts=True)
            logits, readouts = out.logits, out.readouts
            probs = F.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1)
            images_chunks.append(patches.detach().cpu())
            labels_chunks.append(labels.detach().cpu())
            preds_chunks.append(preds.detach().cpu())
            probs_chunks.append(probs.detach().cpu().float())
            readout_chunks.append(readouts.detach().cpu().float())

    # Reassemble patches into images so the misclassification gallery has
    # something to plot. Uses the same patch-grid math as
    # full_experiment._save_test_images_once.
    raw_patches = torch.cat(images_chunks).numpy().astype(np.float32)
    image_size = config.data.image_size
    patch_size = config.data.patch_size
    grid = image_size // patch_size
    n_total = raw_patches.shape[0]
    images = np.zeros((n_total, image_size, image_size), dtype=np.float32)
    for i in range(n_total):
        for k in range(raw_patches.shape[1]):
            r, c = divmod(k, grid)
            images[i, r*patch_size:(r+1)*patch_size,
                      c*patch_size:(c+1)*patch_size] = (
                raw_patches[i, k].reshape(patch_size, patch_size)
            )

    return {
        "patches":  images,
        "y_true":   torch.cat(labels_chunks).numpy().astype(np.int64),
        "y_pred":   torch.cat(preds_chunks).numpy().astype(np.int64),
        "y_probs":  torch.cat(probs_chunks).numpy().astype(np.float32),
        "readouts": torch.cat(readout_chunks).numpy().astype(np.float32),
    }


def _check_parity(run: dict) -> None:
    """Audit M10: assert the file-only and full-inference prediction paths
    agree. Loads both, compares softmax probabilities row-by-row, and prints
    a one-line PASS/WARN summary. Manual safety net — no CI plumbing.
    """
    file = _load_artefacts_recomputed(run)
    if file is None:
        print("check-parity: file-only artefacts unavailable — nothing to "
              "compare (run full_experiment.py to populate predictions/*.npz).")
        return
    try:
        fresh = _load_model_and_run_inference(run)
    except Exception as e:
        print(f"check-parity: full-inference failed "
              f"({type(e).__name__}: {e}) — cannot compare.")
        return

    fp, xp = file["y_probs"], fresh["y_probs"]
    if fp.shape != xp.shape:
        print(f"check-parity: shape mismatch file={fp.shape} "
              f"full-inference={xp.shape} — likely missing subset_indices.npz "
              "so the two paths ran on different sample sets; cannot compare.")
        return

    d = np.abs(fp - xp)
    max_d, mean_d = float(d.max()), float(d.mean())
    pred_agree = float((file["y_pred"] == fresh["y_pred"]).mean())
    verdict = "PASS" if max_d < 1e-5 else "WARN"
    print(f"check-parity [{verdict}]: max|Δp|={max_d:.2e} "
          f"mean|Δp|={mean_d:.2e} pred-agreement={pred_agree:.4f} "
          f"(N={fp.shape[0]})")


# ---------------------------------------------------------------------------
# Individual figure functions (each with try/except in main)
# ---------------------------------------------------------------------------


def plot_training_curves(run: dict) -> None:
    """Render the training-time curves and the latest-epoch test confusion
    matrix from raw artefacts. Cross-checks the derived per-epoch loss /
    accuracy values against the history.json training log and warns on
    mismatch — never reads the logged value into the figure.
    """
    eh = run["history"]["epoch"]
    bh = run["history"].get("batch", {})
    config = run["config"]
    fig_dir = run["fig_dir"]
    run_dir = run["run_dir"]
    num_classes = config.data.num_classes
    classes = class_names(config)

    n_epochs = len(eh.get("test_loss", []))
    if n_epochs == 0:
        print("  - history.json has no epoch entries → skipping training curves")
        return

    # Load per-epoch predictions for both sides — hard-fail if any are missing.
    test_preds = _load_all_per_epoch_predictions(run_dir, side="test",
                                                 n_epochs=n_epochs)
    train_preds = _load_all_per_epoch_predictions(run_dir, side="train",
                                                  n_epochs=n_epochs)

    epoch_x = list(range(1, n_epochs + 1))
    train_loss_derived = [_cross_entropy_from(train_preds[e]) for e in epoch_x]
    test_loss_derived = [_cross_entropy_from(test_preds[e]) for e in epoch_x]
    train_acc_derived = [_accuracy_from(train_preds[e]) for e in epoch_x]
    test_acc_derived = [_accuracy_from(test_preds[e]) for e in epoch_x]

    # Cross-check against the training log. The two values should agree to
    # within float32 + device drift since both come from the same evaluate()
    # call inside the training loop.
    for e, (d, l) in enumerate(zip(train_loss_derived, eh.get("train_loss", [])), 1):
        _warn_mismatch(f"epoch {e} train_loss", d, l, tol=1e-3)
    for e, (d, l) in enumerate(zip(test_loss_derived, eh.get("test_loss", [])), 1):
        _warn_mismatch(f"epoch {e} test_loss", d, l, tol=1e-3)
    for e, (d, l) in enumerate(zip(train_acc_derived, eh.get("train_acc", [])), 1):
        _warn_mismatch(f"epoch {e} train_acc", d, l, tol=1e-4)
    for e, (d, l) in enumerate(zip(test_acc_derived, eh.get("test_acc", [])), 1):
        _warn_mismatch(f"epoch {e} test_acc", d, l, tol=1e-4)

    # Per-epoch loss
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(epoch_x, train_loss_derived, label="train loss", marker="o")
    ax.plot(epoch_x, test_loss_derived, label="test loss", marker="s")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("Loss (per epoch)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "loss_curve.png", dpi=150)
    plt.close(fig)
    print("  ✓ loss_curve.png")

    # Per-epoch accuracy
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(epoch_x, train_acc_derived, label="train acc", marker="o")
    ax.plot(epoch_x, test_acc_derived, label="test acc", marker="s")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy (per epoch)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "accuracy_curve.png", dpi=150)
    plt.close(fig)
    print("  ✓ accuracy_curve.png")

    # Per-epoch trunc loss — training-log-only until per-sample trunc is
    # plumbed in Step 3. Falls back gracefully if both fields are absent.
    trunc = eh.get("trunc_loss") or []
    test_trunc = eh.get("test_trunc_loss") or []
    if trunc or test_trunc:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        if trunc:
            ax.plot(epoch_x[:len(trunc)], trunc,
                    label="train trunc loss", color="tab:orange", marker="o")
        if test_trunc:
            ax.plot(epoch_x[:len(test_trunc)], test_trunc,
                    label="test trunc loss", color="tab:blue", marker="s")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Mean per-patch truncation loss")
        ax.set_title("Truncation loss (per epoch) — from training log")
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / "trunc_loss_curve.png", dpi=150)
        plt.close(fig)
        print("  ✓ trunc_loss_curve.png")

    # Per-epoch CVQNN block (W) truncation loss — the separate 1 − ‖W|ψ⟩‖²
    # leakage, tracked independently of the per-patch trunc loss above. Absent
    # (empty) for pre-W runs / cvqnn_num_layers == 0, in which case this is
    # skipped.
    cvqnn_trunc = eh.get("cvqnn_trunc_loss") or []
    test_cvqnn_trunc = eh.get("test_cvqnn_trunc_loss") or []
    if cvqnn_trunc or test_cvqnn_trunc:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        if cvqnn_trunc:
            ax.plot(epoch_x[:len(cvqnn_trunc)], cvqnn_trunc,
                    label="train CVQNN trunc loss", color="tab:green", marker="o")
        if test_cvqnn_trunc:
            ax.plot(epoch_x[:len(test_cvqnn_trunc)], test_cvqnn_trunc,
                    label="test CVQNN trunc loss", color="tab:red", marker="s")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Mean CVQNN (W) truncation loss")
        ax.set_title("CVQNN block (W) truncation loss (per epoch) — from training log")
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / "cvqnn_trunc_loss_curve.png", dpi=150)
        plt.close(fig)
        print("  ✓ cvqnn_trunc_loss_curve.png")

    # Per-epoch query-unitary truncation loss — the separate
    # mean_i(1 − ‖U_{q,i}|0⟩‖²) stream (seq-to-seq stacked model only,
    # ADR-0003). Canonical runs log a zero stand-in every epoch, so skip when
    # the stream is absent OR identically zero — a flat zero line is noise.
    query_trunc = eh.get("query_trunc_loss") or []
    test_query_trunc = eh.get("test_query_trunc_loss") or []
    if any(v != 0.0 for v in query_trunc + test_query_trunc):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        if query_trunc:
            ax.plot(epoch_x[:len(query_trunc)], query_trunc,
                    label="train query trunc loss", color="tab:purple", marker="o")
        if test_query_trunc:
            ax.plot(epoch_x[:len(test_query_trunc)], test_query_trunc,
                    label="test query trunc loss", color="tab:brown", marker="s")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Mean query-unitary truncation loss")
        ax.set_title("Query-unitary truncation loss (per epoch) — from training log")
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / "query_trunc_loss_curve.png", dpi=150)
        plt.close(fig)
        print("  ✓ query_trunc_loss_curve.png")

    # Per-class accuracy curve (test) — derived per epoch from predictions.
    per_class = np.stack(
        [_per_class_acc_from(test_preds[e], num_classes) for e in epoch_x]
    )  # (n_epochs, num_classes)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    cmap = matplotlib.colormaps.get_cmap("tab10").resampled(num_classes)
    for c in range(num_classes):
        ax.plot(epoch_x, per_class[:, c], marker="o", color=cmap(c),
                label=f"{c}: {classes[c]}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test accuracy (recall)")
    ax.set_title("Per-class test accuracy across epochs")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(fig_dir / "per_class_accuracy_curve.png", dpi=150)
    plt.close(fig)
    print("  ✓ per_class_accuracy_curve.png")

    # Latest-epoch test confusion matrix (counts + row-normalised).
    last_cm = _confusion_from(test_preds[n_epochs], num_classes)
    last_cm_row = last_cm / np.maximum(last_cm.sum(axis=1, keepdims=True), 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax_, mat, title, fmt in (
        (axes[0], last_cm, "Counts", "d"),
        (axes[1], last_cm_row, "Row-normalised", ".2f"),
    ):
        im = ax_.imshow(mat, cmap="Blues")
        ax_.set_title(f"{title} (epoch {n_epochs})")
        ax_.set_xticks(range(num_classes))
        ax_.set_yticks(range(num_classes))
        ax_.set_xticklabels(classes, rotation=45, ha="right", fontsize=8)
        ax_.set_yticklabels(classes, fontsize=8)
        ax_.set_xlabel("Predicted"); ax_.set_ylabel("True")
        for i in range(num_classes):
            for j in range(num_classes):
                val = mat[i, j]
                colour = "white" if (val > mat.max() * 0.6) else "black"
                ax_.text(j, i, format(val, fmt),
                         ha="center", va="center", color=colour, fontsize=7)
        fig.colorbar(im, ax=ax_, fraction=0.046, pad=0.04)
    fig.suptitle("Test confusion matrix")
    fig.tight_layout()
    fig.savefig(fig_dir / "confusion_matrix_test.png", dpi=150)
    plt.close(fig)
    print("  ✓ confusion_matrix_test.png")

    # Per-batch curves — sourced from history["batch"] (intrinsically a
    # training-time observation, not derivable from predictions).
    steps = bh.get("step") or []
    if not steps:
        return
    boundaries = _epoch_boundary_steps(bh)

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
        ax.set_xlabel("Global batch step")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / fname, dpi=150)
        plt.close(fig)
        print(f"  ✓ {fname}")

    _per_batch_plot(bh.get("train_loss") or [], "Cross-entropy loss",
                    "Train CE loss (per batch)", "per_batch_train_loss.png")
    _per_batch_plot(bh.get("trunc_loss") or [], "Truncation loss",
                    "Truncation loss (per batch)", "per_batch_trunc_loss.png")
    _per_batch_plot(bh.get("batch_acc") or [], "Batch accuracy",
                    "Train accuracy (per batch)", "per_batch_train_accuracy.png")
    _per_batch_plot(bh.get("grad_norm") or [], "Gradient L2 norm (log scale)",
                    "Gradient norm (per batch)", "per_batch_grad_norm.png",
                    log_y=True)
    # The W / query streams are zero stand-ins when the block is disabled /
    # the model has no query unitaries — skip the all-zero noise figures.
    cvqnn_batch = bh.get("cvqnn_trunc_loss") or []
    if any(v != 0.0 for v in cvqnn_batch):
        _per_batch_plot(cvqnn_batch, "CVQNN (W) truncation loss",
                        "CVQNN block (W) truncation loss (per batch)",
                        "per_batch_cvqnn_trunc_loss.png")
    query_batch = bh.get("query_trunc_loss") or []
    if any(v != 0.0 for v in query_batch):
        _per_batch_plot(query_batch, "Query-unitary truncation loss",
                        "Query-unitary truncation loss (per batch)",
                        "per_batch_query_trunc_loss.png")


def plot_confusion_matrix_evolution(run: dict) -> None:
    """Render a grid of row-normalised confusion matrices, one per epoch,
    derived from each epoch's test predictions npz.
    """
    eh = run["history"]["epoch"]
    n_epochs = len(eh.get("test_loss", []))
    if n_epochs == 0:
        print("  - history.json has no epoch entries → skipping confusion_matrix_evolution")
        return
    test_preds = _load_all_per_epoch_predictions(
        run["run_dir"], side="test", n_epochs=n_epochs,
    )
    classes = class_names(run["config"])
    num_classes = run["config"].data.num_classes
    cols = min(4, n_epochs)
    rows = (n_epochs + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 3.5 * rows),
                             squeeze=False)
    for idx, ax in enumerate(axes.flat):
        if idx >= n_epochs:
            ax.axis("off")
            continue
        cm = _confusion_from(test_preds[idx + 1], num_classes).astype(np.float64)
        cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_title(f"epoch {idx + 1}", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"Test confusion-matrix evolution ({len(classes)} classes)")
    fig.tight_layout()
    fig.savefig(run["fig_dir"] / "confusion_matrix_evolution.png", dpi=150)
    plt.close(fig)
    print("  ✓ confusion_matrix_evolution.png")


def write_per_class_metrics_table(run: dict) -> None:
    from sklearn.metrics import classification_report

    if run["predictions"] is None:
        print("  - predictions/ missing → skipping per_class_metrics_table")
        return
    y_true = run["predictions"]["y_true"]
    y_pred = run["predictions"]["y_pred"]
    classes = class_names(run["config"])
    report = classification_report(
        y_true, y_pred,
        target_names=classes,
        digits=4,
        zero_division=0,
    )
    out_path = run["fig_dir"] / "per_class_metrics_table.txt"
    out_path.write_text(
        f"Classification report — epoch {run['epoch']}\n"
        f"({len(y_true)} test samples)\n\n{report}\n"
    )
    print(f"  ✓ per_class_metrics_table.txt (epoch {run['epoch']})")


def plot_top_k_accuracy(run: dict) -> None:
    run_dir = run["run_dir"]
    n_epochs = len(run["history"]["epoch"]["test_acc"])
    top_ks = (1, 2, 3)
    accs = {k: [] for k in top_ks}
    epochs_x = []
    for e in range(1, n_epochs + 1):
        p = run_dir / "predictions" / f"epoch_{e:04d}.npz"
        if not p.is_file():
            continue
        data = np.load(p)
        y_true = data["y_true"]
        y_probs = data["y_probs"]
        # top-k predictions
        order = np.argsort(-y_probs, axis=-1)
        epochs_x.append(e)
        for k in top_ks:
            hit = (order[:, :k] == y_true[:, None]).any(axis=-1)
            accs[k].append(float(hit.mean()))
    if not epochs_x:
        print("  - predictions/ missing → skipping top_k_accuracy")
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    markers = {1: "o", 2: "s", 3: "^"}
    for k in top_ks:
        ax.plot(epochs_x, accs[k], marker=markers[k], label=f"top-{k}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Top-k accuracy on the test set")
    ax.set_ylim(0, 1.02)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(run["fig_dir"] / "top_k_accuracy.png", dpi=150)
    plt.close(fig)
    print("  ✓ top_k_accuracy.png")


def plot_misclassification_gallery(run: dict, recomputed: dict | None) -> None:
    if recomputed is None:
        print("  - recomputed test outputs missing → skipping misclassification_gallery")
        return
    y_true = recomputed["y_true"]
    y_pred = recomputed["y_pred"]
    y_probs = recomputed["y_probs"]
    images = recomputed["patches"]    # in this refactor, already (N, H, W)
    classes = class_names(run["config"])

    wrong = np.where(y_true != y_pred)[0]
    if len(wrong) == 0:
        print("  - no misclassifications → skipping misclassification_gallery")
        return
    wrong_conf = y_probs[wrong, y_pred[wrong]]
    order = wrong[np.argsort(-wrong_conf)]
    n_show = min(24, len(order))

    cols = 6
    rows = (n_show + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2.0 * cols, 2.2 * rows),
                             squeeze=False)
    for idx, ax in enumerate(axes.flat):
        if idx >= n_show:
            ax.axis("off")
            continue
        i = order[idx]
        ax.imshow(images[i], cmap="gray")
        ax.set_title(
            f"true: {classes[y_true[i]]}\npred: {classes[y_pred[i]]} "
            f"({y_probs[i, y_pred[i]]:.2f})",
            fontsize=7,
        )
        ax.axis("off")
    fig.suptitle(f"Most-confident misclassifications (epoch {run['epoch']})")
    fig.tight_layout()
    fig.savefig(run["fig_dir"] / "misclassification_gallery.png", dpi=150)
    plt.close(fig)
    print("  ✓ misclassification_gallery.png")


def plot_calibration_reliability(run: dict) -> None:
    if run["predictions"] is None:
        print("  - predictions/ missing → skipping calibration_reliability")
        return
    y_true = run["predictions"]["y_true"]
    y_probs = run["predictions"]["y_probs"]
    conf = y_probs.max(axis=-1)
    pred = y_probs.argmax(axis=-1)
    correct = (pred == y_true).astype(np.float64)
    bins = np.linspace(0, 1, 16)
    bin_ids = np.digitize(conf, bins) - 1
    bin_acc, bin_conf, bin_count = [], [], []
    for b in range(len(bins) - 1):
        mask = bin_ids == b
        if mask.any():
            bin_acc.append(correct[mask].mean())
            bin_conf.append(conf[mask].mean())
            bin_count.append(int(mask.sum()))
        else:
            bin_acc.append(np.nan)
            bin_conf.append((bins[b] + bins[b+1]) / 2)
            bin_count.append(0)
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.plot([0, 1], [0, 1], ls="--", color="gray", label="perfect calibration")
    ax.plot(bin_conf, bin_acc, marker="o", label="model")
    for x, y, n in zip(bin_conf, bin_acc, bin_count):
        if n > 0 and not np.isnan(y):
            ax.annotate(str(n), (x, y), fontsize=6, alpha=0.5,
                        textcoords="offset points", xytext=(2, 2))
    ax.set_xlabel("Confidence (max softmax)")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title(f"Calibration / reliability — epoch {run['epoch']}")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(run["fig_dir"] / "calibration_reliability.png", dpi=150)
    plt.close(fig)
    print("  ✓ calibration_reliability.png")


def plot_embedding_tsne(run: dict, recomputed: dict | None) -> None:
    if recomputed is None:
        print("  - recomputed test outputs missing → skipping embedding_tsne")
        return
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("  - sklearn not installed → skipping embedding_tsne")
        return
    readouts = recomputed["readouts"]
    y_true = recomputed["y_true"]
    classes = class_names(run["config"])
    # Sub-sample if very large (TSNE is O(n^2))
    n_max = 5000
    if len(readouts) > n_max:
        idx = np.random.default_rng(0).choice(len(readouts), n_max, replace=False)
        readouts, y_true = readouts[idx], y_true[idx]
    print(f"  ... running t-SNE on {len(readouts)} samples (this can take ~1 min)")
    emb = TSNE(n_components=2, perplexity=30, init="pca", random_state=0).fit_transform(readouts)
    fig, ax = plt.subplots(figsize=(8, 7))
    cmap = matplotlib.colormaps.get_cmap("tab10").resampled(len(classes))
    for c in range(len(classes)):
        m = y_true == c
        ax.scatter(emb[m, 0], emb[m, 1], s=5, alpha=0.6,
                   color=cmap(c), label=f"{c}: {classes[c]}")
    ax.set_title(f"t-SNE of pre-decoder readouts (epoch {run['epoch']})")
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc="upper right", fontsize=8, markerscale=2)
    fig.tight_layout()
    fig.savefig(run["fig_dir"] / "embedding_tsne.png", dpi=150)
    plt.close(fig)
    print("  ✓ embedding_tsne.png")


def _collect_gate_param_grids(
    diag: dict, prefix: str
) -> tuple[dict[int, dict[str, np.ndarray]], dict[int, dict[str, np.ndarray]]]:
    """Group one stage's gate-param arrays as ``{head: {gate_name: arr}}``.

    Keys follow ``{prefix}head{h}_{gate_name}`` with the seq-to-seq query
    slice marked by a ``q_`` gate-name prefix (``{prefix}head{h}_q_{name}``,
    ADR-0003). Returns ``(key_grid, query_grid)``; the query grid is empty for
    canonical runs and the aggregator stage.
    """
    key_grid: dict[int, dict[str, np.ndarray]] = {}
    query_grid: dict[int, dict[str, np.ndarray]] = {}
    for k, arr in diag.items():
        if not k.startswith(prefix):
            continue
        rest = k[len(prefix):]
        if not rest.startswith("head"):
            continue
        head_str, _, gname = rest.partition("_")
        if not gname or gname == "state_norms":
            continue
        try:
            h = int(head_str[4:])
        except ValueError:
            continue
        if gname.startswith("q_"):
            query_grid.setdefault(h, {})[gname[2:]] = arr
        else:
            key_grid.setdefault(h, {})[gname] = arr
    return key_grid, query_grid


def _render_gate_histogram_grid(
    grid: dict[int, dict[str, np.ndarray]], title: str, out_path: Path
) -> None:
    """Render one heads × gate-types histogram grid (the historic layout)."""
    heads = sorted(grid)
    gate_names = sorted({g for per_head in grid.values() for g in per_head})
    fig, axes = plt.subplots(len(heads), len(gate_names),
                             figsize=(2.2 * len(gate_names), 2.0 * len(heads)),
                             squeeze=False)
    for row, h in enumerate(heads):
        for col, gname in enumerate(gate_names):
            ax = axes[row, col]
            arr = grid[h].get(gname)
            if arr is None or arr.size == 0:
                ax.set_visible(False)
                continue
            ax.hist(arr.reshape(-1), bins=40, color="tab:blue", alpha=0.7)
            if row == 0:
                ax.set_title(gname, fontsize=8)
            if col == 0:
                ax.set_ylabel(f"head {h}", fontsize=8)
            ax.tick_params(axis="both", labelsize=6)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  ✓ {out_path.name}")


def plot_hypernet_gate_histograms(run: dict) -> None:
    diag = run["diagnostics"]
    if diag is None:
        print("  - diagnostics/ missing → skipping hypernet_gate_param_histograms")
        return
    rendered = False
    for prefix, suffix, label in _diag_stages(diag):
        key_grid, query_grid = _collect_gate_param_grids(diag, prefix)
        if key_grid:
            _render_gate_histogram_grid(
                key_grid,
                f"Hypernetwork gate-parameter distributions{label} "
                f"(epoch {run['epoch']})",
                run["fig_dir"] / f"hypernet_gate_param_histograms{suffix}.png",
            )
            rendered = True
        if query_grid:
            _render_gate_histogram_grid(
                query_grid,
                f"Query-unitary gate-parameter distributions{label} "
                f"(epoch {run['epoch']})",
                run["fig_dir"]
                / f"hypernet_gate_param_histograms{suffix}_query.png",
            )
            rendered = True
    if not rendered:
        print("  - no gate-param arrays in diagnostics/ → skipping hypernet histograms")


def plot_cvqnn_param_values(run: dict) -> None:
    """Per-head CVQNN block (W) learned gate-param values, vs the identity (0).

    W's params are fixed (input-independent) per head, so we plot the raw learned
    vector per head as a stem/bar — how far each gate param has drifted from the
    near-identity small-random init directly answers "is W doing anything?".
    Absent for pre-W runs / cvqnn_num_layers == 0 (no `cvqnn_params` key), which
    is skipped (not an error).
    """
    diag = run["diagnostics"]
    stages = _diag_stages(diag) if diag is not None else []
    if diag is None or not any(f"{p}cvqnn_params" in diag for p, _, _ in stages):
        print("  - no `cvqnn_params` in diagnostics/ (cvqnn_num_layers == 0?) "
              "→ skipping cvqnn_param_values")
        return
    for prefix, suffix, label in stages:
        key = f"{prefix}cvqnn_params"
        if key not in diag:
            continue
        arr = np.asarray(diag[key])   # (num_heads, cvqnn_param_count)
        if arr.ndim != 2 or arr.size == 0:
            print(f"  - empty `{key}` → skipping cvqnn_param_values{suffix}")
            continue
        num_heads, n_param = arr.shape
        x = np.arange(n_param)
        fig, axes = plt.subplots(num_heads, 1,
                                 figsize=(max(6, 0.25 * n_param), 1.8 * num_heads),
                                 squeeze=False)
        for h in range(num_heads):
            ax = axes[h, 0]
            ax.axhline(0.0, color="gray", lw=0.8, ls="--")  # identity reference
            ax.bar(x, arr[h], color="tab:purple", alpha=0.8)
            ax.set_ylabel(f"head {h}", fontsize=8)
            ax.tick_params(axis="both", labelsize=6)
        axes[-1, 0].set_xlabel("W gate-param index (plan order)", fontsize=8)
        fig.suptitle(
            f"CVQNN block (W) gate-param values vs identity{label} "
            f"(epoch {run['epoch']})"
        )
        fig.tight_layout()
        fname = f"cvqnn_param_values{suffix}.png"
        fig.savefig(run["fig_dir"] / fname, dpi=150)
        plt.close(fig)
        print(f"  ✓ {fname}")


def plot_photon_number_per_mode(run: dict) -> None:
    diag = run["diagnostics"]
    if diag is None or "mean_photon_number" not in diag:
        raise MissingArtefactError(
            f"diagnostics/epoch_{run['epoch']:04d}.npz missing or has no "
            f"`mean_photon_number` key — run `uv run python "
            f"experiments/backfill_artefacts.py --run-dir {run['run_dir']}` "
            "to produce it."
        )
    arr = np.asarray(diag["mean_photon_number"])  # (num_heads, num_modes)
    num_heads, num_modes = arr.shape
    cutoff = run["config"].quantum.cutoff_dim
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(num_modes)
    width = 0.8 / num_heads
    for h in range(num_heads):
        ax.bar(x + h * width, arr[h], width=width, label=f"head {h}")
    ax.set_xticks(x + width * (num_heads - 1) / 2)
    ax.set_xticklabels([f"mode {k}" for k in range(num_modes)])
    ax.set_ylabel(r"Mean $\langle \hat n_k \rangle$")
    ax.set_title(
        f"Per-mode mean photon number (epoch {run['epoch']}, cutoff={cutoff})"
    )
    ax.axhline(cutoff - 1, color="red", ls="--", lw=0.8,
               label=f"cutoff − 1 = {cutoff - 1}")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(run["fig_dir"] / "photon_number_per_mode.png", dpi=150)
    plt.close(fig)
    print("  ✓ photon_number_per_mode.png")


def plot_state_norm_histogram(run: dict) -> None:
    diag = run["diagnostics"]
    if diag is None:
        print("  - diagnostics/ missing → skipping state_norm_histogram")
        return
    norm_keys = sorted(k for k in diag.keys() if k.endswith("_state_norms"))
    if not norm_keys:
        print("  - no state-norm arrays in diagnostics/ → skipping")
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    all_vals = np.concatenate([diag[k] for k in norm_keys])
    val_range = float(all_vals.max() - all_vals.min())
    # If the distribution is essentially a point mass (norms ≈ 1 to float32
    # precision before any truncation has built up), draw a small fixed window
    # centred on 1.0 instead of trying to fit 40 bins into a sub-precision range.
    if val_range < 1e-4:
        bin_edges = np.linspace(0.999, 1.001, 21)
    else:
        bin_edges = 40
    for k in norm_keys:
        ax.hist(diag[k], bins=bin_edges, alpha=0.5,
                label=k.replace("_state_norms", ""))
    ax.set_xlabel(r"Output state norm $\|\psi\|^2$")
    ax.set_ylabel("Count")
    ax.set_title(f"Output state norm across diagnostic subset (epoch {run['epoch']})")
    ax.axvline(1.0, color="red", ls="--", lw=0.8, label="unit norm")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(run["fig_dir"] / "state_norm_histogram.png", dpi=150)
    plt.close(fig)
    print("  ✓ state_norm_histogram.png")


def plot_lcu_coefficients_heatmap(run: dict) -> None:
    diag = run["diagnostics"]
    stages = _diag_stages(diag) if diag is not None else []
    if diag is None or not any(f"{p}{schema.LCU_COEFFS}" in diag for p, _, _ in stages):
        raise MissingArtefactError(
            f"diagnostics/epoch_{run['epoch']:04d}.npz missing or has no "
            f"`lcu_coeffs` key — run `uv run python "
            f"experiments/backfill_artefacts.py --run-dir {run['run_dir']}` "
            "to produce it."
        )
    for prefix, suffix, label in stages:
        key = f"{prefix}{schema.LCU_COEFFS}"
        if key not in diag:
            continue
        arr = np.asarray(diag[key])   # (num_heads, num_patches, 2)
        magnitude = np.sqrt((arr ** 2).sum(axis=-1))  # (num_heads, num_patches)
        fig, ax = plt.subplots(figsize=(10, 4))
        im = ax.imshow(magnitude, aspect="auto", cmap="viridis")
        ax.set_xlabel("Patch index")
        ax.set_ylabel("Head")
        ax.set_title(
            f"LCU coefficient magnitudes |b_i|{label} (epoch {run['epoch']})"
        )
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fname = f"lcu_coefficients_heatmap{suffix}.png"
        fig.savefig(run["fig_dir"] / fname, dpi=150)
        plt.close(fig)
        print(f"  ✓ {fname}")


def plot_polynomial_coefficient_trajectory(run: dict) -> None:
    eh = run["history"]["epoch"]
    n_epochs = len(eh.get("test_loss", []))
    if n_epochs == 0:
        print("  - history.json has no epoch entries → skipping polynomial_coefficients_trajectory")
        return
    diag_per_epoch = _load_all_diagnostics(run["run_dir"], n_epochs=n_epochs)
    stages = _diag_stages(diag_per_epoch[1])
    if not any(f"{p}{schema.POLY_COEFFS}" in diag_per_epoch[1] for p, _, _ in stages):
        raise MissingArtefactError(
            f"diagnostics/epoch_0001.npz missing `poly_coeffs` — "
            f"run `uv run python experiments/backfill_artefacts.py "
            f"--run-dir {run['run_dir']}` to produce it."
        )
    for prefix, suffix, label in stages:
        key = f"{prefix}{schema.POLY_COEFFS}"
        if key not in diag_per_epoch[1]:
            continue
        poly_chunks = []
        for e in range(1, n_epochs + 1):
            d = diag_per_epoch[e]
            if key not in d:
                raise MissingArtefactError(
                    f"diagnostics/epoch_{e:04d}.npz missing `{key}` — "
                    f"run `uv run python experiments/backfill_artefacts.py "
                    f"--run-dir {run['run_dir']}` to produce it."
                )
            poly_chunks.append(np.asarray(d[key]))
        arr = np.stack(poly_chunks)        # (n_epochs, num_heads, degree+1)
        _, num_heads, degree_plus_1 = arr.shape
        fig, axes = plt.subplots(1, num_heads, figsize=(3.5 * num_heads, 4),
                                 sharey=True, squeeze=False)
        epoch_x = list(range(1, n_epochs + 1))
        for h in range(num_heads):
            ax = axes[0, h]
            for j in range(degree_plus_1):
                ax.plot(epoch_x, arr[:, h, j], marker="o", label=f"c_{j}")
            ax.set_title(f"head {h}")
            ax.set_xlabel("Epoch")
            if h == 0:
                ax.set_ylabel("Coefficient value")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        fig.suptitle(f"Polynomial coefficient trajectory{label}")
        fig.tight_layout()
        fname = f"polynomial_coefficients_trajectory{suffix}.png"
        fig.savefig(run["fig_dir"] / fname, dpi=150)
        plt.close(fig)
        print(f"  ✓ {fname}")


def plot_success_prob_histogram(run: dict) -> None:
    """Distribution of the LCU/QSVT post-selection success probability
    ‖P(M)|ψ⟩‖²/λ² across the test set at the selected epoch, per head.
    λ is derived from the saved lcu/poly coefficients (ADR-0002)."""
    preds = run["predictions"]
    if preds is None or schema.SUCCESS_PROBS not in preds:
        raise MissingArtefactError(
            f"predictions/epoch_{run['epoch']:04d}.npz missing or has no "
            f"`success_probs` key — run `uv run python "
            f"experiments/backfill_artefacts.py --run-dir {run['run_dir']}` "
            "to produce it."
        )
    diag = run["diagnostics"]
    if diag is None or schema.LCU_COEFFS not in diag or schema.POLY_COEFFS not in diag:
        raise MissingArtefactError(
            f"diagnostics/epoch_{run['epoch']:04d}.npz missing or lacks "
            f"`lcu_coeffs`/`poly_coeffs` (needed to derive λ) — run "
            f"`uv run python experiments/backfill_artefacts.py "
            f"--run-dir {run['run_dir']}` to produce it."
        )
    raw = np.asarray(preds["success_probs"], dtype=np.float64)         # (N, H)
    lam = _derive_lcu_lambda(diag["lcu_coeffs"], diag["poly_coeffs"])  # (H,)
    ratio = raw / lam[None, :] ** 2                                    # (N, H)
    # Failure threshold applies to the RAW norm, mirroring the clamp
    # semantics in cv_attention (elements below it are forced to zero state).
    fail_frac = (raw < _SUCCESS_PROB_FLOOR).mean(axis=0)               # (H,)

    pos = ratio[ratio > 0]
    # Log-x when the positive values span more than two decades.
    log_x = pos.size > 0 and float(pos.max()) / float(pos.min()) > 100.0
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if log_x:
        bins = np.logspace(np.log10(pos.min()), np.log10(pos.max()), 41)
        ax.set_xscale("log")
    else:
        bins = 40
    for h in range(ratio.shape[1]):
        vals = ratio[:, h]
        if log_x:
            vals = vals[vals > 0]   # zeros reported via fail= in the legend
        ax.hist(vals, bins=bins, alpha=0.5,
                label=f"head {h} (λ={lam[h]:.3g}, fail={fail_frac[h]:.1%})")
    ax.set_xlabel(
        r"Post-selection success probability $\|P(M)|\psi\rangle\|^2 / \lambda^2$"
    )
    ax.set_ylabel("Count")
    ax.set_title(f"LCU/QSVT post-selection success probability "
                 f"(epoch {run['epoch']}, test set)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(run["fig_dir"] / "success_prob_histogram.png", dpi=150)
    plt.close(fig)
    print("  ✓ success_prob_histogram.png")


def plot_success_prob_trajectory(run: dict) -> None:
    """Mean + 10–90th percentile band of the normalised post-selection
    success probability vs epoch, per head (test side)."""
    preds_per_epoch = _load_all_per_epoch_predictions(run["run_dir"],
                                                      side="test")
    epochs = sorted(e for e, d in preds_per_epoch.items()
                    if "success_probs" in d)
    if preds_per_epoch and not epochs:
        raise MissingArtefactError(
            f"no predictions/epoch_NNNN.npz under {run['run_dir']} carries a "
            f"`success_probs` key — run `uv run python "
            f"experiments/backfill_artefacts.py --run-dir {run['run_dir']}` "
            "to produce it."
        )
    if len(epochs) < 2:
        print("  - fewer than 2 epochs with success_probs → "
              "skipping success_prob_trajectory")
        return
    diag_per_epoch = _load_all_diagnostics(run["run_dir"])
    for e in epochs:
        d = diag_per_epoch.get(e)
        if d is None or "lcu_coeffs" not in d or "poly_coeffs" not in d:
            raise MissingArtefactError(
                f"diagnostics/epoch_{e:04d}.npz missing or lacks "
                f"`lcu_coeffs`/`poly_coeffs` (needed for per-epoch λ) — run "
                f"`uv run python experiments/backfill_artefacts.py "
                f"--run-dir {run['run_dir']}` to produce it."
            )
    num_heads = np.asarray(preds_per_epoch[epochs[0]]["success_probs"]).shape[1]
    mean = np.zeros((len(epochs), num_heads))
    p10 = np.zeros_like(mean)
    p90 = np.zeros_like(mean)
    for i, e in enumerate(epochs):
        raw = np.asarray(preds_per_epoch[e]["success_probs"], dtype=np.float64)
        lam = _derive_lcu_lambda(diag_per_epoch[e]["lcu_coeffs"],
                                 diag_per_epoch[e]["poly_coeffs"])
        ratio = raw / lam[None, :] ** 2
        mean[i] = ratio.mean(axis=0)
        p10[i] = np.percentile(ratio, 10, axis=0)
        p90[i] = np.percentile(ratio, 90, axis=0)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for h in range(num_heads):
        (line,) = ax.plot(epochs, mean[:, h], marker="o", label=f"head {h}")
        ax.fill_between(epochs, p10[:, h], p90[:, h],
                        color=line.get_color(), alpha=0.2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"$\|P(M)|\psi\rangle\|^2 / \lambda^2$")
    ax.set_title("Post-selection success probability vs epoch "
                 "(mean, 10–90th pct band, test set)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(run["fig_dir"] / "success_prob_trajectory.png", dpi=150)
    plt.close(fig)
    print("  ✓ success_prob_trajectory.png")


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------


def sanity_checks(run: dict) -> None:
    # Compare loaded test predictions vs subset_indices.npz / test_images.
    si = run["subset_indices"]
    if si is not None and run["predictions"] is not None:
        n_idx = int(si["test_indices"].size)
        n_pred = int(run["predictions"]["y_true"].size)
        if n_idx != n_pred:
            warnings.warn(
                f"subset_indices.npz says test set has {n_idx} samples but "
                f"predictions/epoch_NNNN.npz has {n_pred}. Predictions may "
                "have been written before the subset was changed, or vice "
                "versa.",
                RuntimeWarning,
            )

    # LCU coeffs shape sanity — pulled from the chosen epoch's diagnostics
    # npz now that the per-epoch history.epoch.lcu_coeffs list is gone.
    # Checks every stage namespace (canonical flat key, or the stacked run's
    # block{b}_/agg_ prefixes, ADR-0003).
    diag = run["diagnostics"]
    if diag is not None:
        n_heads_expected = run["config"].quantum.num_heads
        for prefix, _suffix, label in _diag_stages(diag):
            key = f"{prefix}lcu_coeffs"
            if key not in diag:
                continue
            arr = np.asarray(diag[key])
            if arr.shape[0] != n_heads_expected:
                warnings.warn(
                    f"LCU coeffs shape mismatch{label}: got {arr.shape[0]} "
                    f"heads, config has {n_heads_expected}",
                    RuntimeWarning,
                )


# ---------------------------------------------------------------------------
# Multi-run sample-efficiency mode
# ---------------------------------------------------------------------------


def multi_run_sample_efficiency(root: Path) -> None:
    out_dir = root / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = sorted(root.glob("results/runs/full_fashionmnist_*"))
    if not runs:
        runs = sorted(Path("results/runs").glob("full_fashionmnist_*"))
    if not runs:
        print("No runs found under results/runs/ — skipping sample-efficiency plot")
        return
    points: list[tuple[float, float, str]] = []
    for run_dir in runs:
        h = run_dir / "history.json"
        c = run_dir / "config.json"
        if not (h.is_file() and c.is_file()):
            continue
        with open(h) as f:
            history = json.load(f)
        if not history["epoch"]["test_acc"]:
            continue
        best = history["meta"].get("best_test_acc") or max(history["epoch"]["test_acc"])
        # Derive train_fraction by recovering len(train_loader) from the log if
        # available; otherwise default to 1.0.
        train_size_log = run_dir / "logs" / "train.log"
        fraction = 1.0
        if train_size_log.is_file():
            for line in train_size_log.read_text().splitlines()[:40]:
                if "--train-fraction" in line:
                    try:
                        toks = line.split("--train-fraction")[-1].split()
                        fraction = float(toks[0])
                    except (ValueError, IndexError):
                        pass
                    break
        points.append((fraction, best, run_dir.name))
    if not points:
        print("No populated runs found — skipping")
        return
    points.sort()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    ax.plot(xs, ys, marker="o")
    ax.set_xscale("log")
    ax.set_xlabel("Train fraction")
    ax.set_ylabel("Best test accuracy")
    ax.set_title("Sample-efficiency sweep")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    out = root / "results" / "figures" / "sample_efficiency.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  ✓ {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, default=None,
                        help="Path to a results/runs/full_fashionmnist_<ts>/ directory.")
    parser.add_argument("--epoch", type=str, default="best",
                        help="'best', 'final', or an integer epoch.")
    parser.add_argument("--multi-run", action="store_true",
                        help="Scan for sibling runs and emit sample_efficiency.png.")
    parser.add_argument("--full-inference", action="store_true",
                        help="Rebuild the model from the chosen checkpoint and run "
                             "inference on the full test set instead of reading "
                             "saved predictions/readouts/test_images from disk. "
                             "Slow — use this for older runs that don't have the "
                             "saved artefacts, or to sanity-check the saved data.")
    parser.add_argument("--check-parity", action="store_true",
                        help="Load both the saved-file predictions and a fresh "
                             "full-inference pass, print max/mean |Δ y_probs| "
                             "and prediction agreement, then exit. Safety net "
                             "to verify the two paths agree.")
    args = parser.parse_args()

    # Line-buffer stdout so progress streams live even when piped / redirected /
    # captured (incl. as report_sweep's subprocess) — non-TTY block-buffers by
    # default, making the run look hung.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    if args.multi_run and not args.run_dir:
        multi_run_sample_efficiency(Path("."))
        return

    if not args.run_dir:
        parser.error("--run-dir is required unless --multi-run is used")

    print(f"Loading run from {args.run_dir} (epoch={args.epoch})")
    run = load_run(Path(args.run_dir), args.epoch)
    mode = "full-inference" if args.full_inference else "file-only"
    print(f"  config: {run['config'].name}  |  epoch: {run['epoch']}  "
          f"|  mode: {mode}")
    sanity_checks(run)

    if args.check_parity:
        _check_parity(run)
        return

    # Figures that only need history.json / .npz files
    fast_plots = [
        ("training_curves",                plot_training_curves),
        ("confusion_matrix_evolution",     plot_confusion_matrix_evolution),
        ("per_class_metrics_table",        write_per_class_metrics_table),
        ("top_k_accuracy",                 plot_top_k_accuracy),
        ("calibration_reliability",        plot_calibration_reliability),
        ("hypernet_gate_param_histograms", plot_hypernet_gate_histograms),
        ("cvqnn_param_values",             plot_cvqnn_param_values),
        ("photon_number_per_mode",         plot_photon_number_per_mode),
        ("state_norm_histogram",           plot_state_norm_histogram),
        ("success_prob_histogram",         plot_success_prob_histogram),
        ("success_prob_trajectory",        plot_success_prob_trajectory),
        ("lcu_coefficients_heatmap",       plot_lcu_coefficients_heatmap),
        ("polynomial_coefficients_trajectory",
         plot_polynomial_coefficient_trajectory),
    ]
    for name, fn in fast_plots:
        try:
            fn(run)
        except Exception as e:
            warnings.warn(f"{name} failed: {type(e).__name__}: {e}", RuntimeWarning)

    # Slow figures that need patches/readouts. In file-only mode they're
    # synthesised from saved npz; under --full-inference we re-run the model.
    recomputed: dict | None
    if args.full_inference:
        try:
            recomputed = _load_model_and_run_inference(run)
        except Exception as e:
            warnings.warn(
                f"_load_model_and_run_inference failed: {type(e).__name__}: {e}",
                RuntimeWarning,
            )
            recomputed = None
    else:
        recomputed = _load_artefacts_recomputed(run)
        if recomputed is None:
            print("  - predictions/test_images.npz or readouts column missing "
                  "in saved artefacts; re-run full_experiment.py or pass "
                  "--full-inference to draw the misclassification gallery "
                  "and t-SNE plot.")

    for name, fn in [
        ("misclassification_gallery", plot_misclassification_gallery),
        ("embedding_tsne",            plot_embedding_tsne),
    ]:
        try:
            fn(run, recomputed)
        except Exception as e:
            warnings.warn(f"{name} failed: {type(e).__name__}: {e}", RuntimeWarning)

    print(f"\nAll figures written to {run['fig_dir']}/")


if __name__ == "__main__":
    main()

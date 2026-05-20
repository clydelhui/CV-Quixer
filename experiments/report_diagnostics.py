"""Post-hoc diagnostic figure generator for a `full_experiment.py` run.

Consumes the artefacts already written by `full_experiment.py`
(config.json, history.json, checkpoints/, predictions/, diagnostics/) and
emits the qualitative + quantum-specific figures that don't fit into the
training loop's per-epoch save pass.

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
import warnings
from pathlib import Path

import dacite
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from cv_quixer.config.schema import ExperimentConfig
from cv_quixer.evaluation.labels import class_names

# Heavy / torch-dependent imports (build_model, PatchedDataset, DataLoader,
# torch itself) are intentionally moved into _load_model_and_run_inference()
# below so the default --no-flag path stays fast and works on machines without
# a configured PyTorch backend. cv_quixer.evaluation.labels is torch-free so
# importing class_names at module scope is safe.


# ---------------------------------------------------------------------------
# Loading + helpers
# ---------------------------------------------------------------------------


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
    config: ExperimentConfig = dacite.from_dict(
        data_class=ExperimentConfig,
        data=config_dict,
        config=dacite.Config(strict=False),
    )

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

    return {
        "run_dir":        run_dir,
        "fig_dir":        run_dir / "figures",
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
    needed_pred_keys = {"y_true", "y_pred", "y_probs", "readouts"}
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


def plot_confusion_matrix_evolution(run: dict) -> None:
    eh = run["history"]["epoch"]
    cms = eh.get("test_confusion")
    if not cms:
        print("  - test_confusion missing → skipping confusion_matrix_evolution")
        return
    classes = class_names(run["config"])
    n_epochs = len(cms)
    cols = min(4, n_epochs)
    rows = (n_epochs + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 3.5 * rows),
                             squeeze=False)
    for idx, ax in enumerate(axes.flat):
        if idx >= n_epochs:
            ax.axis("off")
            continue
        cm = np.asarray(cms[idx], dtype=np.float64)
        cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_title(f"epoch {idx + 1}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
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


def plot_hypernet_gate_histograms(run: dict) -> None:
    diag = run["diagnostics"]
    if diag is None:
        print("  - diagnostics/ missing → skipping hypernet_gate_param_histograms")
        return
    # Group keys: head{i}_{gate_name}
    gate_names = sorted({
        k.split("_", 1)[1] for k in diag.keys()
        if k.startswith("head") and "_" in k and not k.endswith("state_norms")
        and not k.startswith("mean_photon_number")
    })
    # Drop "state_norms" / "mean_photon_number" leftovers (safety)
    gate_names = [g for g in gate_names if g not in ("state_norms",)]

    num_heads = len({k[:5] for k in diag.keys() if k.startswith("head")})
    if num_heads == 0 or not gate_names:
        print("  - no gate-param arrays in diagnostics/ → skipping hypernet histograms")
        return

    n_gates = len(gate_names)
    fig, axes = plt.subplots(num_heads, n_gates,
                             figsize=(2.2 * n_gates, 2.0 * num_heads),
                             squeeze=False)
    for h in range(num_heads):
        for g_idx, gname in enumerate(gate_names):
            key = f"head{h}_{gname}"
            ax = axes[h, g_idx]
            if key not in diag or diag[key].size == 0:
                ax.set_visible(False)
                continue
            vals = diag[key].reshape(-1)
            ax.hist(vals, bins=40, color="tab:blue", alpha=0.7)
            if h == 0:
                ax.set_title(gname, fontsize=8)
            if g_idx == 0:
                ax.set_ylabel(f"head {h}", fontsize=8)
            ax.tick_params(axis="both", labelsize=6)
    fig.suptitle(f"Hypernetwork gate-parameter distributions (epoch {run['epoch']})")
    fig.tight_layout()
    fig.savefig(run["fig_dir"] / "hypernet_gate_param_histograms.png", dpi=150)
    plt.close(fig)
    print("  ✓ hypernet_gate_param_histograms.png")


def plot_photon_number_per_mode(run: dict) -> None:
    eh = run["history"]["epoch"]
    if not eh.get("mean_photon_number"):
        print("  - mean_photon_number missing → skipping photon_number_per_mode")
        return
    chosen = run["epoch"] - 1
    entry = eh["mean_photon_number"][chosen]
    if entry is None:
        print("  - mean_photon_number entry for chosen epoch is null → skipping")
        return
    arr = np.asarray(entry)        # (num_heads, num_modes)
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
    eh = run["history"]["epoch"]
    if not eh.get("lcu_coeffs"):
        print("  - lcu_coeffs missing → skipping lcu_coefficients_heatmap")
        return
    chosen = run["epoch"] - 1
    arr = np.asarray(eh["lcu_coeffs"][chosen])   # (num_heads, num_patches, 2)
    magnitude = np.sqrt((arr ** 2).sum(axis=-1))  # (num_heads, num_patches)
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(magnitude, aspect="auto", cmap="viridis")
    ax.set_xlabel("Patch index")
    ax.set_ylabel("Head")
    ax.set_title(f"LCU coefficient magnitudes |b_i| (epoch {run['epoch']})")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(run["fig_dir"] / "lcu_coefficients_heatmap.png", dpi=150)
    plt.close(fig)
    print("  ✓ lcu_coefficients_heatmap.png")


def plot_polynomial_coefficient_trajectory(run: dict) -> None:
    eh = run["history"]["epoch"]
    if not eh.get("poly_coeffs"):
        print("  - poly_coeffs missing → skipping polynomial_coefficients_trajectory")
        return
    arr = np.asarray(eh["poly_coeffs"])   # (n_epochs, num_heads, degree+1)
    n_epochs, num_heads, degree_plus_1 = arr.shape
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
    fig.suptitle("Polynomial coefficient trajectory")
    fig.tight_layout()
    fig.savefig(run["fig_dir"] / "polynomial_coefficients_trajectory.png", dpi=150)
    plt.close(fig)
    print("  ✓ polynomial_coefficients_trajectory.png")


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------


def sanity_checks(run: dict) -> None:
    eh = run["history"]["epoch"]
    # Pick whichever source we have on disk as the authoritative N for the
    # confusion matrix sanity check.
    expected_n = None
    if run["test_images"] is not None and "images" in run["test_images"]:
        expected_n = int(run["test_images"]["images"].shape[0])
    elif run["predictions"] is not None and "y_true" in run["predictions"]:
        expected_n = int(run["predictions"]["y_true"].shape[0])
    if eh.get("test_confusion") and expected_n is not None:
        cm = np.asarray(eh["test_confusion"][-1], dtype=np.int64)
        total = int(cm.sum())
        if total != expected_n:
            warnings.warn(
                f"Confusion matrix row sum {total} != saved test set size "
                f"{expected_n} (this may be expected if the test set was "
                "sub-sampled at training time and the resulting npz reflects "
                "that subset).",
                RuntimeWarning,
            )

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
    if eh.get("lcu_coeffs"):
        arr = np.asarray(eh["lcu_coeffs"][0])
        n_heads_expected = run["config"].quantum.num_heads
        if arr.shape[0] != n_heads_expected:
            warnings.warn(
                f"LCU coeffs shape mismatch: got {arr.shape[0]} heads, "
                f"config has {n_heads_expected}",
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
        ("confusion_matrix_evolution",     plot_confusion_matrix_evolution),
        ("per_class_metrics_table",        write_per_class_metrics_table),
        ("top_k_accuracy",                 plot_top_k_accuracy),
        ("calibration_reliability",        plot_calibration_reliability),
        ("hypernet_gate_param_histograms", plot_hypernet_gate_histograms),
        ("photon_number_per_mode",         plot_photon_number_per_mode),
        ("state_norm_histogram",           plot_state_norm_histogram),
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

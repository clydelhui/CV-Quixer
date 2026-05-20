"""Backfill the artefacts that the current `report_diagnostics.py` consumes for
an older `full_experiment.py` run that predates the diagnostics/predictions
schema.

For each epoch checkpoint in ``<run-dir>/checkpoints/epoch_NNNN.pt``, this
script reproduces the post-epoch evaluation + quantum-diagnostic passes that
the modern training loop performs, writing:

    <run-dir>/subset_indices.npz                # full-test indices + diag perm
    <run-dir>/predictions/test_images.npz       # reassembled (N, H, W) images
    <run-dir>/predictions/epoch_NNNN.npz        # y_true, y_pred, y_probs, readouts
    <run-dir>/diagnostics/epoch_NNNN.npz       # gate-param samples, state norms,
                                                # mean photon number
    <run-dir>/history.json                      # patched in-place with the
                                                # missing per-epoch diagnostic
                                                # series (train/test per-class
                                                # acc, confusion, test trunc
                                                # loss, lcu/poly coeffs,
                                                # hypernet stats, mean photon).

The base series already in ``history.json`` (train/test loss+acc, trunc_loss,
elapsed_sec) are preserved untouched — they reflect training-time state that
cannot be replayed.

Usage:
    uv run python experiments/backfill_artefacts.py \\
        --run-dir results/runs/full_fashionmnist_2026-05-15_01-55-34/

    # Force a re-evaluation even if predictions/diagnostics already exist
    uv run python experiments/backfill_artefacts.py \\
        --run-dir results/runs/<run>/ --overwrite

    # Backfill only specific epochs
    uv run python experiments/backfill_artefacts.py \\
        --run-dir results/runs/<run>/ --epochs 2 3
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import warnings
from pathlib import Path

import dacite
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from cv_quixer.config.schema import ExperimentConfig
from cv_quixer.data.mnist import PatchedDataset
from cv_quixer.evaluation.diagnostics import (
    ensure_history_schema,
    evaluate,
    quantum_diagnostics,
    save_test_images_once,
    snapshot_coefficients,
)
from cv_quixer.models import build_model


DIAG_SIZE_DEFAULT = 512
SUBSET_SEED_DEFAULT = 42


def _pick_device(arg: str) -> torch.device:
    if arg == "cuda":
        return torch.device("cuda")
    if arg == "mps":
        return torch.device("mps")
    if arg == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _discover_epoch_checkpoints(ckpt_dir: Path) -> list[tuple[int, Path]]:
    """Return sorted (epoch, path) pairs for every ``epoch_NNNN.pt``."""
    pat = re.compile(r"epoch_(\d{4})\.pt$")
    found: list[tuple[int, Path]] = []
    for p in ckpt_dir.glob("epoch_*.pt"):
        m = pat.search(p.name)
        if m:
            found.append((int(m.group(1)), p))
    found.sort()
    return found


def _save_subset_indices(
    out_path: Path,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    diag_indices: np.ndarray,
) -> None:
    """Idempotent: skip if the file already exists with matching sizes."""
    if out_path.is_file():
        with np.load(out_path) as existing:
            if (
                existing["train_indices"].size == train_indices.size
                and existing["test_indices"].size == test_indices.size
                and existing["diag_indices"].size == diag_indices.size
            ):
                return
    np.savez_compressed(
        out_path,
        train_indices=train_indices,
        test_indices=test_indices,
        diag_indices=diag_indices,
    )


def _set_epoch_entry(history: dict, key: str, epoch: int, value) -> None:
    """Place ``value`` at position ``epoch - 1`` in ``history.epoch[key]``,
    padding with None if the list is shorter and overwriting any existing
    entry at that position."""
    lst = history["epoch"].setdefault(key, [])
    while len(lst) < epoch - 1:
        lst.append(None)
    if len(lst) == epoch - 1:
        lst.append(value)
    else:
        lst[epoch - 1] = value


def _atomic_write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=str,
                        help="Path to a results/runs/full_fashionmnist_<ts>/ "
                             "directory to backfill in place.")
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cpu", "mps", "cuda"],
                        help="Compute device (default: auto-detect).")
    parser.add_argument("--diag-size", type=int, default=DIAG_SIZE_DEFAULT,
                        help=f"Diagnostic subset size (default: {DIAG_SIZE_DEFAULT}).")
    parser.add_argument("--subset-seed", type=int, default=SUBSET_SEED_DEFAULT,
                        help="Seed matching full_experiment.py --subset-seed "
                             "(diag generator uses seed+1).")
    parser.add_argument("--epochs", nargs="*", type=int, default=None,
                        help="Backfill only these epoch numbers (default: "
                             "every epoch_NNNN.pt in checkpoints/).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-run evaluation even if predictions/ and "
                             "diagnostics/ npz files already exist.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.is_dir():
        parser.error(f"--run-dir {run_dir} does not exist")

    ckpt_dir = run_dir / "checkpoints"
    preds_dir = run_dir / "predictions"
    diag_dir = run_dir / "diagnostics"
    preds_dir.mkdir(exist_ok=True)
    diag_dir.mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # Config + history
    # ------------------------------------------------------------------
    config_path = run_dir / "config.json"
    history_path = run_dir / "history.json"
    if not config_path.is_file():
        sys.exit(f"config.json not found in {run_dir}")
    if not history_path.is_file():
        sys.exit(f"history.json not found in {run_dir}")

    with open(config_path) as f:
        config_dict = json.load(f)
    config: ExperimentConfig = dacite.from_dict(
        data_class=ExperimentConfig,
        data=config_dict,
        config=dacite.Config(strict=False),
    )
    with open(history_path) as f:
        history = json.load(f)
    ensure_history_schema(history)

    # ------------------------------------------------------------------
    # Device + datasets + loaders (mirrors full_experiment.py:257-322)
    # ------------------------------------------------------------------
    device = _pick_device(args.device)
    print(f"Using device: {device}")

    train_ds_full = PatchedDataset(config.data, train=True)
    test_ds_full = PatchedDataset(config.data, train=False)

    # This script assumes the parent run used the full splits — older
    # full_experiment.py versions didn't write subset_indices.npz, so we
    # cannot recover a sub-sampled split. If subset_indices.npz already
    # exists, honour it.
    subset_path = run_dir / "subset_indices.npz"
    if subset_path.is_file():
        with np.load(subset_path) as si:
            train_indices = si["train_indices"].astype(np.int64)
            test_indices = si["test_indices"].astype(np.int64)
            diag_indices = si["diag_indices"].astype(np.int64)
        train_ds = Subset(train_ds_full, indices=train_indices.tolist())
        test_ds = Subset(test_ds_full, indices=test_indices.tolist())
        # diag_indices are absolute into test_ds_full
        diag_indices_in_test = diag_indices.tolist()
        diag_ds = Subset(test_ds_full, indices=diag_indices_in_test)
        print(f"Reusing subset_indices.npz: "
              f"train={len(train_ds):,}  test={len(test_ds):,}  "
              f"diag={len(diag_ds):,}")
    else:
        train_ds = train_ds_full
        test_ds = test_ds_full
        diag_size = min(args.diag_size, len(test_ds))
        diag_g = torch.Generator().manual_seed(args.subset_seed + 1)
        diag_indices_in_test = torch.randperm(
            len(test_ds), generator=diag_g
        )[:diag_size].tolist()
        diag_ds = Subset(test_ds, indices=diag_indices_in_test)

        train_indices = np.arange(len(train_ds_full), dtype=np.int64)
        test_indices = np.arange(len(test_ds_full), dtype=np.int64)
        diag_indices = np.asarray(diag_indices_in_test, dtype=np.int64)
        _save_subset_indices(subset_path, train_indices, test_indices, diag_indices)
        print(f"Wrote {subset_path.name} "
              f"(train={train_indices.size:,}  test={test_indices.size:,}  "
              f"diag={diag_indices.size:,})")

    batch_size = config.data.batch_size
    train_eval_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    diag_loader = DataLoader(diag_ds, batch_size=batch_size, shuffle=False)

    save_test_images_once(
        test_loader,
        config.data.image_size,
        config.data.patch_size,
        preds_dir / "test_images.npz",
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = build_model(config).to(device)

    # ------------------------------------------------------------------
    # Per-epoch backfill
    # ------------------------------------------------------------------
    available = _discover_epoch_checkpoints(ckpt_dir)
    if not available:
        sys.exit(f"No epoch_NNNN.pt checkpoints found in {ckpt_dir}")
    if args.epochs:
        wanted = set(args.epochs)
        available = [(e, p) for (e, p) in available if e in wanted]
        if not available:
            sys.exit(f"None of the requested epochs {args.epochs} matched "
                     f"existing checkpoints in {ckpt_dir}")

    print(f"Backfilling {len(available)} epoch(s): "
          f"{[e for e, _ in available]}")

    for epoch, ckpt_path in available:
        pred_path = preds_dir / f"epoch_{epoch:04d}.npz"
        diag_path = diag_dir / f"epoch_{epoch:04d}.npz"
        if (pred_path.is_file() and diag_path.is_file() and not args.overwrite):
            print(f"  epoch {epoch}: predictions + diagnostics already exist → "
                  f"skipping (pass --overwrite to redo)")
            continue

        print(f"  epoch {epoch}: loading {ckpt_path.name}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        # Train eval — populates per-class acc + confusion only (no npz)
        train_eval = evaluate(model, train_eval_loader, device,
                              num_classes=config.data.num_classes)
        _set_epoch_entry(history, "train_per_class_acc", epoch,
                         train_eval["per_class_acc"].tolist())
        _set_epoch_entry(history, "train_confusion", epoch,
                         train_eval["confusion"].tolist())

        # Test eval — full predictions npz + history entries
        test_eval = evaluate(model, test_loader, device,
                             num_classes=config.data.num_classes)
        _set_epoch_entry(history, "test_per_class_acc", epoch,
                         test_eval["per_class_acc"].tolist())
        _set_epoch_entry(history, "test_confusion", epoch,
                         test_eval["confusion"].tolist())
        _set_epoch_entry(history, "test_trunc_loss", epoch,
                         float(test_eval["trunc_loss"]))
        np.savez_compressed(
            pred_path,
            y_true=test_eval["y_true"],
            y_pred=test_eval["y_pred"],
            y_probs=test_eval["y_probs"],
            readouts=test_eval["readouts"],
        )

        # LCU + polynomial coefficient snapshot
        lcu_snap, poly_snap = snapshot_coefficients(model)
        _set_epoch_entry(history, "lcu_coeffs", epoch, lcu_snap.tolist())
        _set_epoch_entry(history, "poly_coeffs", epoch, poly_snap.tolist())

        # Quantum diagnostics — wrap in try/except like full_experiment.py
        try:
            stats_summary, mean_photon, diag_raw = quantum_diagnostics(
                model, diag_loader, device
            )
            _set_epoch_entry(history, "hypernet_stats", epoch, stats_summary)
            _set_epoch_entry(history, "mean_photon_number", epoch,
                             mean_photon.tolist())
            np.savez_compressed(diag_path, **diag_raw)
        except Exception as e:
            warnings.warn(
                f"quantum_diagnostics failed at epoch {epoch}: "
                f"{type(e).__name__}: {e}. Skipping diagnostic outputs.",
                RuntimeWarning,
            )
            _set_epoch_entry(history, "hypernet_stats", epoch, None)
            _set_epoch_entry(history, "mean_photon_number", epoch, None)

        # Persist incrementally so a kill mid-loop still leaves usable state
        _atomic_write_json(history_path, history)
        print(f"  epoch {epoch}: done "
              f"(test acc {test_eval['acc']:.4f}, loss {test_eval['loss']:.4f})")

    print(f"\nBackfill complete. Run "
          f"`uv run python experiments/report_diagnostics.py "
          f"--run-dir {run_dir} --epoch best` to generate the figures.")


if __name__ == "__main__":
    main()

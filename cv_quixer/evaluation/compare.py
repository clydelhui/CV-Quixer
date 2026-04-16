"""Utilities for side-by-side comparison of classical and quantum model results.

Used by experiments/compare_models.py to produce the thesis-ready tables and
figures showing the two models' performance under equivalent conditions.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch


def load_history(checkpoint_path: str | Path) -> dict:
    """Load the training history dict from a saved checkpoint.

    Args:
        checkpoint_path: Path to a .pt checkpoint saved by Trainer.

    Returns:
        history dict with keys train_loss, train_acc, test_loss, test_acc.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    return ckpt["history"]


def print_comparison_table(
    classical_history: dict,
    quantum_history: dict,
    classical_params: int,
    quantum_params: int,
) -> None:
    """Print a comparison table suitable for copying into a thesis."""
    best_cls_acc = max(classical_history["test_acc"])
    best_qnt_acc = max(quantum_history["test_acc"])

    print("\n" + "=" * 60)
    print(f"{'Metric':<30} {'Classical ViT':>14} {'CV-Quixer':>14}")
    print("=" * 60)
    print(f"{'Best test accuracy':<30} {best_cls_acc:>14.4f} {best_qnt_acc:>14.4f}")
    print(f"{'Trainable parameters':<30} {classical_params:>14,} {quantum_params:>14,}")
    print("=" * 60 + "\n")


def plot_training_curves(
    classical_history: dict,
    quantum_history: dict,
    save_path: str | Path | None = None,
) -> None:
    """Plot loss and accuracy curves for both models on a single figure.

    Args:
        classical_history: History dict from the classical ViT run.
        quantum_history:   History dict from the CV-Quixer run.
        save_path:         If provided, save the figure here instead of showing.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(classical_history["test_acc"]) + 1)

    # Accuracy curves
    axes[0].plot(epochs, classical_history["test_acc"], label="Classical ViT")
    axes[0].plot(epochs, quantum_history["test_acc"], label="CV-Quixer")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Test Accuracy")
    axes[0].set_title("Test Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss curves
    axes[1].plot(epochs, classical_history["test_loss"], label="Classical ViT")
    axes[1].plot(epochs, quantum_history["test_loss"], label="CV-Quixer")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Test Loss")
    axes[1].set_title("Test Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    else:
        plt.show()

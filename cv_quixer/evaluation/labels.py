"""Class-name constants shared by the producer (full_experiment.py) and the
post-hoc tools (report_diagnostics.py, eval_cutoff_sweep.py).

This module is **deliberately torch-free** so report_diagnostics.py — which
keeps torch out of its top-level imports so the default file-only path runs
without a configured PyTorch backend — can import from it at module scope
without paying the torch import cost.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cv_quixer.config.schema import ExperimentConfig


FASHIONMNIST_CLASSES: tuple[str, ...] = (
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
)
MNIST_CLASSES: tuple[str, ...] = tuple(str(d) for d in range(10))


def class_names(config: "ExperimentConfig") -> tuple[str, ...]:
    """Return the class-name tuple for the dataset selected in `config`."""
    return MNIST_CLASSES if config.data.dataset == "mnist" else FASHIONMNIST_CLASSES

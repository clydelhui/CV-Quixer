"""Evaluation metrics for classification experiments."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix


def accuracy(predictions: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute classification accuracy.

    Args:
        predictions: Predicted class indices (batch_size,).
        labels:      Ground-truth class indices (batch_size,).

    Returns:
        Accuracy as a float in [0, 1].
    """
    return (predictions == labels).float().mean().item()


def get_confusion_matrix(
    predictions: torch.Tensor | np.ndarray,
    labels: torch.Tensor | np.ndarray,
    num_classes: int = 10,
) -> np.ndarray:
    """Return a (num_classes × num_classes) confusion matrix.

    Rows = true labels, columns = predicted labels.
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    return confusion_matrix(labels, predictions, labels=list(range(num_classes)))


def classification_summary(
    predictions: torch.Tensor | np.ndarray,
    labels: torch.Tensor | np.ndarray,
) -> str:
    """Return a sklearn classification report string (precision, recall, F1)."""
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    return classification_report(labels, predictions, digits=4)

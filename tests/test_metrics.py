"""Tests for evaluation metrics."""

import numpy as np
import torch
import pytest

from cv_quixer.evaluation.metrics import accuracy, get_confusion_matrix, classification_summary


class TestAccuracy:
    def test_perfect_accuracy(self):
        preds = torch.tensor([0, 1, 2, 3])
        labels = torch.tensor([0, 1, 2, 3])
        assert accuracy(preds, labels) == pytest.approx(1.0)

    def test_zero_accuracy(self):
        preds = torch.tensor([1, 2, 3, 0])
        labels = torch.tensor([0, 1, 2, 3])
        assert accuracy(preds, labels) == pytest.approx(0.0)

    def test_partial_accuracy(self):
        preds = torch.tensor([0, 1, 9, 9])
        labels = torch.tensor([0, 1, 2, 3])
        assert accuracy(preds, labels) == pytest.approx(0.5)


class TestConfusionMatrix:
    def test_shape(self):
        preds = torch.tensor([0, 1, 2])
        labels = torch.tensor([0, 1, 2])
        cm = get_confusion_matrix(preds, labels, num_classes=10)
        assert cm.shape == (10, 10)

    def test_diagonal_on_perfect_predictions(self):
        preds = np.array([0, 1, 2, 3, 4])
        labels = np.array([0, 1, 2, 3, 4])
        cm = get_confusion_matrix(preds, labels, num_classes=5)
        assert np.all(cm == np.eye(5, dtype=int))

    def test_sum_equals_total_samples(self):
        preds = torch.tensor([0, 0, 1, 2])
        labels = torch.tensor([0, 1, 1, 2])
        cm = get_confusion_matrix(preds, labels, num_classes=3)
        assert cm.sum() == 4


class TestClassificationSummary:
    def test_returns_string(self):
        preds = torch.tensor([0, 1, 2])
        labels = torch.tensor([0, 1, 2])
        report = classification_summary(preds, labels)
        assert isinstance(report, str)
        assert "precision" in report

"""Tests for the shared per-epoch artefacts module
(``cv_quixer.evaluation.epoch_artefacts``) + the torch-free key/filename
contract (``cv_quixer.evaluation.artefact_schema``).

These pin the two drifts that existed between the live writers before the module
existed (eval_cutoff_sweep wrote neither ``cvqnn_params`` nor the stacked
block-prefixed keys; the history field set omitted the cvqnn/query trunc
streams), and the byte-level npz contract the reader depends on.

Small circuits (num_modes=2, cutoff_dim=4) keep simulation tractable; the
coefficient-dispatch tests pass ``diag_loader=None`` so they exercise the
snapshot dispatch without running a full quantum-diagnostics pass.
"""

import subprocess
import sys

import numpy as np
import pytest
import torch

from cv_quixer.config.schema import ExperimentConfig, QuantumConfig
from cv_quixer.evaluation import artefact_schema as schema
from cv_quixer.evaluation.epoch_artefacts import (
    EpochArtefacts,
    build_epoch_artefacts,
    eval_epoch_metrics,
)
from cv_quixer.models import build_model

CPU = torch.device("cpu")


def _canonical_config(**overrides) -> QuantumConfig:
    base = dict(
        num_modes=2, num_layers=1, cutoff_dim=4, num_heads=2,
        cnn_channels_1=4, cnn_channels_2=8, cnn_kernel_size=3,
        decoder_hidden_dim=16, poly_degree=2, dtype="complex64",
        trunc_penalty="norm",
    )
    base.update(overrides)
    return QuantumConfig(**base)


def _stacked_config(**overrides) -> QuantumConfig:
    return _canonical_config(**overrides)


def _fake_eval(n: int = 5, num_classes: int = 10, num_heads: int = 2,
               *, with_success_probs: bool = True) -> dict:
    """A stand-in for an ``evaluate()`` result: the scalar metrics plus the
    per-sample arrays the predictions npz carries."""
    rng = np.random.default_rng(0)
    d = {
        "loss": 0.5, "acc": 0.8,
        "trunc_loss": 0.01, "cvqnn_trunc_loss": 0.002, "query_trunc_loss": 0.003,
        "y_true":  rng.integers(0, num_classes, n).astype(np.int64),
        "y_pred":  rng.integers(0, num_classes, n).astype(np.int64),
        "y_probs": rng.random((n, num_classes)).astype(np.float32),
        "readouts": rng.random((n, num_heads * 2)).astype(np.float32),
    }
    if with_success_probs:
        d["success_probs"] = rng.random((n, num_heads)).astype(np.float32)
    return d


# ---------------------------------------------------------------------------
# Coefficient-snapshot dispatch (the npz key-set drift)
# ---------------------------------------------------------------------------


class TestCoefficientDispatch:
    def test_w_enabled_canonical_writes_cvqnn_params(self, tiny_data_config):
        model = build_model(ExperimentConfig(
            model="quantum", data=tiny_data_config,
            quantum=_canonical_config(cvqnn_num_layers=1)))
        art = build_epoch_artefacts(model, CPU, test_eval=_fake_eval(),
                                    diag_loader=None)
        assert schema.LCU_COEFFS in art.diagnostics
        assert schema.POLY_COEFFS in art.diagnostics
        assert schema.CVQNN_PARAMS in art.diagnostics

    def test_w_disabled_omits_cvqnn_params(self, tiny_data_config):
        model = build_model(ExperimentConfig(
            model="quantum", data=tiny_data_config,
            quantum=_canonical_config(cvqnn_num_layers=0)))
        art = build_epoch_artefacts(model, CPU, test_eval=_fake_eval(),
                                    diag_loader=None)
        assert schema.CVQNN_PARAMS not in art.diagnostics

    def test_stacked_writes_block_prefixed_keys(self, tiny_data_config):
        model = build_model(ExperimentConfig(
            model="quantum_stacked", data=tiny_data_config,
            quantum=_stacked_config()))
        art = build_epoch_artefacts(model, CPU, test_eval=_fake_eval(),
                                    diag_loader=None)
        assert "block0_lcu_coeffs" in art.diagnostics
        assert "block0_poly_coeffs" in art.diagnostics
        # The flat canonical keys must NOT appear for a stacked model.
        assert schema.LCU_COEFFS not in art.diagnostics


# ---------------------------------------------------------------------------
# eval_epoch_metrics (the history field-set drift)
# ---------------------------------------------------------------------------


class TestEvalEpochMetrics:
    def test_includes_cvqnn_and_query_trunc(self):
        m = eval_epoch_metrics(_fake_eval(), _fake_eval())
        assert {"train_loss", "train_acc", "test_loss", "test_acc",
                "test_trunc_loss", "test_cvqnn_trunc_loss",
                "test_query_trunc_loss"}.issubset(m)

    def test_no_train_fields_without_train_eval(self):
        m = eval_epoch_metrics(_fake_eval())
        assert "train_loss" not in m and "train_acc" not in m
        assert "test_cvqnn_trunc_loss" in m

    def test_values_are_plain_floats(self):
        m = eval_epoch_metrics(_fake_eval(), _fake_eval())
        assert all(isinstance(v, float) for v in m.values())


# ---------------------------------------------------------------------------
# EpochArtefacts.write round-trip (the on-disk contract)
# ---------------------------------------------------------------------------


class TestWriteRoundTrip:
    def test_writes_expected_files_and_keys(self, tmp_path):
        diagnostics = {
            schema.LCU_COEFFS: np.zeros((2, 3, 2), np.float32),
            schema.POLY_COEFFS: np.zeros((2, 3), np.float32),
        }
        art = EpochArtefacts(
            test_eval=_fake_eval(with_success_probs=True),
            train_eval=_fake_eval(with_success_probs=True),
            diagnostics=diagnostics,
        )
        art.write(tmp_path, epoch=5)

        test_npz = tmp_path / "predictions" / "epoch_0005.npz"
        train_npz = tmp_path / "predictions" / "epoch_0005_train.npz"
        diag_npz = tmp_path / "diagnostics" / "epoch_0005.npz"
        assert test_npz.is_file() and train_npz.is_file() and diag_npz.is_file()

        with np.load(test_npz) as d:
            assert set(schema.PREDICTION_KEYS_REQUIRED).issubset(d.files)
            assert schema.SUCCESS_PROBS in d.files
        with np.load(diag_npz) as d:
            assert schema.LCU_COEFFS in d.files and schema.POLY_COEFFS in d.files

    def test_success_probs_absent_and_no_train_file_when_omitted(self, tmp_path):
        art = EpochArtefacts(
            test_eval=_fake_eval(with_success_probs=False),
            train_eval=None,
            diagnostics={},
        )
        art.write(tmp_path, epoch=1)

        with np.load(tmp_path / "predictions" / "epoch_0001.npz") as d:
            assert schema.SUCCESS_PROBS not in d.files
            assert set(schema.PREDICTION_KEYS_REQUIRED).issubset(d.files)
        assert not (tmp_path / "predictions" / "epoch_0001_train.npz").exists()


# ---------------------------------------------------------------------------
# artefact_schema stays torch-free (reader fast-path invariant)
# ---------------------------------------------------------------------------


def test_artefact_schema_does_not_import_torch():
    # Run in a fresh interpreter so an already-imported torch in this test
    # process can't mask a transitive import.
    code = (
        "import cv_quixer.evaluation.artefact_schema as s, sys; "
        "assert 'torch' not in sys.modules, 'artefact_schema pulled in torch'"
    )
    subprocess.run([sys.executable, "-c", code], check=True)

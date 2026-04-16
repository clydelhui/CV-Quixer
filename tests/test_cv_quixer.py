"""Tests for the CV quantum model components.

These tests use very small circuit parameters (num_modes=2, cutoff_dim=4) to
keep simulation time tractable. The Fock backend scales as cutoff_dim^num_modes,
so large values make tests prohibitively slow.
"""

import torch
import pytest

from cv_quixer.config.schema import DataConfig, QuantumConfig
from cv_quixer.models.base import BaseVisionTransformer
from cv_quixer.models.quantum.cv_encoding import DisplacementEncoding
from cv_quixer.models.quantum.cv_layers import CVLayer, interferometer_param_count


@pytest.fixture
def small_quantum_config():
    return QuantumConfig(
        num_modes=2,
        num_layers=1,
        cutoff_dim=4,
        backend="strawberryfields.fock",
    )


@pytest.fixture
def data_config():
    return DataConfig(image_size=28, patch_size=4, num_classes=10)


class TestDisplacementEncoding:
    def test_scale_parameter_shape(self):
        enc = DisplacementEncoding(patch_dim=16, num_modes=4)
        assert enc.scale.shape == (4,)

    def test_scale_is_trainable(self):
        enc = DisplacementEncoding(patch_dim=16, num_modes=4)
        assert enc.scale.requires_grad


class TestInterferometerParamCount:
    def test_2_modes(self):
        # 2 modes: 1 beamsplitter × 2 params + 2 phase shifts = 4
        assert interferometer_param_count(2) == 4

    def test_4_modes(self):
        # 4 modes: 6 BS × 2 + 4 phase = 16
        assert interferometer_param_count(4) == 16


class TestCVLayer:
    def test_parameter_count_positive(self, small_quantum_config):
        layer = CVLayer(num_modes=2)
        n_params = sum(p.numel() for p in layer.parameters())
        assert n_params > 0

    def test_is_nn_module(self, small_quantum_config):
        import torch.nn as nn
        layer = CVLayer(num_modes=2)
        assert isinstance(layer, nn.Module)


class TestCVQuixer:
    def test_is_base_model(self, small_quantum_config, data_config):
        from cv_quixer.models.quantum.cv_quixer import CVQuixer
        model = CVQuixer(small_quantum_config, data_config)
        assert isinstance(model, BaseVisionTransformer)

    def test_get_num_parameters(self, small_quantum_config, data_config):
        from cv_quixer.models.quantum.cv_quixer import CVQuixer
        model = CVQuixer(small_quantum_config, data_config)
        assert model.get_num_parameters() > 0

    def test_forward_shape(self, small_quantum_config, data_config):
        """Forward pass produces correct output shape (batch × num_classes)."""
        from cv_quixer.models.quantum.cv_quixer import CVQuixer
        model = CVQuixer(small_quantum_config, data_config)
        # Use batch=1, 1 patch, dim=num_modes to keep the circuit tiny
        patches = torch.randn(1, 1, small_quantum_config.num_modes)
        logits = model(patches)
        assert logits.shape == (1, data_config.num_classes)

    def test_no_nan_output(self, small_quantum_config, data_config):
        from cv_quixer.models.quantum.cv_quixer import CVQuixer
        model = CVQuixer(small_quantum_config, data_config)
        patches = torch.randn(1, 1, small_quantum_config.num_modes)
        logits = model(patches)
        assert not torch.isnan(logits).any(), "NaN detected — check Fock cutoff_dim"

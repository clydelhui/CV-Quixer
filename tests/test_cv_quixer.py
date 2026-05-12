"""Tests for the CV quantum model components.

These tests use very small circuit parameters (num_modes=2, cutoff_dim=4) to
keep simulation time tractable. The Fock backend scales as cutoff_dim^num_modes,
so large values make tests prohibitively slow.
"""

import torch
import pytest

from cv_quixer.config.schema import DataConfig, QuantumConfig
from cv_quixer.models.base import BaseVisionTransformer
from cv_quixer.models.quantum.cv_attention import (
    HyperCVAttentionHead,
    LCUSumCoefficients,
    PolynomialCoefficients,
    _gate_param_count,
    norm_truncation_penalty,
    photon_number_penalty,
)
from cv_quixer.models.quantum.cv_encoding import DisplacementEncoding
from cv_quixer.models.quantum.cv_layers import CVLayer, interferometer_param_count
from cv_quixer.models.quantum.cv_quixer import CVQuixer, _param_count_formula
from cv_quixer.quantum import CVCircuit, FockState
from cv_quixer.utils.params import count_parameters


@pytest.fixture
def small_quantum_config():
    return QuantumConfig(
        num_modes=2,
        num_layers=1,
        cutoff_dim=4,
        grad_mode="backprop",
        num_heads=2,
        cnn_channels_1=4,
        cnn_channels_2=8,
        cnn_kernel_size=3,
        decoder_hidden_dim=16,
        poly_degree=2,
        dtype="complex64",
    )


@pytest.fixture
def data_config():
    return DataConfig(image_size=28, patch_size=7, num_classes=10)


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


class TestGateParamCount:
    def test_linear_4_modes(self):
        assert _gate_param_count(4, "linear") == 30   # 6*4 + 2*3

    def test_ring_4_modes(self):
        assert _gate_param_count(4, "ring") == 32     # 6*4 + 2*4

    def test_single_mode(self):
        assert _gate_param_count(1, "linear") == 6    # 6*1 + 0 BS

    def test_linear_2_modes(self):
        assert _gate_param_count(2, "linear") == 14   # 6*2 + 2*1


class TestLCUAndPolyCoeffs:
    def test_lcu_forward_is_complex(self):
        lcu = LCUSumCoefficients(num_patches=10)
        assert lcu().is_complex()
        assert lcu().shape == (10,)

    def test_lcu_initial_value(self):
        lcu = LCUSumCoefficients(num_patches=5)
        assert torch.allclose(lcu().real, torch.full((5,), 0.2))
        assert torch.allclose(lcu().imag, torch.zeros(5))

    def test_poly_forward_is_real(self):
        pc = PolynomialCoefficients(degree=3)
        assert not pc().is_complex()
        assert pc().shape == (4,)

    def test_poly_initial_value(self):
        pc = PolynomialCoefficients(degree=2)
        expected = torch.tensor([1.0, 0.0, 0.0])
        assert torch.allclose(pc(), expected)


class TestHyperCVAttentionHead:
    def test_readout_shape(self, small_quantum_config):
        head = HyperCVAttentionHead(
            patch_size=7, num_patches=16, config=small_quantum_config,
        )
        patches = torch.randn(16, 49)   # 16 patches of 7×7 pixels
        readout, _, _ = head(patches)
        assert readout.shape == (small_quantum_config.num_modes,)

    def test_readout_is_real(self, small_quantum_config):
        head = HyperCVAttentionHead(
            patch_size=7, num_patches=16, config=small_quantum_config,
        )
        readout, _, _ = head(torch.randn(16, 49))
        assert not readout.is_complex()

    def test_state_data_is_tensor(self, small_quantum_config):
        head = HyperCVAttentionHead(
            patch_size=7, num_patches=16, config=small_quantum_config,
        )
        _, state_data, _ = head(torch.randn(16, 49))
        assert isinstance(state_data, torch.Tensor)

    def test_success_prob_positive(self, small_quantum_config):
        head = HyperCVAttentionHead(
            patch_size=7, num_patches=16, config=small_quantum_config,
        )
        _, _, success_prob = head(torch.randn(16, 49))
        assert success_prob.item() > 0


class TestTruncationHelpers:
    def test_norm_penalty_vacuum_is_zero(self):
        state = FockState.vacuum(num_modes=2, cutoff_dim=4)
        assert norm_truncation_penalty(state).item() == pytest.approx(0.0, abs=1e-8)

    def test_norm_penalty_half_amplitude(self):
        state = FockState.vacuum(num_modes=2, cutoff_dim=4)
        half = FockState(state.data * 0.5, 2, 4)
        # ‖ψ‖² = 0.25 → penalty = 0.75
        assert norm_truncation_penalty(half).item() == pytest.approx(0.75, abs=1e-6)

    def test_photon_penalty_vacuum_is_zero(self):
        circuit = CVCircuit(num_modes=2, cutoff_dim=4)
        state = FockState.vacuum(num_modes=2, cutoff_dim=4)
        assert photon_number_penalty(state, circuit).item() == pytest.approx(0.0, abs=1e-8)

    def test_photon_penalty_bounded(self):
        from cv_quixer.quantum.gates.gaussian import displacement_matrix
        circuit = CVCircuit(num_modes=1, cutoff_dim=4)
        state = FockState.vacuum(num_modes=1, cutoff_dim=4)
        D_mat = displacement_matrix(torch.tensor(1.0), 4)
        state = circuit.apply_single_mode_gate(D_mat, 0, state)
        penalty = photon_number_penalty(state, circuit).item()
        assert 0.0 <= penalty <= 1.0


class TestParamCountConsistency:
    def test_formula_matches_actual(self, small_quantum_config, data_config):
        model = CVQuixer(small_quantum_config, data_config)
        actual = count_parameters(model)
        patch_size = data_config.patch_size
        num_patches = (data_config.image_size // patch_size) ** 2
        expected = _param_count_formula(
            patch_size, num_patches,
            small_quantum_config.num_heads, small_quantum_config.num_modes,
            small_quantum_config.cnn_channels_1, small_quantum_config.cnn_channels_2,
            small_quantum_config.cnn_kernel_size,
            small_quantum_config.decoder_hidden_dim, data_config.num_classes,
            small_quantum_config.bs_topology, small_quantum_config.poly_degree,
        )
        assert actual == expected


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
        patch_dim = data_config.patch_size ** 2
        num_patches = (data_config.image_size // data_config.patch_size) ** 2
        patches = torch.randn(1, num_patches, patch_dim)
        logits = model(patches)
        assert logits.shape == (1, data_config.num_classes)

    def test_no_nan_output(self, small_quantum_config, data_config):
        from cv_quixer.models.quantum.cv_quixer import CVQuixer
        model = CVQuixer(small_quantum_config, data_config)
        patch_dim = data_config.patch_size ** 2
        num_patches = (data_config.image_size // data_config.patch_size) ** 2
        patches = torch.randn(1, num_patches, patch_dim)
        logits = model(patches)
        assert not torch.isnan(logits).any(), "NaN detected — check Fock cutoff_dim"

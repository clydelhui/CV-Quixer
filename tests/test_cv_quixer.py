"""Tests for the CV quantum model components.

These tests use very small circuit parameters (num_modes=2, cutoff_dim=4) to
keep simulation time tractable. The Fock backend scales as cutoff_dim^num_modes,
so large values make tests prohibitively slow.
"""

import math

import numpy as np
import torch
import pytest
from unittest.mock import MagicMock, patch

from cv_quixer.config.schema import QuantumConfig, DataConfig
from cv_quixer.models.base import BaseVisionTransformer
from cv_quixer.models.quantum.cv_attention import (
    CNNHypernetwork,
    HyperCVAttention,
    HyperCVAttentionHead,
    LCUSumCoefficients,
    LinearCVHead,
    PolynomialCoefficients,
    SharedCVAttention,
    SharedPatchCNN,
    _GATE_SEQUENCE,
    _INTERFEROMETER_SEQUENCE,
    _bs_pair_count,
    _build_op_plan,
    _gate_param_count,
    _op_plan_param_count,
    norm_truncation_penalty,
    photon_number_penalty,
)
from cv_quixer.quantum.interferometer import interferometer_param_count
from cv_quixer.models.quantum.cv_quixer import (
    CVQuixer,
    CVQuixerOut,
    SharedCVQuixer,
)
from cv_quixer.quantum import CVCircuit, FockState
from cv_quixer.utils.params import count_parameters


@pytest.fixture(scope="module")
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


# data_config is provided by tests/conftest.py (session scope).


class TestInterferometerParamCount:
    def test_2_modes(self):
        # 2 modes: 1 beamsplitter × 2 params + 2 phase shifts = 4
        assert interferometer_param_count(2) == 4

    def test_4_modes(self):
        # 4 modes: 6 BS × 2 + 4 phase = 16
        assert interferometer_param_count(4) == 16


class TestGateParamCount:
    def test_linear_4_modes(self):
        assert _gate_param_count(4, "linear") == 30   # 6*4 + 2*3

    def test_ring_4_modes(self):
        assert _gate_param_count(4, "ring") == 32     # 6*4 + 2*4

    def test_single_mode(self):
        assert _gate_param_count(1, "linear") == 6    # 6*1 + 0 BS

    def test_linear_2_modes(self):
        assert _gate_param_count(2, "linear") == 14   # 6*2 + 2*1

    @pytest.mark.parametrize("m,topo", [
        (1, "linear"),
        (2, "linear"),
        (4, "linear"),
        (4, "ring"),
        (8, "linear"),
        (8, "ring"),
    ])
    def test_derivation_matches_gate_sequence(self, m, topo):
        """_gate_param_count must equal a direct sum over _GATE_SEQUENCE.

        Locks in the contract that _gate_param_count is derived from the
        gate-op list rather than a separate magic formula.
        """
        n_bs = _bs_pair_count(m, topo)
        expected = sum(
            len(op.param_names) * (m if op.site_kind == "mode" else n_bs)
            for op in _GATE_SEQUENCE
        )
        assert _gate_param_count(m, topo) == expected


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
        readout, _, _, _, _, _ = head(patches)
        assert readout.shape == (small_quantum_config.num_modes,)

    def test_readout_is_real(self, small_quantum_config):
        head = HyperCVAttentionHead(
            patch_size=7, num_patches=16, config=small_quantum_config,
        )
        readout, _, _, _, _, _ = head(torch.randn(16, 49))
        assert not readout.is_complex()

    def test_state_data_is_tensor(self, small_quantum_config):
        head = HyperCVAttentionHead(
            patch_size=7, num_patches=16, config=small_quantum_config,
        )
        _, state_data, _, _, _, _ = head(torch.randn(16, 49))
        assert isinstance(state_data, torch.Tensor)

    def test_success_prob_positive(self, small_quantum_config):
        head = HyperCVAttentionHead(
            patch_size=7, num_patches=16, config=small_quantum_config,
        )
        _, _, success_prob, _, _, _ = head(torch.randn(16, 49))
        assert success_prob.item() > 0


class TestPatchTruncLoss:
    """Tests for HyperCVAttentionHead._compute_patch_trunc_loss.

    Uses a mocked hypernetwork to inject known gate parameters, bypassing the
    CNN so the test controls exactly what circuits U_i are applied to the vacuum.

    Gate matrices are analytic Fock-basis sub-isometries (column norms ≤ 1),
    so _compute_patch_trunc_loss returns the physically correct norm-based loss
    1 - ‖U_i|v⟩‖², which is non-zero whenever amplitude leaks to n ≥ cutoff_dim.
    """

    # m=2, linear topology (n_bs = m-1 = 1), total gate params = 6m + 2*n_bs = 14
    # disp_re lives at offset 3m + 2*n_bs = 8
    _M = 2
    _D = 4
    _N_BS = _M - 1
    _TOTAL_PARAMS = 6 * _M + 2 * _N_BS        # 14
    _DISP_RE_OFFSET = 3 * _M + 2 * _N_BS      # 8

    @pytest.fixture
    def head(self):
        config = QuantumConfig(
            num_modes=self._M,
            cutoff_dim=self._D,
            num_heads=1,
            cnn_channels_1=4,
            cnn_channels_2=8,
            cnn_kernel_size=3,
            decoder_hidden_dim=16,
            poly_degree=1,
            dtype="complex64",
            bs_topology="linear",
        )
        return HyperCVAttentionHead(patch_size=7, num_patches=16, config=config)

    def _vacuum_flat(self):
        device = torch.device("cpu")
        return FockState.vacuum(self._M, self._D, device, torch.complex64).data.reshape(-1)

    def test_nonzero_with_large_displacement(self, head):
        """Large real displacement (|α|=2) causes real truncation loss: norm-based
        loss 1 - ‖U|v⟩‖² > 0.5 for D=4, α=2 applied to both modes."""
        m, offset, total = self._M, self._DISP_RE_OFFSET, self._TOTAL_PARAMS

        def _large_disp_all(patches):
            p = torch.zeros(total)
            p[offset:offset + m] = 2.0
            return p.unsqueeze(0).expand(patches.shape[0], -1).clone()

        mock_hn = MagicMock()
        mock_hn.forward_all = MagicMock(side_effect=_large_disp_all)
        # object.__setattr__ bypasses nn.Module's __setattr__, which rejects non-Module values
        object.__setattr__(head, 'hypernetwork', mock_hn)
        state_flat = self._vacuum_flat()
        patches = torch.zeros(16, 49)
        device = torch.device("cpu")

        loss = head._compute_patch_trunc_loss(patches, state_flat, device, torch.complex64)

        assert loss.item() > 0.5, f"Expected norm-based truncation loss > 0.5, got {loss.item():.4f}"
        assert loss.item() <= 1.0, f"Expected norm-based truncation loss <= 1.0, got {loss.item():.4f}"

    def test_near_zero_with_identity_gates(self, head):
        """All-zero gate parameters leave the vacuum unchanged: norm loss
        1 - ‖U|v⟩‖² = 0 since the state is still unit-norm."""
        total = self._TOTAL_PARAMS

        mock_hn = MagicMock()
        mock_hn.forward_all = MagicMock(side_effect=lambda patches: torch.zeros(patches.shape[0], total))
        object.__setattr__(head, 'hypernetwork', mock_hn)
        state_flat = self._vacuum_flat()
        patches = torch.zeros(16, 49)
        device = torch.device("cpu")

        loss = head._compute_patch_trunc_loss(patches, state_flat, device, torch.complex64)

        assert loss.item() == pytest.approx(0.0, abs=1e-5)


class TestPatchParallelEquivalence:
    """Locks the patch-parallel (vmap-over-N) LCU / trunc-loss path against a
    pinned sequential reference. Reduction order changes when summing
    ``Σ_i b_i U_i|v⟩`` as a batched dot product instead of a Python loop, so
    use a tight-but-not-zero tolerance.
    """

    @pytest.fixture
    def head(self):
        config = QuantumConfig(
            num_modes=2,
            cutoff_dim=4,
            num_heads=1,
            cnn_channels_1=4,
            cnn_channels_2=8,
            cnn_kernel_size=3,
            decoder_hidden_dim=16,
            poly_degree=2,
            dtype="complex64",
            bs_topology="linear",
        )
        torch.manual_seed(0)
        return HyperCVAttentionHead(patch_size=7, num_patches=16, config=config)

    @staticmethod
    def _sequential_lcu(head, patches, v, device, dtype):
        """Original sequential reference for _apply_lcu_to_vector — pinned for parity."""
        D, m = head.cutoff_dim, head.num_modes
        b = head.lcu_coeffs().to(device)
        result = torch.zeros_like(v)
        norm_sq_sum = torch.zeros((), device=device, dtype=head._real_dtype)
        for i in range(patches.shape[0]):
            params = head.hypernetwork(patches[i], i).to(device)
            state_i = FockState(v.reshape((D,) * m), m, D)
            out_i = head._apply_patch_gates_to_state(params, state_i, device, dtype)
            norm_sq_sum = norm_sq_sum + (out_i.data.abs() ** 2).sum()
            result = result + b[i].to(dtype) * out_i.data.reshape(-1)
        return result, norm_sq_sum

    def test_lcu_matches_sequential(self, head):
        """``_apply_lcu_to_vector`` matches the pinned sequential reference."""
        torch.manual_seed(1)
        patches = torch.randn(16, 49)
        v = FockState.vacuum(head.num_modes, head.cutoff_dim,
                             torch.device("cpu"), torch.complex64).data.reshape(-1)
        device, dtype = torch.device("cpu"), torch.complex64

        ref_result, ref_norm = self._sequential_lcu(head, patches, v, device, dtype)
        new_result, new_norm = head._apply_lcu_to_vector(
            patches, v, device, dtype, accumulate_norm_sq=True
        )

        assert torch.allclose(new_result, ref_result, atol=1e-5, rtol=1e-5)
        assert torch.allclose(new_norm, ref_norm, atol=1e-5, rtol=1e-5)

    def test_trunc_loss_matches_sequential_mean(self, head):
        """``_compute_patch_trunc_loss`` matches the sequential 1 − Σ‖U_i|v⟩‖²/N."""
        torch.manual_seed(2)
        patches = torch.randn(16, 49)
        v = FockState.vacuum(head.num_modes, head.cutoff_dim,
                             torch.device("cpu"), torch.complex64).data.reshape(-1)
        device, dtype = torch.device("cpu"), torch.complex64

        # Reference: per-patch sequential loop equivalent to the old impl.
        D, m = head.cutoff_dim, head.num_modes
        ref_losses = []
        for i in range(patches.shape[0]):
            params = head.hypernetwork(patches[i], i).to(device)
            state_i = FockState(v.reshape((D,) * m), m, D)
            out_i = head._apply_patch_gates_to_state(params, state_i, device, dtype)
            ref_losses.append(1.0 - (out_i.data.abs() ** 2).sum())
        ref_loss = torch.stack(ref_losses).mean()

        new_loss = head._compute_patch_trunc_loss(patches, v, device, dtype)

        assert torch.allclose(new_loss, ref_loss, atol=1e-5, rtol=1e-5)

    def test_lcu_backward_finite(self, head):
        """Gradient flows through the vmapped LCU path without producing NaNs."""
        torch.manual_seed(3)
        patches = torch.randn(16, 49, requires_grad=False)
        v = FockState.vacuum(head.num_modes, head.cutoff_dim,
                             torch.device("cpu"), torch.complex64).data.reshape(-1)

        out = head._apply_lcu_to_vector(patches, v, torch.device("cpu"), torch.complex64)
        loss = (out.abs() ** 2).sum()
        loss.backward()

        for name, p in head.named_parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), f"non-finite grad in {name}"


class TestDiagnosticsParallelEquivalence:
    """Locks the two vectorised paths inside `quantum_diagnostics` against
    the original Python-loop references — pinned to catch regressions.
    """

    def test_forward_grid_matches_single_patch_forward(self):
        """``CNNHypernetwork.forward_grid`` matches the (B, N) Python loop
        of ``forward(patches[b, i], i)``."""
        torch.manual_seed(0)
        patch_size, num_patches, num_modes = 7, 16, 2
        hn = CNNHypernetwork(
            patch_size=patch_size,
            num_patches=num_patches,
            num_modes=num_modes,
            cnn_channels_1=4,
            cnn_channels_2=8,
            cnn_kernel_size=3,
            bs_topology="linear",
        )
        hn.eval()

        torch.manual_seed(1)
        B = 5
        patches = torch.randn(B, num_patches, patch_size * patch_size)

        with torch.no_grad():
            ref = torch.stack([
                torch.stack([hn.forward(patches[b, i], i) for i in range(num_patches)])
                for b in range(B)
            ])                                              # (B, N, gate_params)
            new = hn.forward_grid(patches)                  # (B, N, gate_params)

        assert ref.shape == new.shape
        assert torch.allclose(new, ref, atol=1e-6, rtol=1e-6)

    def test_photon_number_and_state_norm_tensor_path(self):
        """The (B, D, ..., D) tensor reductions match the per-element
        ``circuit.measure_photon_number`` / ``FockState.norm()`` triple loop.
        """
        torch.manual_seed(2)
        num_modes, cutoff = 2, 6
        B = 7
        # Random complex128 batched state — not normalised; the diagnostics
        # path is supposed to report whatever norm the model produced.
        state_batch = (
            torch.randn(B, cutoff, cutoff, dtype=torch.complex128)
        )
        circuit = CVCircuit(num_modes=num_modes, cutoff_dim=cutoff)

        # Reference: original triple loop.
        ref_photon = [0.0] * num_modes
        ref_norms: list[float] = []
        for b in range(B):
            fs = FockState(state_batch[b], num_modes, cutoff)
            for k in range(num_modes):
                ref_photon[k] += float(circuit.measure_photon_number(k, fs).item())
            ref_norms.append(float(fs.norm().item()))

        # New tensor path (matches diagnostics.py implementation).
        probs = state_batch.abs() ** 2
        ns = torch.arange(cutoff, dtype=probs.dtype)
        new_photon = [0.0] * num_modes
        for k in range(num_modes):
            other_axes = tuple(
                ax for ax in range(1, num_modes + 1) if ax != k + 1
            )
            p_k = probs.sum(dim=other_axes) if other_axes else probs
            mean_n_per_b = (p_k * ns).sum(dim=-1)
            new_photon[k] = float(mean_n_per_b.sum().item())
        new_norms = probs.flatten(start_dim=1).sum(dim=-1).tolist()

        assert torch.allclose(
            torch.tensor(new_photon), torch.tensor(ref_photon),
            atol=1e-10, rtol=1e-10,
        )
        assert torch.allclose(
            torch.tensor(new_norms), torch.tensor(ref_norms),
            atol=1e-10, rtol=1e-10,
        )


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


class TestCVQuixer:
    def test_is_base_model(self, small_quantum_config, data_config):
        from cv_quixer.models.quantum.cv_quixer import CVQuixer
        model = CVQuixer(small_quantum_config, data_config)
        assert isinstance(model, BaseVisionTransformer)

    def test_get_num_parameters(self, small_quantum_config, data_config):
        from cv_quixer.models.quantum.cv_quixer import CVQuixer
        model = CVQuixer(small_quantum_config, data_config)
        assert model.get_num_parameters() > 0

    def test_forward_shape(self, small_quantum_config, tiny_data_config):
        """Forward pass produces correct output shape (batch × num_classes)."""
        from cv_quixer.models.quantum.cv_quixer import CVQuixer
        model = CVQuixer(small_quantum_config, tiny_data_config)
        patch_dim = tiny_data_config.patch_size ** 2
        num_patches = (tiny_data_config.image_size // tiny_data_config.patch_size) ** 2
        patches = torch.randn(1, num_patches, patch_dim)
        logits = model(patches)
        assert logits.shape == (1, tiny_data_config.num_classes)

    def test_no_nan_output(self, small_quantum_config, tiny_data_config):
        from cv_quixer.models.quantum.cv_quixer import CVQuixer
        model = CVQuixer(small_quantum_config, tiny_data_config)
        patch_dim = tiny_data_config.patch_size ** 2
        num_patches = (tiny_data_config.image_size // tiny_data_config.patch_size) ** 2
        patches = torch.randn(1, num_patches, patch_dim)
        logits = model(patches)
        assert not torch.isnan(logits).any(), "NaN detected — check Fock cutoff_dim"


class TestReadoutObservable:
    """Configurable photon-number / PNR readout in place of ⟨x̂⟩."""

    def _config(self, readout: str) -> QuantumConfig:
        return QuantumConfig(
            num_modes=2,
            cutoff_dim=4,
            num_heads=2,
            cnn_channels_1=4,
            cnn_channels_2=8,
            cnn_kernel_size=3,
            decoder_hidden_dim=16,
            poly_degree=2,
            dtype="complex64",
            readout_observable=readout,
        )

    def test_default_readout_is_quadrature_x(self, small_quantum_config, tiny_data_config):
        """Omitting readout_observable / readout_observables yields x-per-mode."""
        assert small_quantum_config.readout_observable is None
        assert small_quantum_config.readout_observables is None
        plan = small_quantum_config._observable_plan
        assert len(plan) == small_quantum_config.num_modes
        assert all(entry.type == "x" for entry in plan)
        model = CVQuixer(small_quantum_config, tiny_data_config)
        first_linear = model.decoder.net[0]
        assert first_linear.in_features == (
            small_quantum_config.num_heads * small_quantum_config.num_modes
        )

    def test_photon_number_forward_shape(self, tiny_data_config):
        config = self._config("photon_number")
        model = CVQuixer(config, tiny_data_config)
        first_linear = model.decoder.net[0]
        assert first_linear.in_features == config.num_heads * config.num_modes

        patch_dim = tiny_data_config.patch_size ** 2
        num_patches = (tiny_data_config.image_size // tiny_data_config.patch_size) ** 2
        patches = torch.randn(2, num_patches, patch_dim)
        logits = model(patches)
        assert logits.shape == (2, tiny_data_config.num_classes)
        assert torch.isfinite(logits).all()

    def test_pnr_distribution_decoder_dim_and_forward(self, tiny_data_config):
        config = self._config("pnr_distribution")
        model = CVQuixer(config, tiny_data_config)
        first_linear = model.decoder.net[0]
        expected_in = config.num_heads * config.num_modes * config.cutoff_dim
        assert first_linear.in_features == expected_in

        patch_dim = tiny_data_config.patch_size ** 2
        num_patches = (tiny_data_config.image_size // tiny_data_config.patch_size) ** 2
        patches = torch.randn(2, num_patches, patch_dim)
        logits = model(patches)
        assert logits.shape == (2, tiny_data_config.num_classes)
        assert torch.isfinite(logits).all()

    def test_pnr_distribution_gradient_flows_to_hypernetwork(self, tiny_data_config):
        """A backward pass through the PNR readout must reach CNN params.

        The default PolynomialCoefficients init is [1, 0, 0…], which collapses
        the polynomial to c_0·|ψ_in⟩ = |vacuum⟩ — CNN-independent. To exercise
        the data-dependent branch we set c_1 = 0.5 on every head before forward.
        """
        config = self._config("pnr_distribution")
        model = CVQuixer(config, tiny_data_config)
        with torch.no_grad():
            for head in model.cv_attention.heads:
                head.poly_coeffs.c.data[1] = 0.5

        patch_dim = tiny_data_config.patch_size ** 2
        num_patches = (tiny_data_config.image_size // tiny_data_config.patch_size) ** 2
        patches = torch.randn(2, num_patches, patch_dim)
        logits = model(patches)
        loss = logits.sum()
        loss.backward()

        cnn_params = [
            p for n, p in model.named_parameters()
            if "hypernetwork" in n and p.requires_grad
        ]
        assert cnn_params, "No CNN hypernetwork parameters found"
        assert all(p.grad is not None for p in cnn_params), (
            "Some CNN hypernetwork params received no gradient"
        )
        nonzero = any(p.grad.abs().sum().item() > 0 for p in cnn_params)
        assert nonzero, "PNR readout did not propagate gradient to the hypernetwork"

    def test_invalid_readout_observable_rejected(self):
        with pytest.raises(ValueError, match="readout_observable"):
            QuantumConfig(readout_observable="not_a_real_observable")


class TestM1M2TruncFusion:
    """Audit M1+M2: the per-patch truncation pass is skipped when the penalty
    is disabled, and otherwise fused into the first (vacuum) LCU pass so U_i|0⟩
    is computed once instead of twice. These tests pin the equivalence of the
    fused path to the standalone _compute_patch_trunc_loss oracle, vmap-vs-loop
    agreement, and autograd-graph preservation.
    """

    def _config(self, trunc_penalty, poly_degree=2, dtype="complex128"):
        return QuantumConfig(
            num_modes=2, cutoff_dim=4, num_heads=1,
            cnn_channels_1=4, cnn_channels_2=8, cnn_kernel_size=3,
            decoder_hidden_dim=16, poly_degree=poly_degree,
            dtype=dtype, bs_topology="linear",
            trunc_penalty=trunc_penalty,
        )

    def _vacuum_flat(self, m, D, dtype):
        return FockState.vacuum(m, D, torch.device("cpu"), dtype).data.reshape(-1)

    def test_fused_trunc_matches_helper(self):
        """M2: forward's fused avg_trunc_loss == standalone helper value."""
        torch.manual_seed(0)
        cfg = self._config("norm", poly_degree=2)
        head = HyperCVAttentionHead(patch_size=7, num_patches=16, config=cfg)
        with torch.no_grad():
            head.poly_coeffs.c.data[1] = 0.5   # make `result` depend on the LCU pass
        patches = torch.randn(16, 49)

        _, _, _, fused, _, _ = head(patches)
        helper = head._compute_patch_trunc_loss(
            patches, self._vacuum_flat(2, 4, head.torch_dtype),
            torch.device("cpu"), head.torch_dtype,
        )
        assert torch.allclose(fused, helper, atol=1e-10)

    def test_trunc_none_skips_and_returns_zero(self):
        """M1: trunc_penalty='none' never runs the per-patch vacuum pass and
        returns a real 0-dim zero matching the enabled branch's dtype/device."""
        cfg = self._config("none")
        head = HyperCVAttentionHead(patch_size=7, num_patches=16, config=cfg)
        patches = torch.randn(16, 49)

        with patch.object(
            head, "_compute_patch_trunc_loss",
            wraps=head._compute_patch_trunc_loss,
        ) as spy:
            _, _, _, tl, _, _ = head(patches)
        assert spy.call_count == 0
        assert tl.dim() == 0 and not tl.is_complex()
        assert tl.item() == 0.0

        head_n = HyperCVAttentionHead(
            patch_size=7, num_patches=16, config=self._config("norm"),
        )
        _, _, _, tl_n, _, _ = head_n(patches)
        assert tl.dtype == tl_n.dtype
        assert tl.device == tl_n.device

    def test_trunc_none_vs_norm_same_output_state(self):
        """The trunc branch must not perturb readout/state/success_prob."""
        torch.manual_seed(1)
        head_none = HyperCVAttentionHead(
            patch_size=7, num_patches=16, config=self._config("none"),
        )
        head_norm = HyperCVAttentionHead(
            patch_size=7, num_patches=16, config=self._config("norm"),
        )
        head_norm.load_state_dict(head_none.state_dict())
        with torch.no_grad():
            head_none.poly_coeffs.c.data[1] = 0.5
            head_norm.poly_coeffs.c.data[1] = 0.5
        patches = torch.randn(16, 49)

        r0, s0, sp0, _, _, _ = head_none(patches)
        r1, s1, sp1, _, _, _ = head_norm(patches)
        assert torch.allclose(r0, r1, atol=1e-10)
        assert torch.allclose(s0, s1, atol=1e-10)
        assert torch.allclose(sp0, sp1, atol=1e-10)

    def test_vmap_matches_loop(self):
        """The vmapped HyperCVAttention path must equal a manual per-batch
        loop, for both trunc disabled and enabled (guards vmap signature).

        Also closes audit M7: the vmap path is the source of truth for what
        training sees, so readouts, output states, success_probs, and
        trunc_loss are all asserted element-wise against the loop reference.
        """
        for tp in ("none", "norm"):
            torch.manual_seed(2)
            attn = HyperCVAttention(
                patch_size=7, num_patches=16, config=self._config(tp),
            )
            with torch.no_grad():
                for h in attn.heads:
                    h.poly_coeffs.c.data[1] = 0.5
            patches = torch.randn(3, 16, 49)

            readouts, states, sps, tl, _, _ = attn(patches)

            B = patches.shape[0]
            man_r, man_s, man_sp, man_tl = [], [], [], []
            for head in attn.heads:
                rb, sb, spb, tlb = [], [], [], []
                for b in range(B):
                    r, s, sp, t, _, _ = head(patches[b])
                    rb.append(r)
                    sb.append(s)
                    spb.append(sp)
                    tlb.append(t)
                man_r.append(torch.stack(rb))
                man_s.append(torch.stack(sb))
                man_sp.append(torch.stack(spb))
                man_tl.append(torch.stack(tlb))
            exp_readouts = torch.cat(man_r, dim=-1)
            exp_tl = torch.stack(man_tl).mean()

            assert torch.allclose(readouts, exp_readouts, atol=1e-8), tp
            for got, exp in zip(states, man_s):
                assert torch.allclose(got, exp, atol=1e-8), tp
            for got, exp in zip(sps, man_sp):
                assert torch.allclose(got, exp, atol=1e-8), tp
            assert torch.allclose(tl, exp_tl, atol=1e-8), tp

    def test_grad_reaches_every_head(self):
        """The head-axis vmap stacks head params with plain (differentiable)
        torch.stack, so a backward pass must deposit a non-trivial gradient on
        EVERY head's leaf parameters. Guards against a regression to
        stack_module_state (which detaches → strands head gradients, leaving
        .grad None and silently breaking training).
        """
        torch.manual_seed(5)
        cfg = QuantumConfig(
            num_modes=2, cutoff_dim=4, num_heads=3,
            cnn_channels_1=4, cnn_channels_2=8, cnn_kernel_size=3,
            decoder_hidden_dim=16, poly_degree=2,
            dtype="complex64", bs_topology="linear",
            trunc_penalty="norm",
        )
        attn = HyperCVAttention(patch_size=7, num_patches=16, config=cfg)
        patches = torch.randn(2, 16, 49)

        readouts, _, _, trunc_loss, _, _ = attn(patches)
        # Depend on both the readout path and the trunc path so the gradient
        # exercises every head's hypernetwork + coefficient parameters.
        (readouts.sum() + trunc_loss).backward()

        assert len(attn.heads) == 3
        for h_idx, head in enumerate(attn.heads):
            g = head.hypernetwork.conv1.weight.grad
            assert g is not None, f"head {h_idx} conv1.weight.grad is None"
            assert torch.isfinite(g).all(), f"head {h_idx} grad not finite"
            assert g.abs().sum().item() > 0.0, f"head {h_idx} grad all-zero"

    def test_poly_degree_0_trunc_fallback(self):
        """poly_degree=0 has no LCU pass to fuse into, so forward falls back
        to the standalone _compute_patch_trunc_loss."""
        torch.manual_seed(3)
        cfg = self._config("norm", poly_degree=0)
        head = HyperCVAttentionHead(patch_size=7, num_patches=16, config=cfg)
        patches = torch.randn(16, 49)

        with patch.object(
            head, "_compute_patch_trunc_loss",
            wraps=head._compute_patch_trunc_loss,
        ) as spy:
            _, _, _, tl, _, _ = head(patches)
        assert spy.call_count == 1

        helper = head._compute_patch_trunc_loss(
            patches, self._vacuum_flat(2, 4, head.torch_dtype),
            torch.device("cpu"), head.torch_dtype,
        )
        assert torch.allclose(tl, helper, atol=1e-10)
        assert 0.0 <= tl.item() <= 1.0

    def test_trunc_grad_equivalence(self):
        """M2 autograd: gradients of the fused trunc loss w.r.t. hypernetwork
        params match those of the standalone helper (graph preserved)."""
        torch.manual_seed(4)
        cfg = self._config("norm", poly_degree=2)
        head = HyperCVAttentionHead(patch_size=7, num_patches=16, config=cfg)
        patches = torch.randn(16, 49)
        vac = self._vacuum_flat(2, 4, head.torch_dtype)
        dev, dt = torch.device("cpu"), head.torch_dtype

        _, _, fused_tl = head._apply_polynomial_iterative(
            patches, vac, dev, dt, want_trunc=True,
        )
        head.zero_grad(set_to_none=True)
        fused_tl.backward()
        g_fused = {
            n: p.grad.clone() for n, p in head.named_parameters()
            if p.grad is not None
        }

        head.zero_grad(set_to_none=True)
        helper_tl = head._compute_patch_trunc_loss(patches, vac, dev, dt)
        helper_tl.backward()
        g_helper = {
            n: p.grad.clone() for n, p in head.named_parameters()
            if p.grad is not None
        }

        assert g_fused, "no gradients flowed from the fused trunc loss"
        assert g_fused.keys() == g_helper.keys()
        for n in g_fused:
            assert torch.allclose(g_fused[n], g_helper[n], atol=1e-7, rtol=1e-5), n


class TestM3PostSelection:
    """Audit M3: post-selection renormalisation must not explode gradients
    when ‖P(M)|ψ⟩‖² collapses toward zero. Degenerate polynomial coefficients
    drive success_prob to (or near) zero; forward + backward must stay finite
    and a failed post-selection must yield an exactly-zero state.
    """

    def _config(self, poly_degree=2):
        return QuantumConfig(
            num_modes=2, cutoff_dim=4, num_heads=2,
            cnn_channels_1=4, cnn_channels_2=8, cnn_kernel_size=3,
            decoder_hidden_dim=16, poly_degree=poly_degree,
            dtype="complex64", bs_topology="linear",
            trunc_penalty="norm",
        )

    # P(M)=0 → success_prob == 0 exactly (the 0/0 worst case the divisor
    # clamp guards). P(M)=M−I → success_prob driven toward 0.
    DEGENERATE_COEFFS = [
        ("zero_poly", [0.0, 0.0, 0.0]),
        ("m_minus_i", [-1.0, 1.0, 0.0]),
    ]

    @pytest.mark.parametrize("name,coeffs", DEGENERATE_COEFFS)
    def test_low_success_prob_no_nan_head(self, name, coeffs):
        """Direct HyperCVAttentionHead forward+backward stays finite."""
        torch.manual_seed(0)
        head = HyperCVAttentionHead(
            patch_size=7, num_patches=16, config=self._config(),
        )
        with torch.no_grad():
            head.poly_coeffs.c.data.copy_(torch.tensor(coeffs))
        patches = torch.randn(16, 49)

        readout, _, success_prob, _, _, _ = head(patches)
        assert torch.isfinite(readout).all(), name
        assert torch.isfinite(success_prob).all(), name

        head.zero_grad(set_to_none=True)
        readout.sum().backward()
        grads = [p.grad for p in head.parameters() if p.grad is not None]
        assert grads, f"{name}: no gradients flowed"
        for g in grads:
            assert torch.isfinite(g).all(), name

    @pytest.mark.parametrize("name,coeffs", DEGENERATE_COEFFS)
    def test_low_success_prob_no_nan_end_to_end(
        self, name, coeffs, tiny_data_config,
    ):
        """End-to-end CVQuixer (vmapped HyperCVAttention path, where the
        torch.where runs under vmap) forward+backward stays finite."""
        torch.manual_seed(0)
        model = CVQuixer(self._config(), tiny_data_config)
        with torch.no_grad():
            for h in model.cv_attention.heads:
                h.poly_coeffs.c.data.copy_(torch.tensor(coeffs))
        patch_dim = tiny_data_config.patch_size ** 2
        num_patches = (
            tiny_data_config.image_size // tiny_data_config.patch_size
        ) ** 2
        patches = torch.randn(2, num_patches, patch_dim)

        logits = model(patches)
        assert torch.isfinite(logits).all(), name

        model.zero_grad(set_to_none=True)
        logits.sum().backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads, f"{name}: no gradients flowed"
        for g in grads:
            assert torch.isfinite(g).all(), name

    def test_failed_postselection_yields_zero_state(self):
        """P(M)=0 → success_prob 0 → renormalised state forced to exactly
        zero (the failing branch is intended behaviour, not incidental)."""
        torch.manual_seed(0)
        head = HyperCVAttentionHead(
            patch_size=7, num_patches=16, config=self._config(),
        )
        with torch.no_grad():
            head.poly_coeffs.c.data.zero_()
        patches = torch.randn(16, 49)

        readout, state_data, success_prob, _, _, _ = head(patches)
        assert success_prob.item() == 0.0
        assert torch.all(state_data == 0)
        assert torch.all(readout == 0)


class TestM5ReturnContract:
    """Audit M5: forward is name-based. No flags → plain logits tensor
    (BaseVisionTransformer contract preserved). Any return_* flag →
    CVQuixerOut whose unrequested fields are None, so adding/reordering
    optional outputs can never silently mis-bind a caller.
    """

    def _config(self, trunc_penalty="none"):
        return QuantumConfig(
            num_modes=2, cutoff_dim=4, num_heads=2,
            cnn_channels_1=4, cnn_channels_2=8, cnn_kernel_size=3,
            decoder_hidden_dim=16, poly_degree=2,
            dtype="complex64", trunc_penalty=trunc_penalty,
        )

    def _patches(self, tiny_data_config):
        patch_dim = tiny_data_config.patch_size ** 2
        num_patches = (
            tiny_data_config.image_size // tiny_data_config.patch_size
        ) ** 2
        return torch.randn(2, num_patches, patch_dim)

    def test_no_flags_returns_plain_tensor(self, tiny_data_config):
        model = CVQuixer(self._config(), tiny_data_config)
        result = model(self._patches(tiny_data_config))
        assert isinstance(result, torch.Tensor)
        assert not isinstance(result, CVQuixerOut)
        assert result.shape == (2, tiny_data_config.num_classes)

    def test_flagged_returns_namedtuple_with_correct_fields(
        self, tiny_data_config,
    ):
        model = CVQuixer(self._config("norm"), tiny_data_config)
        out = model(
            self._patches(tiny_data_config),
            return_trunc_loss=True,
            return_readouts=True,
        )
        assert isinstance(out, CVQuixerOut)
        assert isinstance(out.logits, torch.Tensor)
        assert out.logits.shape == (2, tiny_data_config.num_classes)
        assert out.trunc_loss is not None
        assert out.readouts is not None
        # Not requested → must be None (the M5 invariant).
        assert out.states is None
        assert out.success_probs is None

    def test_return_states_populates_states_and_success_probs(
        self, tiny_data_config,
    ):
        model = CVQuixer(self._config("norm"), tiny_data_config)
        out = model(self._patches(tiny_data_config), return_states=True)
        assert isinstance(out, CVQuixerOut)
        assert out.states is not None and len(out.states) == 2
        assert out.success_probs is not None and len(out.success_probs) == 2
        assert out.readouts is None
        assert out.trunc_loss is None

    def test_trunc_loss_none_when_penalty_disabled(self, tiny_data_config):
        """return_trunc_loss=True but trunc_penalty='none' → trunc_loss is
        None (not a missing positional element) — this is what lets callers
        branch on `out.trunc_loss is not None` safely."""
        model = CVQuixer(self._config("none"), tiny_data_config)
        out = model(self._patches(tiny_data_config), return_trunc_loss=True)
        assert isinstance(out, CVQuixerOut)
        assert out.trunc_loss is None

    def test_return_success_prob_populates_only_success_probs(
        self, tiny_data_config,
    ):
        """return_success_prob=True alone → CVQuixerOut with success_probs
        populated (list of (B,) per head, finite, positive), every other
        optional field None, and logits identical to the flag-free call."""
        model = CVQuixer(self._config(), tiny_data_config)
        model.eval()
        patches = self._patches(tiny_data_config)
        with torch.no_grad():
            plain = model(patches)
            out = model(patches, return_success_prob=True)
        assert isinstance(out, CVQuixerOut)
        assert out.success_probs is not None and len(out.success_probs) == 2
        for sp in out.success_probs:
            assert sp.shape == (2,)
            assert torch.isfinite(sp).all()
            assert (sp > 0).all()
        assert out.states is None and out.readouts is None
        assert out.trunc_loss is None and out.cvqnn_trunc_loss is None
        assert torch.allclose(plain, out.logits)


class TestEvaluateSuccessProbs:
    """evaluate() surfaces the per-sample raw post-selection norms."""

    def test_evaluate_returns_NH_success_probs(self, tiny_data_config):
        from torch.utils.data import DataLoader, TensorDataset

        from cv_quixer.evaluation.diagnostics import evaluate

        cfg = QuantumConfig(
            num_modes=2, cutoff_dim=4, num_heads=2,
            cnn_channels_1=4, cnn_channels_2=8, cnn_kernel_size=3,
            decoder_hidden_dim=16, poly_degree=2, dtype="complex64",
        )
        model = CVQuixer(cfg, tiny_data_config)
        num_patches = (
            tiny_data_config.image_size // tiny_data_config.patch_size
        ) ** 2
        patches = torch.randn(6, num_patches, tiny_data_config.patch_size ** 2)
        labels = torch.randint(0, tiny_data_config.num_classes, (6,))
        loader = DataLoader(TensorDataset(patches, labels), batch_size=4)

        result = evaluate(model, loader, torch.device("cpu"),
                          num_classes=tiny_data_config.num_classes)
        sp = result["success_probs"]
        assert sp.shape == (6, 2)
        assert sp.dtype == np.float32
        assert np.isfinite(sp).all() and (sp > 0).all()


class TestAutoScaling:
    """Build-and-count auto-scaling via autoscale_to_target."""

    def _data(self):
        return DataConfig(
            dataset="fashionmnist", normalize=False, patch_size=7,
            batch_size=64, num_workers=0, data_root="data/",
        )

    def _quantum(self, **over):
        from cv_quixer.config.observable_presets import resolve_observables
        base = dict(
            num_modes=2, cutoff_dim=6, num_heads=4, cnn_channels_1=8,
            cnn_channels_2=16, cnn_kernel_size=3, decoder_hidden_dim=32,
            poly_degree=3, dtype="complex64", trunc_penalty="norm",
            trunc_lambda=0.01, target_params=13760,
            readout_observables=resolve_observables("xpxsps", 6),
        )
        base.update(over)
        return QuantumConfig(**base)

    def test_cnn_channels_2_knob_matches_historical_architecture(self):
        # Pin: build-and-count resolves the same cnn_channels_2 (=14) and param
        # count (=13050) the pre-refactor closed-form formula produced for the
        # canonical full config, so existing runs stay comparable. Captured 2026-05-31.
        # scaling_knob is passed explicitly (the default is now num_heads) so this
        # keeps pinning the cnn_channels_2 resolution specifically. cvqnn_num_layers=0
        # pins the *pre-W* architecture this count was captured for (the CVQNN block
        # post-dates it); the W-on count is necessarily different.
        model = CVQuixer(
            self._quantum(scaling_knob="cnn_channels_2", cvqnn_num_layers=0),
            self._data(),
        )
        assert model.config.cnn_channels_2 == 14
        assert count_parameters(model) == 13050

    def test_default_scaling_knob_is_num_heads(self):
        # The default auto-scaling knob is num_heads (the robust knob).
        assert QuantumConfig().scaling_knob == "num_heads"

    def test_scale_by_num_heads(self):
        model = CVQuixer(self._quantum(scaling_knob="num_heads"), self._data())
        # num_heads is a coarse knob -> loose tolerance, but must land near budget.
        assert abs(count_parameters(model) - 13760) / 13760 < 0.40
        assert model.config.num_heads >= 1

    def test_unreachable_knob_raises(self):
        # cutoff_dim sets the Fock-sim dimension but does not change the trainable
        # param count under the xpxsps readout (the observable plan width is
        # cutoff-independent), so it cannot reach the budget; autoscale must raise
        # rather than loop forever.
        with pytest.raises(ValueError):
            CVQuixer(
                self._quantum(target_params=10_000_000, scaling_knob="cutoff_dim"),
                self._data(),
            )

    def test_scale_by_num_layers(self):
        # num_layers now deepens the per-patch unitary (widening the hypernetwork
        # output linear), so it is a valid — if coarse — scaling knob.
        model = CVQuixer(self._quantum(scaling_knob="num_layers"), self._data())
        assert abs(count_parameters(model) - 13760) / 13760 < 0.50
        assert model.config.num_layers >= 1


class TestSharedCVQuixer:
    """Tests for the shared-CNN + per-head-linear model (model="quantum_shared").

    The shared model swaps each head's CNNHypernetwork for a single shared
    SharedPatchCNN feeding per-head Linear projections. These verify the forward
    contract, gradient flow into both the shared CNN and the per-head linears,
    embed_dim resolution, the diagnostics accessor, and num_heads auto-scaling.
    """

    def _config(self, *, num_heads=2, target_params=-1,
                scaling_knob="num_heads", readout_observable=None):
        return QuantumConfig(
            num_modes=2,
            cutoff_dim=4,
            num_heads=num_heads,
            cnn_channels_1=4,
            cnn_channels_2=8,
            cnn_kernel_size=3,
            decoder_hidden_dim=16,
            poly_degree=2,
            dtype="complex64",
            target_params=target_params,
            scaling_knob=scaling_knob,
            readout_observable=readout_observable,
        )

    def test_forward_shape(self, tiny_data_config):
        model = SharedCVQuixer(self._config(), tiny_data_config)
        # tiny_data_config → 14×14 image / 7×7 patch = 4 patches.
        patches = torch.randn(3, 4, 49)
        logits = model(patches)
        assert logits.shape == (3, 10)
        assert torch.isfinite(logits).all()

    def test_head_linear_consumes_feature_dim(self, tiny_data_config):
        # No projection: the shared CNN emits flattened conv features of width
        # feature_dim = cnn_channels_2 * h_out². For C2=8, h_out = 7-2*2 = 3 → 72.
        # Each head's linear consumes exactly that width.
        model = SharedCVQuixer(self._config(), tiny_data_config)
        feature_dim = 8 * 3 * 3
        assert model.cv_attention.patch_cnn.out_dim == feature_dim
        assert not hasattr(model.cv_attention.patch_cnn, "proj")
        for head in model.cv_attention.heads:
            assert isinstance(head, LinearCVHead)
            assert head.linear.in_features == feature_dim

    def test_gradients_flow_to_shared_cnn_and_head_linears(self, tiny_data_config):
        # Use the PNR readout and set c_1 = 0.5 on every head: the default
        # PolynomialCoefficients init [1, 0, …] collapses P(M) to c_0·|vacuum⟩,
        # which is gate-param-independent (and ⟨x̂⟩ on the vacuum is degenerately
        # zero). This mirrors TestForwardWithObservables's gradient test for the
        # canonical model so the data-dependent branch is actually exercised.
        model = SharedCVQuixer(
            self._config(readout_observable="pnr_distribution"), tiny_data_config
        )
        with torch.no_grad():
            for head in model.cv_attention.heads:
                head.poly_coeffs.c.data[1] = 0.5

        patches = torch.randn(2, 4, 49)
        model(patches).sum().backward()

        cnn_grads = [
            p.grad for n, p in model.named_parameters()
            if "patch_cnn" in n and p.requires_grad
        ]
        assert cnn_grads, "No shared patch_cnn parameters found"
        assert all(g is not None for g in cnn_grads)
        assert any(g.abs().sum() > 0 for g in cnn_grads), (
            "PNR readout did not propagate gradient to the shared patch CNN"
        )

        lin_grads = [
            p.grad for n, p in model.named_parameters()
            if ".linear." in n and p.requires_grad
        ]
        assert lin_grads, "No per-head linear parameters found"
        assert all(g is not None for g in lin_grads)
        assert any(g.abs().sum() > 0 for g in lin_grads), (
            "PNR readout did not propagate gradient to the per-head linears"
        )

    def test_gate_params_grid_shape(self, tiny_data_config):
        cfg = self._config(num_heads=3)
        model = SharedCVQuixer(cfg, tiny_data_config)
        patches = torch.randn(2, 4, 49)
        grids = model.cv_attention.gate_params_grid(patches)
        gp = _gate_param_count(cfg.num_modes, cfg.bs_topology)
        assert len(grids) == 3                      # one per head
        for g in grids:
            assert g.shape == (2, 4, gp)            # (B, N, gate_params)

    def test_shared_cnn_runs_once_for_all_heads(self, tiny_data_config):
        # There is exactly one SharedPatchCNN regardless of head count.
        model = SharedCVQuixer(self._config(num_heads=4), tiny_data_config)
        assert isinstance(model.cv_attention, SharedCVAttention)
        assert isinstance(model.cv_attention.patch_cnn, SharedPatchCNN)
        assert len(model.cv_attention.heads) == 4

    def test_autoscale_on_num_heads(self, tiny_data_config):
        # Self-calibrate the per-head param increment, then target ~4 heads and
        # assert the auto-scaler lands within ±1 head. With no projection the
        # shared-CNN floor is small, so num_heads scales freely.
        m1 = SharedCVQuixer(self._config(num_heads=1), tiny_data_config)
        m2 = SharedCVQuixer(self._config(num_heads=2), tiny_data_config)
        p1, p2 = m1.get_num_parameters(), m2.get_num_parameters()
        assert p2 > p1                              # monotonic in num_heads
        inc = p2 - p1
        target = p1 + 3 * inc + inc // 2            # ≈ 4.5 heads worth
        scaled = SharedCVQuixer(
            self._config(num_heads=2, target_params=target), tiny_data_config,
        )
        assert scaled.config.num_heads in (4, 5)

    def test_is_base_model_with_inherited_forward(self, tiny_data_config):
        model = SharedCVQuixer(self._config(), tiny_data_config)
        assert isinstance(model, BaseVisionTransformer)
        # Optional outputs flow through the shared _CVQuixerBase.forward.
        out = model(torch.randn(2, 4, 49), return_readouts=True)
        assert isinstance(out, CVQuixerOut)
        per_head = len(model.config._observable_plan)
        assert out.readouts.shape == (2, model.config.num_heads * per_head)


class TestMultiLayer:
    """Depth-L per-patch unitary: stacked gate sequences + BS→Rot interferometers."""

    def _data(self):
        return DataConfig(
            dataset="fashionmnist", normalize=False, patch_size=7,
            batch_size=8, num_workers=0, data_root="data/",
        )

    def _quantum(self, **over):
        base = dict(
            num_modes=2, cutoff_dim=4, num_heads=2, cnn_channels_1=4,
            cnn_channels_2=8, cnn_kernel_size=3, decoder_hidden_dim=16,
            poly_degree=2, dtype="complex64", trunc_penalty="norm",
            trunc_lambda=0.01,
        )
        base.update(over)
        return QuantumConfig(**base)

    def test_op_plan_structure(self):
        # L=1 is exactly the single-layer gate sequence (backward compatible).
        assert _build_op_plan(1) == _GATE_SEQUENCE
        # L layers interleave L full sequences with L-1 interferometers.
        for L in (1, 2, 3, 4):
            plan = _build_op_plan(L)
            assert len(plan) == L * len(_GATE_SEQUENCE) + (L - 1) * len(_INTERFEROMETER_SEQUENCE)
            # First block is always a full layer (no leading interferometer).
            assert plan[: len(_GATE_SEQUENCE)] == _GATE_SEQUENCE

    def test_op_plan_param_count_grows_linearly(self):
        # Each extra layer adds one full sequence + one interferometer worth of params.
        m, topo = 2, "linear"
        p_full = _gate_param_count(m, topo)
        p_if = _op_plan_param_count(_INTERFEROMETER_SEQUENCE, m, topo)
        for L in (1, 2, 3):
            got = _op_plan_param_count(_build_op_plan(L), m, topo)
            assert got == L * p_full + (L - 1) * p_if
        # L=1 matches the legacy single-layer helper exactly.
        assert _op_plan_param_count(_build_op_plan(1), m, topo) == _gate_param_count(m, topo)

    def test_invalid_num_layers_raises(self):
        with pytest.raises(ValueError):
            _build_op_plan(0)

    def test_hypernetwork_width_scales_with_layers(self):
        # The CNN hypernetwork's final linear emits the full op-plan width.
        for L in (1, 2, 3):
            head = HyperCVAttentionHead(7, 16, self._quantum(num_layers=L))
            expected = _op_plan_param_count(_build_op_plan(L), 2, "linear")
            assert head.hypernetwork.linear.out_features == expected
            assert head.num_layers == L
            assert len(head._op_plan) == len(_build_op_plan(L))

    def test_param_count_strictly_increases_with_layers(self):
        counts = [
            count_parameters(CVQuixer(self._quantum(num_layers=L), self._data()))
            for L in (1, 2, 3)
        ]
        assert counts[0] < counts[1] < counts[2]

    def test_forward_and_grad_flow_multilayer(self):
        torch.manual_seed(0)
        model = CVQuixer(self._quantum(num_layers=2), self._data())
        patches = torch.randn(4, 16, 49)
        out = model(patches, return_trunc_loss=True)
        assert out.logits.shape == (4, 10)
        loss = torch.nn.functional.cross_entropy(
            out.logits, torch.randint(0, 10, (4,))
        ) + out.trunc_loss
        loss.backward()
        # Every hypernetwork parameter (which carries all layer + interferometer
        # gate params) must receive a gradient.
        for name, p in model.named_parameters():
            if "hypernetwork" in name:
                assert p.grad is not None and p.grad.abs().sum() > 0, name

    def test_num_layers_one_matches_default(self):
        # Explicit num_layers=1 is byte-identical to the default single-layer model.
        cfg = self._quantum()
        assert cfg.num_layers == 1
        torch.manual_seed(7); a = CVQuixer(self._quantum(num_layers=1), self._data())
        torch.manual_seed(7); b = CVQuixer(self._quantum(), self._data())
        patches = torch.randn(3, 16, 49)
        assert torch.allclose(a(patches), b(patches))

    def test_diagnostics_layout_per_block(self):
        from cv_quixer.evaluation.diagnostics import gate_param_layout

        m, n_bs = 2, _bs_pair_count(2, "linear")
        # L=1: legacy keys + offsets, total width = single-layer count.
        l1 = gate_param_layout(m, n_bs, 1)
        assert [name for name, _, _ in l1] == [
            "squeeze_r", "squeeze_phi", "bs_theta", "bs_phi", "rot_phi",
            "disp_re", "disp_im", "kerr_kappa",
        ]
        assert l1[-1][1] + l1[-1][2] == _gate_param_count(m, "linear")
        # L=2: contiguous, non-overlapping slices spanning the full op-plan width,
        # with prefixed keys for the interferometer (I0_) and second layer (L1_).
        l2 = gate_param_layout(m, n_bs, 2)
        offset = 0
        for _, start, count in l2:
            assert start == offset
            offset += count
        assert offset == _op_plan_param_count(_build_op_plan(2), m, "linear")
        names = {name for name, _, _ in l2}
        assert "I0_bs_theta" in names and "L1_squeeze_r" in names


class TestGateParamBound:
    """Soft-clip on magnitude gate params (squeeze r, displacement re/im) keeps
    the analytic Fock matrices from overflowing in complex64 — the cause of NaN
    heads at high num_heads. Default (None) is off.
    """

    def _head(self, bound):
        cfg = QuantumConfig(
            num_modes=2, cutoff_dim=6, num_heads=1,
            cnn_channels_1=4, cnn_channels_2=8, cnn_kernel_size=3,
            decoder_hidden_dim=16, poly_degree=2, dtype="complex64",
            trunc_penalty="norm", gate_param_bound=bound,
        )
        return HyperCVAttentionHead(patch_size=7, num_patches=16, config=cfg)

    def test_default_is_off(self):
        assert QuantumConfig(num_modes=2, cutoff_dim=4).gate_param_bound is None

    def test_auto_gate_bound_matches_photon_budget(self):
        from cv_quixer.config.schema import auto_gate_bound
        assert auto_gate_bound(6) == pytest.approx(1.5444, abs=1e-3)
        # squeezed-vacuum mean photon at the bound == cutoff-1 (representable budget)
        for D in (4, 6, 10):
            assert math.sinh(auto_gate_bound(D)) ** 2 == pytest.approx(D - 1, rel=1e-6)

    def test_bound_keeps_extreme_gate_params_finite(self):
        torch.manual_seed(0)
        head = self._head(bound=4.0)
        # Force the hypernetwork to emit absurd gate params (would overflow the
        # squeeze/displacement Fock matrices in complex64 without the clip).
        with torch.no_grad():
            head.hypernetwork.linear.bias.fill_(200.0)
        patches = torch.randn(16, 49)
        readout, _, success_prob, trunc, _, _ = head(patches)
        assert torch.isfinite(readout).all()
        assert torch.isfinite(trunc)
        assert torch.isfinite(success_prob)
        (readout.sum() + trunc).backward()
        grads = [p.grad for p in head.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)

    def test_clip_applied_in_gate_application(self):
        # Direct (no vmap / no post-selection): applying the per-patch unitary with
        # large gate params, bounded vs unbounded, must give DIFFERENT states (so
        # the clip is wired in, not a no-op) while the bounded one stays finite.
        head_b = self._head(bound=2.0)
        head_n = self._head(bound=None)
        D, m = 6, 2
        dev, dt = torch.device("cpu"), head_b.torch_dtype
        gp = _op_plan_param_count(_build_op_plan(head_b.num_layers), m, "linear")
        params = torch.full((gp,), 8.0)          # squeeze/disp ~8 >> bound 2
        vac = FockState.vacuum(m, D, dev, dt)
        sb = head_b._apply_patch_gates_to_state(params, vac, dev, dt)
        sn = head_n._apply_patch_gates_to_state(params, vac, dev, dt)
        assert torch.isfinite(sb.data).all()
        assert not torch.allclose(sb.data, sn.data)


class TestArchitectureDepth:
    """Configurable depth (cnn_num_conv_layers / hypernet_num_linear_layers /
    decoder_num_layers) is additive: defaults reproduce the historic architecture
    byte-for-byte (same state-dict keys, checkpoint-compatible), and deeper values
    build, forward, add parameters, and receive gradients."""

    def _patches(self, data_cfg, batch=2):
        patch_dim = data_cfg.patch_size ** 2
        num_patches = (data_cfg.image_size // data_cfg.patch_size) ** 2
        return torch.randn(batch, num_patches, patch_dim)

    def test_defaults_preserve_state_dict_keys(self, small_quantum_config, tiny_data_config):
        """At default depth, no extra_convs/hidden_linears keys exist and the
        decoder keys are exactly the historic net.0/net.2 — so existing
        checkpoints still load."""
        model = CVQuixer(small_quantum_config, tiny_data_config)
        keys = set(model.state_dict().keys())
        assert not any("extra_convs" in k for k in keys)
        assert not any("hidden_linears" in k for k in keys)
        decoder_keys = sorted(k for k in keys if k.startswith("decoder.net"))
        assert decoder_keys == [
            "decoder.net.0.bias", "decoder.net.0.weight",
            "decoder.net.2.bias", "decoder.net.2.weight",
        ]

    def test_deep_config_builds_and_adds_params(self, small_quantum_config, tiny_data_config):
        import dataclasses
        base = CVQuixer(small_quantum_config, tiny_data_config)
        deep_cfg = dataclasses.replace(
            small_quantum_config,
            cnn_num_conv_layers=4,
            hypernet_num_linear_layers=3,
            decoder_num_layers=4,
        )
        deep = CVQuixer(deep_cfg, tiny_data_config)
        keys = set(deep.state_dict().keys())
        assert any("extra_convs" in k for k in keys)
        assert any("hidden_linears" in k for k in keys)
        # decoder keeps net.0/net.2 and appends deeper layers
        assert {"decoder.net.0.weight", "decoder.net.4.weight"} <= keys
        assert count_parameters(deep) > count_parameters(base)

    def test_deep_config_forward_and_grad(self, small_quantum_config, tiny_data_config):
        import dataclasses
        import torch.nn.functional as F
        deep_cfg = dataclasses.replace(
            small_quantum_config,
            cnn_num_conv_layers=3,
            hypernet_num_linear_layers=2,
            decoder_num_layers=3,
        )
        model = CVQuixer(deep_cfg, tiny_data_config)
        patches = self._patches(tiny_data_config)
        logits = model(patches)
        assert logits.shape == (2, tiny_data_config.num_classes)
        assert not torch.isnan(logits).any()
        loss = F.cross_entropy(logits, torch.randint(0, tiny_data_config.num_classes, (2,)))
        loss.backward()
        # The new depth modules must be connected to the autograd graph.
        deep_params = [
            p for n, p in model.named_parameters()
            if "extra_convs" in n or "hidden_linears" in n
        ]
        assert deep_params
        assert all(p.grad is not None for p in deep_params)

    def test_shared_variant_deepens(self, small_quantum_config, tiny_data_config):
        import dataclasses
        deep_cfg = dataclasses.replace(
            small_quantum_config,
            cnn_num_conv_layers=3,
            hypernet_num_linear_layers=2,
        )
        model = SharedCVQuixer(deep_cfg, tiny_data_config)
        keys = set(model.state_dict().keys())
        assert any("extra_convs" in k for k in keys)
        assert any("hidden_linears" in k for k in keys)
        logits = model(self._patches(tiny_data_config))
        assert logits.shape == (2, tiny_data_config.num_classes)

    def test_depth_fields_are_valid_scaling_knobs(self):
        """Each new int field is auto-accepted as a scaling_knob by __post_init__."""
        for knob in ("cnn_num_conv_layers", "hypernet_num_linear_layers", "decoder_num_layers"):
            QuantumConfig(num_modes=2, cutoff_dim=4, scaling_knob=knob)  # must not raise

    def test_decoder_hidden_mult_sizes_decoder(self, small_quantum_config, tiny_data_config):
        """decoder_hidden_mult = c resolves decoder_hidden_dim = max(1, round(c*in_dim))."""
        import dataclasses
        from cv_quixer.models.quantum.cv_attention import _readout_total_dim
        cfg = dataclasses.replace(small_quantum_config, decoder_hidden_mult=2.0)
        model = CVQuixer(cfg, tiny_data_config)
        in_dim = cfg.num_heads * _readout_total_dim(cfg._observable_plan)
        expected = max(1, round(2.0 * in_dim))
        assert model.config.decoder_hidden_dim == expected
        assert model.decoder.net[0].out_features == expected

    def test_decoder_hidden_mult_reload_idempotent(self, small_quantum_config, tiny_data_config):
        """Rebuilding from the resolved config (asdict round-trip) is idempotent."""
        import dataclasses
        cfg = dataclasses.replace(small_quantum_config, decoder_hidden_mult=1.5)
        m1 = CVQuixer(cfg, tiny_data_config)
        m2 = CVQuixer(m1.config, tiny_data_config)
        assert m1.config.decoder_hidden_dim == m2.config.decoder_hidden_dim

    def test_decoder_hidden_mult_must_be_positive(self):
        with pytest.raises(ValueError, match="decoder_hidden_mult must be > 0"):
            QuantumConfig(num_modes=2, cutoff_dim=4, decoder_hidden_mult=0.0)

    def test_decoder_hidden_mult_default_off(self, small_quantum_config, tiny_data_config):
        """Default (None) leaves decoder_hidden_dim at its configured static value."""
        model = CVQuixer(small_quantum_config, tiny_data_config)
        assert model.config.decoder_hidden_dim == small_quantum_config.decoder_hidden_dim

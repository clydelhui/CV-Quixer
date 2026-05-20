"""Tests for the CV quantum model components.

These tests use very small circuit parameters (num_modes=2, cutoff_dim=4) to
keep simulation time tractable. The Fock backend scales as cutoff_dim^num_modes,
so large values make tests prohibitively slow.
"""

import torch
import pytest
from unittest.mock import MagicMock, patch

from cv_quixer.config.schema import QuantumConfig
from cv_quixer.models.base import BaseVisionTransformer
from cv_quixer.models.quantum.cv_attention import (
    HyperCVAttention,
    HyperCVAttentionHead,
    LCUSumCoefficients,
    PolynomialCoefficients,
    _GATE_SEQUENCE,
    _bs_pair_count,
    _gate_param_count,
    norm_truncation_penalty,
    photon_number_penalty,
)
from cv_quixer.quantum.interferometer import interferometer_param_count
from cv_quixer.models.quantum.cv_quixer import (
    CVQuixer,
    CVQuixerOut,
    _param_count_formula,
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
        readout, _, _, _ = head(patches)
        assert readout.shape == (small_quantum_config.num_modes,)

    def test_readout_is_real(self, small_quantum_config):
        head = HyperCVAttentionHead(
            patch_size=7, num_patches=16, config=small_quantum_config,
        )
        readout, _, _, _ = head(torch.randn(16, 49))
        assert not readout.is_complex()

    def test_state_data_is_tensor(self, small_quantum_config):
        head = HyperCVAttentionHead(
            patch_size=7, num_patches=16, config=small_quantum_config,
        )
        _, state_data, _, _ = head(torch.randn(16, 49))
        assert isinstance(state_data, torch.Tensor)

    def test_success_prob_positive(self, small_quantum_config):
        head = HyperCVAttentionHead(
            patch_size=7, num_patches=16, config=small_quantum_config,
        )
        _, _, success_prob, _ = head(torch.randn(16, 49))
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
            small_quantum_config._observable_plan,
        )
        assert actual == expected

    def test_formula_matches_actual_pnr_distribution(self, data_config):
        """When pnr_distribution is selected, the decoder grows by cutoff_dim
        and the formula still equals the materialised parameter count."""
        config = QuantumConfig(
            num_modes=2,
            cutoff_dim=4,
            num_heads=2,
            cnn_channels_1=4,
            cnn_channels_2=8,
            cnn_kernel_size=3,
            decoder_hidden_dim=16,
            poly_degree=2,
            dtype="complex64",
            readout_observable="pnr_distribution",
        )
        model = CVQuixer(config, data_config)
        actual = count_parameters(model)
        patch_size = data_config.patch_size
        num_patches = (data_config.image_size // patch_size) ** 2
        expected = _param_count_formula(
            patch_size, num_patches,
            config.num_heads, config.num_modes,
            config.cnn_channels_1, config.cnn_channels_2,
            config.cnn_kernel_size,
            config.decoder_hidden_dim, data_config.num_classes,
            config.bs_topology, config.poly_degree,
            config._observable_plan,
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

        _, _, _, fused = head(patches)
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
            _, _, _, tl = head(patches)
        assert spy.call_count == 0
        assert tl.dim() == 0 and not tl.is_complex()
        assert tl.item() == 0.0

        head_n = HyperCVAttentionHead(
            patch_size=7, num_patches=16, config=self._config("norm"),
        )
        _, _, _, tl_n = head_n(patches)
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

        r0, s0, sp0, _ = head_none(patches)
        r1, s1, sp1, _ = head_norm(patches)
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

            readouts, states, sps, tl = attn(patches)

            B = patches.shape[0]
            man_r, man_s, man_sp, man_tl = [], [], [], []
            for head in attn.heads:
                rb, sb, spb, tlb = [], [], [], []
                for b in range(B):
                    r, s, sp, t = head(patches[b])
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
            _, _, _, tl = head(patches)
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

        readout, _, success_prob, _ = head(patches)
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

        readout, state_data, success_prob, _ = head(patches)
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

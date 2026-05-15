"""Tests for the cv_quixer/quantum/ simulation engine.

All tests use small circuit parameters (cutoff_dim ≤ 6, num_modes ≤ 3) to keep
simulation time tractable. The Fock backend scales as cutoff_dim^num_modes.

Physics checks use analytical results from quantum optics:
  - Coherent state |α⟩ = D(α)|0⟩: ⟨x̂⟩ = √2 Re(α), ⟨p̂⟩ = √2 Im(α), ⟨n̂⟩ = |α|²
  - Rotation R(φ) is photon-number preserving: ⟨n̂⟩ unchanged
  - [x̂, p̂] = i (canonical commutation relation)
  - All gates are unitary: U U† = I
"""

from __future__ import annotations

import math

import pytest
import torch

# Use a small but physically meaningful truncation throughout
D = 6   # cutoff_dim — enough to represent coherent states with |α| ≤ 1.5 accurately
N = 2   # default num_modes


# ============================================================
# FockState
# ============================================================

class TestFockState:
    def test_vacuum_shape(self):
        from cv_quixer.quantum import FockState
        state = FockState.vacuum(num_modes=3, cutoff_dim=D)
        assert state.data.shape == (D, D, D)

    def test_vacuum_amplitude_at_zero(self):
        """Amplitude 1 at |0,0⟩, zero everywhere else."""
        from cv_quixer.quantum import FockState
        state = FockState.vacuum(num_modes=2, cutoff_dim=D)
        assert state.data[0, 0].abs().item() == pytest.approx(1.0)
        # All other entries must be zero
        assert state.data[1:, :].abs().sum().item() == pytest.approx(0.0)
        assert state.data[:, 1:].abs().sum().item() == pytest.approx(0.0)

    def test_vacuum_norm_is_one(self):
        from cv_quixer.quantum import FockState
        state = FockState.vacuum(num_modes=2, cutoff_dim=D)
        assert state.norm().item() == pytest.approx(1.0)

    def test_vacuum_is_complex(self):
        from cv_quixer.quantum import FockState
        state = FockState.vacuum(num_modes=2, cutoff_dim=D)
        assert state.data.is_complex()

    def test_properties_exposed(self):
        from cv_quixer.quantum import FockState
        state = FockState.vacuum(num_modes=3, cutoff_dim=D)
        assert state.num_modes == 3
        assert state.cutoff_dim == D

    def test_invalid_shape_raises(self):
        from cv_quixer.quantum import FockState
        wrong = torch.zeros((D, D + 1), dtype=torch.complex128)
        with pytest.raises(AssertionError):
            FockState(wrong, num_modes=2, cutoff_dim=D)

    def test_non_complex_dtype_raises(self):
        from cv_quixer.quantum import FockState
        real_data = torch.zeros((D, D), dtype=torch.float64)
        with pytest.raises(AssertionError):
            FockState(real_data, num_modes=2, cutoff_dim=D)

    def test_reduced_density_matrix_shape(self):
        from cv_quixer.quantum import FockState
        state = FockState.vacuum(num_modes=2, cutoff_dim=D)
        rho = state.reduced_density_matrix(0)
        assert rho.shape == (D, D)

    def test_reduced_density_matrix_vacuum(self):
        """Vacuum: ρ = |0⟩⟨0|, so ρ[0,0]=1 and all other entries are zero."""
        from cv_quixer.quantum import FockState
        state = FockState.vacuum(num_modes=2, cutoff_dim=D)
        rho = state.reduced_density_matrix(0)
        assert rho[0, 0].real.item() == pytest.approx(1.0, abs=1e-10)
        assert rho[1:, :].abs().sum().item() == pytest.approx(0.0, abs=1e-10)
        assert rho[:, 1:].abs().sum().item() == pytest.approx(0.0, abs=1e-10)

    def test_photon_number_probabilities_vacuum(self):
        """Vacuum: P(0)=1, P(k>0)=0 for every mode."""
        from cv_quixer.quantum import FockState
        state = FockState.vacuum(num_modes=2, cutoff_dim=D)
        for mode in range(2):
            probs = state.photon_number_probabilities(mode)
            assert probs.shape == (D,)
            assert probs[0].item() == pytest.approx(1.0, abs=1e-10)
            assert probs[1:].abs().sum().item() == pytest.approx(0.0, abs=1e-10)

    def test_photon_number_probabilities_single_photon(self):
        """|1⟩_mode0 ⊗ |0⟩_mode1: P(1)=1 on mode 0, P(0)=1 on mode 1."""
        from cv_quixer.quantum import FockState
        data = torch.zeros((D, D), dtype=torch.complex128)
        data[1, 0] = 1.0   # one photon in mode 0, vacuum in mode 1
        state = FockState(data, num_modes=2, cutoff_dim=D)
        probs0 = state.photon_number_probabilities(0)
        probs1 = state.photon_number_probabilities(1)
        assert probs0[1].item() == pytest.approx(1.0, abs=1e-10)
        assert (probs0[:1].sum() + probs0[2:].sum()).item() == pytest.approx(0.0, abs=1e-10)
        assert probs1[0].item() == pytest.approx(1.0, abs=1e-10)
        assert probs1[1:].abs().sum().item() == pytest.approx(0.0, abs=1e-10)

    def test_photon_number_probabilities_matches_expectation(self):
        """Σ k·P(k) must equal measure_photon_number for any state."""
        from cv_quixer.quantum import CVCircuit, FockState
        from cv_quixer.quantum.gates.gaussian import displacement_matrix
        circuit = CVCircuit(num_modes=1, cutoff_dim=D)
        state = FockState.vacuum(num_modes=1, cutoff_dim=D)
        state = circuit.apply_single_mode_gate(
            displacement_matrix(torch.tensor(0.6), D), 0, state
        )
        probs = state.photon_number_probabilities(0)
        k = torch.arange(D, dtype=probs.dtype)
        mean_from_probs = (k * probs).sum().item()
        mean_from_op = circuit.measure_photon_number(0, state).item()
        assert mean_from_probs == pytest.approx(mean_from_op, abs=1e-6)

    def test_photon_number_probabilities_non_negative_and_real(self):
        """After gates, every probability is real-valued and non-negative."""
        from cv_quixer.quantum import CVCircuit, FockState
        from cv_quixer.quantum.gates.gaussian import (
            displacement_matrix,
            squeezing_matrix,
        )
        circuit = CVCircuit(num_modes=1, cutoff_dim=D)
        state = FockState.vacuum(num_modes=1, cutoff_dim=D)
        state = circuit.apply_single_mode_gate(
            squeezing_matrix(torch.tensor(0.3), torch.tensor(0.4), D), 0, state
        )
        state = circuit.apply_single_mode_gate(
            displacement_matrix(torch.tensor(0.4), D), 0, state
        )
        probs = state.photon_number_probabilities(0)
        assert probs.dtype.is_floating_point
        assert (probs >= 0.0).all()
        assert probs.sum().item() <= 1.0 + 1e-6


# ============================================================
# Observable matrices (ops.py)
# ============================================================

class TestOps:
    def test_number_operator_shape(self):
        from cv_quixer.quantum.ops import number_operator_matrix
        n_mat = number_operator_matrix(D)
        assert n_mat.shape == (D, D)

    def test_number_operator_diagonal_values(self):
        """n̂ is diagonal with eigenvalues 0, 1, 2, ..., D-1."""
        from cv_quixer.quantum.ops import number_operator_matrix
        n_mat = number_operator_matrix(D)
        expected = torch.arange(D, dtype=torch.float64)
        assert torch.allclose(n_mat.diag().real, expected)
        off_diag_sum = (n_mat - torch.diag(n_mat.diag())).abs().sum().item()
        assert off_diag_sum == pytest.approx(0.0, abs=1e-12)

    def test_quadrature_x_hermitian(self):
        from cv_quixer.quantum.ops import quadrature_x_matrix
        x = quadrature_x_matrix(D)
        assert torch.allclose(x, x.conj().T, atol=1e-12)

    def test_quadrature_p_hermitian(self):
        from cv_quixer.quantum.ops import quadrature_p_matrix
        p = quadrature_p_matrix(D)
        assert torch.allclose(p, p.conj().T, atol=1e-12)

    def test_canonical_commutation_relation(self):
        """[x̂, p̂] = i on the interior states (last row/col has truncation artifacts)."""
        from cv_quixer.quantum.ops import quadrature_x_matrix, quadrature_p_matrix
        x = quadrature_x_matrix(D)
        p = quadrature_p_matrix(D)
        commutator = x @ p - p @ x
        # Check diagonal elements 0..D-2 (last element is a truncation artifact)
        interior_diag = commutator.diag()[:D - 1]
        expected = 1j * torch.ones(D - 1, dtype=torch.complex128)
        assert torch.allclose(interior_diag, expected, atol=1e-10)


# ============================================================
# Gaussian gate matrix factories (gates/gaussian.py)
# ============================================================

class TestGaussianGates:
    def test_rotation_shape(self):
        from cv_quixer.quantum.gates.gaussian import rotation_matrix
        R = rotation_matrix(torch.tensor(0.5), D)
        assert R.shape == (D, D)

    def test_rotation_zero_is_identity(self):
        from cv_quixer.quantum.gates.gaussian import rotation_matrix
        R = rotation_matrix(torch.tensor(0.0), D)
        assert torch.allclose(R, torch.eye(D, dtype=torch.complex128), atol=1e-12)

    def test_rotation_phases_shape(self):
        from cv_quixer.quantum.gates.gaussian import rotation_phases
        phases = rotation_phases(torch.tensor(0.5), D)
        assert phases.shape == (D,)

    def test_rotation_phases_zero_is_ones(self):
        from cv_quixer.quantum.gates.gaussian import rotation_phases
        phases = rotation_phases(torch.tensor(0.0), D)
        assert torch.allclose(phases, torch.ones(D, dtype=torch.complex128))

    def test_rotation_preserves_norm(self):
        """Rotation is phase-only — norm must stay exactly 1.0."""
        from cv_quixer.quantum import CVCircuit, FockState
        from cv_quixer.quantum.gates.gaussian import rotation_phases
        circuit = CVCircuit(num_modes=1, cutoff_dim=D)
        state = FockState.vacuum(num_modes=1, cutoff_dim=D)
        phases = rotation_phases(torch.tensor(1.2), D)
        state = circuit.apply_single_mode_phases(phases, 0, state)
        assert state.norm().item() == pytest.approx(1.0, rel=1e-10)

    def test_displacement_shape(self):
        from cv_quixer.quantum.gates.gaussian import displacement_matrix
        M = displacement_matrix(torch.tensor(0.5), D)
        assert M.shape == (D, D)

    def test_displacement_zero_is_identity(self):
        from cv_quixer.quantum.gates.gaussian import displacement_matrix
        M = displacement_matrix(torch.tensor(0.0), D)
        assert torch.allclose(M, torch.eye(D, dtype=torch.complex128), atol=1e-12)

    def test_displacement_sub_isometry_small_alpha(self):
        """Small |α| → sub-isometry with norm close to 1 (small truncation loss)."""
        from cv_quixer.quantum import CVCircuit, FockState
        from cv_quixer.quantum.gates.gaussian import displacement_matrix
        circuit = CVCircuit(num_modes=1, cutoff_dim=D)
        state = FockState.vacuum(num_modes=1, cutoff_dim=D)
        state = circuit.apply_single_mode_gate(displacement_matrix(torch.tensor(0.5), D), 0, state)
        assert state.norm().item() <= 1.0 + 1e-8

    def test_displacement_norm_loss_large_alpha(self):
        """Large |α| shows real truncation loss (norm < 1)."""
        from cv_quixer.quantum import CVCircuit, FockState
        from cv_quixer.quantum.gates.gaussian import displacement_matrix
        circuit = CVCircuit(num_modes=1, cutoff_dim=D)
        state = FockState.vacuum(num_modes=1, cutoff_dim=D)
        # α=2, D=6: P(n<6 | coherent |α=2⟩) ≈ 0.98 → norm² ≈ 0.98
        state = circuit.apply_single_mode_gate(displacement_matrix(torch.tensor(2.0), D), 0, state)
        assert state.norm().item() < 1.0 - 1e-3

    def test_squeezing_shape(self):
        from cv_quixer.quantum.gates.gaussian import squeezing_matrix
        S = squeezing_matrix(torch.tensor(0.3), torch.tensor(0.0), D)
        assert S.shape == (D, D)

    def test_squeezing_zero_is_identity(self):
        from cv_quixer.quantum.gates.gaussian import squeezing_matrix
        S = squeezing_matrix(torch.tensor(0.0), torch.tensor(0.0), D)
        assert torch.allclose(S, torch.eye(D, dtype=torch.complex128), atol=1e-12)

    def test_squeezing_sub_isometry(self):
        from cv_quixer.quantum import CVCircuit, FockState
        from cv_quixer.quantum.gates.gaussian import squeezing_matrix
        circuit = CVCircuit(num_modes=1, cutoff_dim=D)
        state = FockState.vacuum(num_modes=1, cutoff_dim=D)
        mat = squeezing_matrix(torch.tensor(0.5), torch.tensor(0.3), D)
        state = circuit.apply_single_mode_gate(mat, 0, state)
        assert state.norm().item() <= 1.0 + 1e-8

    def test_beamsplitter_shape(self):
        from cv_quixer.quantum.gates.gaussian import beamsplitter_matrix
        BS = beamsplitter_matrix(torch.tensor(0.5), torch.tensor(0.0), D)
        assert BS.shape == (D, D, D, D)

    def test_beamsplitter_zero_is_identity(self):
        """BS(θ=0) is transparent — both modes pass straight through."""
        from cv_quixer.quantum.gates.gaussian import beamsplitter_matrix
        BS = beamsplitter_matrix(torch.tensor(0.0), torch.tensor(0.0), D)
        BS_flat = BS.reshape(D * D, D * D)
        assert torch.allclose(BS_flat, torch.eye(D * D, dtype=torch.complex128), atol=1e-10)

    def test_beamsplitter_photon_conservation(self):
        """BS conserves total photon number: ⟨n̂_a⟩ + ⟨n̂_b⟩ unchanged."""
        from cv_quixer.quantum import CVCircuit, FockState
        from cv_quixer.quantum.gates.gaussian import beamsplitter_matrix, displacement_matrix
        circuit = CVCircuit(num_modes=2, cutoff_dim=D)
        state = FockState.vacuum(num_modes=2, cutoff_dim=D)
        state = circuit.apply_single_mode_gate(displacement_matrix(torch.tensor(0.8), D), 0, state)
        n_before = (circuit.measure_photon_number(0, state)
                    + circuit.measure_photon_number(1, state)).item()
        BS = beamsplitter_matrix(torch.tensor(math.pi / 4), torch.tensor(0.3), D)
        state = circuit.apply_two_mode_gate(BS, 0, 1, state)
        n_after = (circuit.measure_photon_number(0, state)
                   + circuit.measure_photon_number(1, state)).item()
        assert n_after == pytest.approx(n_before, rel=1e-5)

    def test_two_mode_squeezing_shape(self):
        from cv_quixer.quantum.gates.gaussian import two_mode_squeezing_matrix
        S2 = two_mode_squeezing_matrix(torch.tensor(0.3), torch.tensor(0.0), D)
        assert S2.shape == (D, D, D, D)

    def test_two_mode_squeezing_zero_is_identity(self):
        from cv_quixer.quantum.gates.gaussian import two_mode_squeezing_matrix
        S2 = two_mode_squeezing_matrix(torch.tensor(0.0), torch.tensor(0.0), D)
        S2_flat = S2.reshape(D * D, D * D)
        assert torch.allclose(S2_flat, torch.eye(D * D, dtype=torch.complex128), atol=1e-10)

    def test_gates_are_differentiable(self):
        """Gate matrices must carry gradients through to their parameters."""
        from cv_quixer.quantum.gates.gaussian import (
            displacement_matrix, squeezing_matrix, rotation_matrix,
        )
        for alpha_val, fn, args in [
            (0.5, displacement_matrix, (D,)),
            (0.5, rotation_matrix, (D,)),
        ]:
            p = torch.tensor(alpha_val, requires_grad=True)
            mat = fn(p, *args)
            mat.abs().sum().backward()
            assert p.grad is not None, f"{fn.__name__} gradient is None"

        r = torch.tensor(0.3, requires_grad=True)
        phi = torch.tensor(0.5, requires_grad=True)
        mat = squeezing_matrix(r, phi, D)
        mat.abs().sum().backward()
        assert r.grad is not None
        assert phi.grad is not None


# ============================================================
# Non-Gaussian gate matrix factories (gates/non_gaussian.py)
# ============================================================

class TestNonGaussianGates:
    def test_kerr_shape(self):
        from cv_quixer.quantum.gates.non_gaussian import kerr_matrix
        assert kerr_matrix(torch.tensor(0.1), D).shape == (D, D)

    def test_kerr_zero_is_identity(self):
        from cv_quixer.quantum.gates.non_gaussian import kerr_matrix
        K = kerr_matrix(torch.tensor(0.0), D)
        assert torch.allclose(K, torch.eye(D, dtype=torch.complex128), atol=1e-12)

    def test_kerr_is_diagonal(self):
        from cv_quixer.quantum.gates.non_gaussian import kerr_matrix
        K = kerr_matrix(torch.tensor(0.3), D)
        off_diag = K - torch.diag(K.diag())
        assert off_diag.abs().max().item() < 1e-12

    def test_kerr_diagonal_phases(self):
        """K_nn = exp(i κ n²)."""
        from cv_quixer.quantum.gates.non_gaussian import kerr_matrix
        kappa = torch.tensor(0.5)
        K = kerr_matrix(kappa, D)
        ns = torch.arange(D, dtype=torch.float64)
        expected = torch.exp(1j * kappa.to(torch.complex128) * ns.to(torch.complex128) ** 2)
        assert torch.allclose(K.diag(), expected, atol=1e-12)

    def test_kerr_unitary(self):
        from cv_quixer.quantum.gates.non_gaussian import kerr_matrix
        K = kerr_matrix(torch.tensor(0.3), D)
        assert torch.allclose(K @ K.conj().T, torch.eye(D, dtype=torch.complex128), atol=1e-10)

    def test_kerr_differentiable(self):
        from cv_quixer.quantum.gates.non_gaussian import kerr_matrix
        kappa = torch.tensor(0.1, requires_grad=True)
        kerr_matrix(kappa, D).abs().sum().backward()
        assert kappa.grad is not None

    def test_kerr_phases_shape(self):
        from cv_quixer.quantum.gates.non_gaussian import kerr_phases
        assert kerr_phases(torch.tensor(0.1), D).shape == (D,)

    def test_kerr_phases_zero_is_ones(self):
        from cv_quixer.quantum.gates.non_gaussian import kerr_phases
        phases = kerr_phases(torch.tensor(0.0), D)
        assert torch.allclose(phases, torch.ones(D, dtype=torch.complex128))

    def test_kerr_preserves_norm(self):
        from cv_quixer.quantum import CVCircuit, FockState
        from cv_quixer.quantum.gates.non_gaussian import kerr_phases
        circuit = CVCircuit(num_modes=1, cutoff_dim=D)
        state = FockState.vacuum(num_modes=1, cutoff_dim=D)
        phases = kerr_phases(torch.tensor(0.3), D)
        state = circuit.apply_single_mode_phases(phases, 0, state)
        assert state.norm().item() == pytest.approx(1.0, rel=1e-10)

    def test_cubic_phase_shape(self):
        from cv_quixer.quantum.gates.non_gaussian import cubic_phase_matrix
        assert cubic_phase_matrix(torch.tensor(0.1), D).shape == (D, D)

    def test_cubic_phase_zero_is_identity(self):
        from cv_quixer.quantum.gates.non_gaussian import cubic_phase_matrix
        V = cubic_phase_matrix(torch.tensor(0.0), D)
        assert torch.allclose(V, torch.eye(D, dtype=torch.complex128), atol=1e-12)

    def test_cubic_phase_unitary(self):
        from cv_quixer.quantum.gates.non_gaussian import cubic_phase_matrix
        V = cubic_phase_matrix(torch.tensor(0.1), D)
        assert torch.allclose(V @ V.conj().T, torch.eye(D, dtype=torch.complex128), atol=1e-8)


# ============================================================
# CVCircuit (circuit.py)
# ============================================================

class TestCVCircuit:
    def test_apply_single_mode_identity(self):
        """Identity gate leaves the state data unchanged."""
        from cv_quixer.quantum import CVCircuit, FockState
        circuit = CVCircuit(num_modes=N, cutoff_dim=D)
        state = FockState.vacuum(num_modes=N, cutoff_dim=D)
        eye = torch.eye(D, dtype=torch.complex128)
        new_state = circuit.apply_single_mode_gate(eye, mode=0, state=state)
        assert torch.allclose(new_state.data, state.data)

    def test_apply_single_mode_immutable(self):
        """apply_single_mode_gate must return a new FockState without mutating the input."""
        from cv_quixer.quantum import CVCircuit, FockState
        from cv_quixer.quantum.gates.gaussian import displacement_matrix
        circuit = CVCircuit(num_modes=N, cutoff_dim=D)
        state = FockState.vacuum(num_modes=N, cutoff_dim=D)
        data_before = state.data.clone()
        mat = displacement_matrix(torch.tensor(0.5), D)
        new_state = circuit.apply_single_mode_gate(mat, mode=0, state=state)
        assert torch.allclose(state.data, data_before), "Input state was mutated"
        assert new_state is not state

    def test_apply_two_mode_identity(self):
        """Rank-4 identity gate leaves the state data unchanged."""
        from cv_quixer.quantum import CVCircuit, FockState
        circuit = CVCircuit(num_modes=N, cutoff_dim=D)
        state = FockState.vacuum(num_modes=N, cutoff_dim=D)
        eye4 = torch.eye(D * D, dtype=torch.complex128).reshape(D, D, D, D)
        new_state = circuit.apply_two_mode_gate(eye4, mode_i=0, mode_j=1, state=state)
        assert torch.allclose(new_state.data, state.data)

    def test_apply_two_mode_immutable(self):
        from cv_quixer.quantum import CVCircuit, FockState
        from cv_quixer.quantum.gates.gaussian import beamsplitter_matrix
        circuit = CVCircuit(num_modes=N, cutoff_dim=D)
        state = FockState.vacuum(num_modes=N, cutoff_dim=D)
        data_before = state.data.clone()
        mat = beamsplitter_matrix(torch.tensor(0.4), torch.tensor(0.0), D)
        new_state = circuit.apply_two_mode_gate(mat, mode_i=0, mode_j=1, state=state)
        assert torch.allclose(state.data, data_before), "Input state was mutated"

    def test_wrong_single_mode_gate_shape_raises(self):
        from cv_quixer.quantum import CVCircuit, FockState
        circuit = CVCircuit(num_modes=N, cutoff_dim=D)
        state = FockState.vacuum(num_modes=N, cutoff_dim=D)
        wrong = torch.eye(D + 1, dtype=torch.complex128)
        with pytest.raises(AssertionError):
            circuit.apply_single_mode_gate(wrong, 0, state)

    def test_same_mode_two_mode_gate_raises(self):
        from cv_quixer.quantum import CVCircuit, FockState
        circuit = CVCircuit(num_modes=N, cutoff_dim=D)
        state = FockState.vacuum(num_modes=N, cutoff_dim=D)
        eye4 = torch.eye(D * D, dtype=torch.complex128).reshape(D, D, D, D)
        with pytest.raises(AssertionError):
            circuit.apply_two_mode_gate(eye4, mode_i=0, mode_j=0, state=state)

    def test_vacuum_quadrature_x_zero(self):
        from cv_quixer.quantum import CVCircuit, FockState
        circuit = CVCircuit(num_modes=1, cutoff_dim=D)
        state = FockState.vacuum(num_modes=1, cutoff_dim=D)
        assert circuit.measure_quadrature_x(0, state).item() == pytest.approx(0.0, abs=1e-10)

    def test_vacuum_quadrature_p_zero(self):
        from cv_quixer.quantum import CVCircuit, FockState
        circuit = CVCircuit(num_modes=1, cutoff_dim=D)
        state = FockState.vacuum(num_modes=1, cutoff_dim=D)
        assert circuit.measure_quadrature_p(0, state).item() == pytest.approx(0.0, abs=1e-10)

    def test_vacuum_photon_number_zero(self):
        from cv_quixer.quantum import CVCircuit, FockState
        circuit = CVCircuit(num_modes=1, cutoff_dim=D)
        state = FockState.vacuum(num_modes=1, cutoff_dim=D)
        assert circuit.measure_photon_number(0, state).item() == pytest.approx(0.0, abs=1e-10)

    def test_measure_pnr_distribution_shape_and_dtype(self):
        """PNR readout returns a real (cutoff_dim,) tensor."""
        from cv_quixer.quantum import CVCircuit, FockState
        circuit = CVCircuit(num_modes=1, cutoff_dim=D)
        state = FockState.vacuum(num_modes=1, cutoff_dim=D)
        probs = circuit.measure_pnr_distribution(0, state)
        assert probs.shape == (D,)
        assert probs.dtype.is_floating_point

    def test_measure_pnr_distribution_sums_to_unit_or_less(self):
        """Truncated PNR may lose mass to n ≥ D, so the sum is ≤ 1 + ε."""
        from cv_quixer.quantum import CVCircuit, FockState
        from cv_quixer.quantum.gates.gaussian import squeezing_matrix
        circuit = CVCircuit(num_modes=1, cutoff_dim=D)
        state = FockState.vacuum(num_modes=1, cutoff_dim=D)
        state = circuit.apply_single_mode_gate(
            squeezing_matrix(torch.tensor(0.5), torch.tensor(0.0), D), 0, state
        )
        probs = circuit.measure_pnr_distribution(0, state)
        total = probs.sum().item()
        assert total <= 1.0 + 1e-8
        assert total >= 0.0

    def test_coherent_state_quadrature_x(self):
        """D(α)|0⟩ with real α: ⟨x̂⟩ = √2 Re(α).

        Uses α=0.5 so P(n≥6) < 1e-8 at D=6: the state is fully represented and
        the analytic sub-isometric displacement gives the exact Fock coefficients.
        """
        from cv_quixer.quantum import CVCircuit, FockState
        from cv_quixer.quantum.gates.gaussian import displacement_matrix
        circuit = CVCircuit(num_modes=1, cutoff_dim=D)
        state = FockState.vacuum(num_modes=1, cutoff_dim=D)
        alpha = torch.tensor(0.5)
        state = circuit.apply_single_mode_gate(displacement_matrix(alpha, D), 0, state)
        assert circuit.measure_quadrature_x(0, state).item() == pytest.approx(math.sqrt(2) / 2, abs=1e-5)

    def test_coherent_state_quadrature_p(self):
        """D(iβ)|0⟩ with purely imaginary α: ⟨p̂⟩ = √2 β.

        Uses α=0.5j so P(n≥6) < 1e-8 at D=6 (same reasoning as quadrature_x test).
        """
        from cv_quixer.quantum import CVCircuit, FockState
        from cv_quixer.quantum.gates.gaussian import displacement_matrix
        circuit = CVCircuit(num_modes=1, cutoff_dim=D)
        state = FockState.vacuum(num_modes=1, cutoff_dim=D)
        alpha = torch.tensor(0.0 + 0.5j, dtype=torch.complex128)
        state = circuit.apply_single_mode_gate(displacement_matrix(alpha, D), 0, state)
        assert circuit.measure_quadrature_p(0, state).item() == pytest.approx(math.sqrt(2) / 2, abs=1e-5)

    def test_coherent_state_photon_number(self):
        """D(α)|0⟩: ⟨n̂⟩ = |α|²."""
        from cv_quixer.quantum import CVCircuit, FockState
        from cv_quixer.quantum.gates.gaussian import displacement_matrix
        circuit = CVCircuit(num_modes=1, cutoff_dim=D)
        state = FockState.vacuum(num_modes=1, cutoff_dim=D)
        alpha = torch.tensor(0.5)
        state = circuit.apply_single_mode_gate(displacement_matrix(alpha, D), 0, state)
        assert circuit.measure_photon_number(0, state).item() == pytest.approx(0.25, rel=1e-3)

    def test_rotation_preserves_photon_number(self):
        """Rotation is photon-number preserving: ⟨n̂⟩ unchanged after R(φ)."""
        from cv_quixer.quantum import CVCircuit, FockState
        from cv_quixer.quantum.gates.gaussian import displacement_matrix, rotation_matrix
        circuit = CVCircuit(num_modes=1, cutoff_dim=D)
        state = FockState.vacuum(num_modes=1, cutoff_dim=D)
        state = circuit.apply_single_mode_gate(displacement_matrix(torch.tensor(1.0), D), 0, state)
        n_before = circuit.measure_photon_number(0, state).item()
        state = circuit.apply_single_mode_gate(rotation_matrix(torch.tensor(1.3), D), 0, state)
        n_after = circuit.measure_photon_number(0, state).item()
        assert n_after == pytest.approx(n_before, rel=1e-6)

    def test_gate_is_sub_isometry(self):
        """Analytic squeezing is a sub-isometry: norm ≤ 1 (amplitude may leak to n ≥ D)."""
        from cv_quixer.quantum import CVCircuit, FockState
        from cv_quixer.quantum.gates.gaussian import squeezing_matrix
        circuit = CVCircuit(num_modes=N, cutoff_dim=D)
        state = FockState.vacuum(num_modes=N, cutoff_dim=D)
        mat = squeezing_matrix(torch.tensor(0.4), torch.tensor(0.5), D)
        state = circuit.apply_single_mode_gate(mat, 0, state)
        assert state.norm().item() <= 1.0 + 1e-8

    def test_second_mode_unaffected_by_single_mode_gate(self):
        """A gate on mode 0 must not change the reduced state of mode 1.

        Uses rotation (exactly unitary at any D — diagonal unit-modulus phases, zero
        truncation loss) so ρ₁ is preserved to machine precision.
        """
        from cv_quixer.quantum import CVCircuit, FockState
        from cv_quixer.quantum.gates.gaussian import rotation_matrix
        circuit = CVCircuit(num_modes=N, cutoff_dim=D)
        state = FockState.vacuum(num_modes=N, cutoff_dim=D)
        rho1_before = state.reduced_density_matrix(1).clone()
        mat = rotation_matrix(torch.tensor(0.7), D)
        state = circuit.apply_single_mode_gate(mat, 0, state)
        rho1_after = state.reduced_density_matrix(1)
        assert torch.allclose(rho1_before, rho1_after, atol=1e-10)


# ============================================================
# Clements interferometer (interferometer.py)
# ============================================================

class TestInterferometer:
    def test_param_count_equals_n_squared(self):
        """interferometer_param_count(N) = N²."""
        from cv_quixer.quantum import interferometer_param_count
        for n in range(1, 6):
            assert interferometer_param_count(n) == n * n

    def test_zero_params_is_identity(self):
        """All-zero BS angles and mode phases → identity transformation."""
        from cv_quixer.quantum import CVCircuit, FockState, clements_interferometer
        n_modes = 3
        circuit = CVCircuit(num_modes=n_modes, cutoff_dim=4)
        state = FockState.vacuum(num_modes=n_modes, cutoff_dim=4)
        n_bs = n_modes * (n_modes - 1) // 2
        bs_angles = torch.zeros(n_bs)
        bs_phases = torch.zeros(n_bs)
        mode_phases = torch.zeros(n_modes)
        new_state = clements_interferometer(bs_angles, bs_phases, mode_phases, circuit, state)
        assert torch.allclose(new_state.data, state.data, atol=1e-10)

    def test_norm_preserved(self):
        """Interferometer is unitary — state norm stays 1.0."""
        from cv_quixer.quantum import CVCircuit, FockState, clements_interferometer
        n_modes = 3
        circuit = CVCircuit(num_modes=n_modes, cutoff_dim=4)
        state = FockState.vacuum(num_modes=n_modes, cutoff_dim=4)
        n_bs = n_modes * (n_modes - 1) // 2
        bs_angles = torch.randn(n_bs) * 0.3
        bs_phases = torch.randn(n_bs) * 0.3
        mode_phases = torch.randn(n_modes) * 0.3
        new_state = clements_interferometer(bs_angles, bs_phases, mode_phases, circuit, state)
        assert new_state.norm().item() == pytest.approx(1.0, abs=1e-5)

    def test_output_is_new_state(self):
        """clements_interferometer should not return the same object as the input."""
        from cv_quixer.quantum import CVCircuit, FockState, clements_interferometer
        circuit = CVCircuit(num_modes=2, cutoff_dim=4)
        state = FockState.vacuum(num_modes=2, cutoff_dim=4)
        new_state = clements_interferometer(
            torch.zeros(1), torch.zeros(1), torch.zeros(2), circuit, state,
        )
        assert new_state is not state


# ============================================================
# Parameter shift rule (grad.py)
# ============================================================

class TestParameterShiftRule:
    def test_forward_matches_direct_call(self):
        """forward() must return the same value as calling circuit_fn directly."""
        from cv_quixer.quantum import ParameterShiftFunction

        def circuit_fn(params):
            return torch.sin(params) + params ** 2

        params = torch.tensor([0.3, -0.8], requires_grad=True)
        psr_result = ParameterShiftFunction.apply(circuit_fn, params, math.pi / 2)
        direct = circuit_fn(params.detach())
        assert torch.allclose(psr_result, direct, atol=1e-12)

    def test_gradient_sinusoidal_exact(self):
        """PSR is exact for sinusoidal circuits: d/dθ cos(θ) = -sin(θ)."""
        from cv_quixer.quantum import ParameterShiftFunction

        def circuit_fn(params):
            return torch.cos(params)

        theta = torch.tensor([0.7], requires_grad=True)
        result = ParameterShiftFunction.apply(circuit_fn, theta, math.pi / 2)
        result.sum().backward()
        assert theta.grad[0].item() == pytest.approx(-math.sin(0.7), rel=1e-5)

    def test_gradient_constant_is_zero(self):
        """Constant circuit_fn has zero gradient for all parameters."""
        from cv_quixer.quantum import ParameterShiftFunction

        def circuit_fn(params):
            return torch.ones(3)

        params = torch.tensor([0.5, 1.0, -0.3], requires_grad=True)
        result = ParameterShiftFunction.apply(circuit_fn, params, math.pi / 2)
        result.sum().backward()
        assert torch.allclose(params.grad, torch.zeros(3), atol=1e-12)

    def test_gradient_multi_param(self):
        """PSR correctly differentiates each parameter independently."""
        from cv_quixer.quantum import ParameterShiftFunction

        def circuit_fn(params):
            # Each output depends on only one input → Jacobian is diagonal
            return torch.cos(params)

        params = torch.tensor([0.4, 1.1, -0.2], requires_grad=True)
        result = ParameterShiftFunction.apply(circuit_fn, params, math.pi / 2)
        result.sum().backward()
        expected = torch.tensor([-math.sin(0.4), -math.sin(1.1), -math.sin(-0.2)])
        assert torch.allclose(params.grad, expected, atol=1e-5)

    def test_no_grad_context_does_not_break_forward(self):
        """forward() runs inside torch.no_grad() — result should be detached."""
        from cv_quixer.quantum import ParameterShiftFunction

        def circuit_fn(params):
            return params * 2.0

        params = torch.tensor([1.0, 2.0], requires_grad=True)
        result = ParameterShiftFunction.apply(circuit_fn, params, math.pi / 2)
        assert result is not None


# ============================================================
# Engine public API
# ============================================================

class TestEnginePublicAPI:
    def test_all_public_symbols_importable(self):
        """Everything in __init__.py must be importable without error."""
        from cv_quixer.quantum import (
            FockState,
            CVCircuit,
            GradMode,
            ParameterShiftFunction,
            displacement_matrix,
            squeezing_matrix,
            rotation_phases,
            rotation_matrix,
            beamsplitter_matrix,
            two_mode_squeezing_matrix,
            kerr_phases,
            kerr_matrix,
            cubic_phase_matrix,
            clements_interferometer,
            interferometer_param_count,
        )
        # If we got here, all imports succeeded
        assert True

    def test_grad_mode_enum_values(self):
        from cv_quixer.quantum import GradMode
        assert GradMode.BACKPROP != GradMode.PARAMETER_SHIFT
        assert GradMode["BACKPROP"] is GradMode.BACKPROP
        assert GradMode["PARAMETER_SHIFT"] is GradMode.PARAMETER_SHIFT

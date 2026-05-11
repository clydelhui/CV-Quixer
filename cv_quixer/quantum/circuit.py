"""CVCircuit — the core einsum-based gate application engine.

CVCircuit is a stateless executor. It holds no nn.Parameters and has no
knowledge of the ViT model. It takes FockState objects and GateMatrix tensors
and contracts them together using torch.einsum.

Immutable state threading: every apply_* method returns a NEW FockState
rather than mutating the input. This is required for torch.autograd to
correctly walk the computation graph backwards through the einsum chain.

Backprop note: gate matrices are differentiable functions of their parameters
(computed in gates/gaussian.py and gates/non_gaussian.py), so gradients
flow automatically through the einsum contractions into the gate parameters.
No special treatment is needed for GradMode.BACKPROP.
"""

from __future__ import annotations

import torch

from cv_quixer.quantum.state import FockState
from cv_quixer.quantum.types import GateMatrix


class CVCircuit:
    """Stateless CV quantum circuit executor.

    Args:
        num_modes:  Number of bosonic modes.
        cutoff_dim: Fock space truncation dimension.
    """

    def __init__(self, num_modes: int, cutoff_dim: int) -> None:
        self.num_modes = num_modes
        self.cutoff_dim = cutoff_dim

    # ------------------------------------------------------------------
    # Gate application
    # ------------------------------------------------------------------

    def apply_single_mode_gate(
        self,
        gate: GateMatrix,   # shape (D, D)
        mode: int,
        state: FockState,
    ) -> FockState:
        """Contract a (D, D) gate matrix into the state along one mode axis.

        The operation is:
            ψ'_{i₀,...,i_{mode-1}, j, i_{mode+1},...} =
                Σ_k  gate[j, k] · ψ_{i₀,...,k,...}

        Implemented via torch.einsum with dynamic subscript strings so the
        mode index can be any integer in [0, num_modes).

        Args:
            gate:  Complex (D, D) unitary matrix.
            mode:  Which mode axis to contract along (0-indexed).
            state: Input FockState. Not mutated.

        Returns:
            New FockState after applying the gate.
        """
        assert gate.shape == (self.cutoff_dim, self.cutoff_dim), (
            f"Single-mode gate must be ({self.cutoff_dim}, {self.cutoff_dim}), "
            f"got {gate.shape}"
        )

        N = self.num_modes
        # Build einsum subscripts dynamically.
        # Input state indices: a b c ...  (N letters)
        # Gate contracts on the target mode index: output[...j...] = sum_k gate[j,k] * state[...k...]
        state_idx = [chr(ord('a') + i) for i in range(N)]          # ['a','b','c',...]
        out_idx = state_idx.copy()
        contracted = state_idx[mode]          # the index we sum over
        out_char = chr(ord('a') + N)          # a fresh letter for the output
        out_idx[mode] = out_char

        subscript = (
            f"{out_char}{contracted},"        # gate: out,in
            f"{''.join(state_idx)}"           # state: a,b,...
            f"->{''.join(out_idx)}"           # result: a,...,out,...
        )

        new_data = torch.einsum(subscript, gate, state.data)
        return FockState(new_data, N, self.cutoff_dim)

    def apply_two_mode_gate(
        self,
        gate: GateMatrix,   # shape (D, D, D, D)
        mode_i: int,
        mode_j: int,
        state: FockState,
    ) -> FockState:
        """Contract a rank-4 gate tensor into the state along two mode axes.

        The gate is a (D, D, D, D) tensor where:
            gate[out_i, out_j, in_i, in_j]

        The operation is:
            ψ'_{..., out_i, ..., out_j, ...} =
                Σ_{in_i, in_j} gate[out_i, out_j, in_i, in_j] · ψ_{..., in_i, ..., in_j, ...}

        Args:
            gate:   Complex (D, D, D, D) two-mode unitary tensor.
            mode_i: First mode index.
            mode_j: Second mode index.
            state:  Input FockState. Not mutated.

        Returns:
            New FockState after applying the gate.
        """
        assert gate.shape == (self.cutoff_dim,) * 4, (
            f"Two-mode gate must be shape ({self.cutoff_dim},)*4, got {gate.shape}"
        )
        assert mode_i != mode_j, "mode_i and mode_j must be different"

        N = self.num_modes
        state_idx = [chr(ord('a') + i) for i in range(N)]
        out_idx = state_idx.copy()

        in_i = state_idx[mode_i]
        in_j = state_idx[mode_j]
        out_i = chr(ord('a') + N)
        out_j = chr(ord('a') + N + 1)
        out_idx[mode_i] = out_i
        out_idx[mode_j] = out_j

        # gate subscript: out_i, out_j, in_i, in_j
        subscript = (
            f"{out_i}{out_j}{in_i}{in_j},"
            f"{''.join(state_idx)}"
            f"->{''.join(out_idx)}"
        )

        new_data = torch.einsum(subscript, gate, state.data)
        return FockState(new_data, N, self.cutoff_dim)

    # ------------------------------------------------------------------
    # Measurements (expectation values)
    # ------------------------------------------------------------------

    def measure_quadrature_x(self, mode: int, state: FockState) -> torch.Tensor:
        """Compute 〈x̂〉 = 〈ψ|x̂|ψ〉 for the given mode.

        Args:
            mode:  Mode index.
            state: Current FockState.

        Returns:
            Real-valued scalar tensor (imaginary part is numerically zero
            for a pure state and a Hermitian observable).
        """
        from cv_quixer.quantum.ops import quadrature_x_matrix
        x_mat = quadrature_x_matrix(self.cutoff_dim, dtype=state.dtype).to(state.device)
        rho = state.reduced_density_matrix(mode)
        return torch.trace(rho @ x_mat).real

    def measure_quadrature_p(self, mode: int, state: FockState) -> torch.Tensor:
        """Compute 〈p̂〉 = 〈ψ|p̂|ψ〉 for the given mode."""
        from cv_quixer.quantum.ops import quadrature_p_matrix
        p_mat = quadrature_p_matrix(self.cutoff_dim, dtype=state.dtype).to(state.device)
        rho = state.reduced_density_matrix(mode)
        return torch.trace(rho @ p_mat).real

    def measure_photon_number(self, mode: int, state: FockState) -> torch.Tensor:
        """Compute 〈n̂〉 = 〈ψ|n̂|ψ〉 for the given mode."""
        from cv_quixer.quantum.ops import number_operator_matrix
        n_mat = number_operator_matrix(self.cutoff_dim, dtype=state.dtype).to(state.device)
        rho = state.reduced_density_matrix(mode)
        return torch.trace(rho @ n_mat).real

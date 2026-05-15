"""FockState — the N-mode Fock-basis statevector container.

The state of N bosonic modes with Fock space cutoff D is represented as a
complex tensor of shape (D, D, ..., D) = (D,)*N.

Memory scales as D^N, so keep num_modes ≤ 8 and cutoff_dim ≤ 10 for
tractable simulation.

Reference: Serafini, "Quantum Continuous Variables" (2017), Ch. 2.
"""

from __future__ import annotations

import torch

from cv_quixer.quantum.types import FockTensor


class FockState:
    """Typed container for an N-mode Fock-basis statevector.

    This is a value object — CVCircuit acts on it, it does not act on itself.
    All circuit operations return a new FockState rather than mutating in
    place, preserving the torch.autograd computation graph.

    Args:
        data:       Complex tensor of shape (cutoff_dim,) * num_modes.
        num_modes:  Number of bosonic modes.
        cutoff_dim: Fock space truncation dimension.
    """

    def __init__(self, data: FockTensor, num_modes: int, cutoff_dim: int) -> None:
        assert data.shape == (cutoff_dim,) * num_modes, (
            f"Expected shape {(cutoff_dim,) * num_modes}, got {data.shape}"
        )
        assert data.is_complex(), "FockState tensor must be complex dtype"
        self._data = data
        self._num_modes = num_modes
        self._cutoff_dim = cutoff_dim

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def vacuum(
        cls,
        num_modes: int,
        cutoff_dim: int,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.complex128,
    ) -> "FockState":
        """Return the N-mode vacuum state |0,0,...,0〉.

        The vacuum is the tensor with a 1 at index (0, 0, ..., 0) and 0
        everywhere else.

        Args:
            num_modes:  Number of bosonic modes N.
            cutoff_dim: Fock space truncation D.
            device:     Torch device.
            dtype:      Complex dtype (complex64 or complex128).

        Returns:
            FockState representing |0〉^⊗N.
        """
        total = cutoff_dim ** num_modes
        flat = torch.zeros(total, dtype=dtype, device=device)
        flat = flat.index_put(
            (torch.zeros(1, dtype=torch.long, device=device),),
            torch.ones(1, dtype=dtype, device=device),
        )
        data = flat.reshape((cutoff_dim,) * num_modes)
        return cls(data, num_modes, cutoff_dim)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def data(self) -> FockTensor:
        """The raw complex tensor. Shape: (cutoff_dim,) * num_modes."""
        return self._data

    @property
    def num_modes(self) -> int:
        return self._num_modes

    @property
    def cutoff_dim(self) -> int:
        return self._cutoff_dim

    @property
    def device(self) -> torch.device:
        return self._data.device

    @property
    def dtype(self) -> torch.dtype:
        return self._data.dtype

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def norm(self) -> torch.Tensor:
        """Fock-basis norm ||ψ||².

        Should be ≈ 1.0 after every gate application. Significant deviation
        from 1.0 indicates that the Fock truncation cutoff_dim is too small
        for the current gate parameters (state is leaking out of the truncated
        space).
        """
        return (self._data.abs() ** 2).sum()

    def reduced_density_matrix(self, mode: int) -> torch.Tensor:
        """Compute the single-mode reduced density matrix by tracing out all other modes.

        Args:
            mode: Index of the mode to keep (0-indexed).

        Returns:
            Complex tensor of shape (cutoff_dim, cutoff_dim) representing
            ρ_mode = Tr_{other}(|ψ〉〈ψ|).
        """
        assert 0 <= mode < self._num_modes, f"mode {mode} out of range [0, {self._num_modes})"
        # Reshape state to (D, D^(N-1)) by moving target mode to front
        psi = self._data
        psi = psi.movedim(mode, 0)                      # (D, D, ..., D) with mode first
        psi = psi.reshape(self._cutoff_dim, -1)          # (D, D^(N-1))
        return torch.einsum("ia,ja->ij", psi, psi.conj())  # (D, D)

    def photon_number_probabilities(self, mode: int) -> torch.Tensor:
        """Photon-number probability distribution P(n) for a single mode.

        Returns the diagonal of the reduced density matrix ρ_mode, which is
        the probability of measuring n photons in `mode` for n = 0, …, D-1.
        Truncation may cause the sum to be slightly less than 1; values are
        clamped to be non-negative for numerical safety.

        Args:
            mode: Index of the mode to measure (0-indexed).

        Returns:
            Real tensor of shape (cutoff_dim,) with non-negative entries.
        """
        rho = self.reduced_density_matrix(mode)
        return rho.diagonal().real.clamp(min=0.0)

    def __repr__(self) -> str:
        return (
            f"FockState(num_modes={self._num_modes}, "
            f"cutoff_dim={self._cutoff_dim}, "
            f"norm={self.norm().item():.6f})"
        )

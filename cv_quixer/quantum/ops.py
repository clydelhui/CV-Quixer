"""Fock-basis matrix representations of quantum observables.

These matrices are used by CVCircuit.measure_* to compute expectation values
via Tr(ρ O) = 〈ψ|O|ψ〉.

All returned matrices are of shape (cutoff_dim, cutoff_dim) and are NOT
differentiable w.r.t. any circuit parameter — they are fixed observables,
not trainable gates.
"""

import torch


def number_operator_matrix(cutoff_dim: int, dtype: torch.dtype = torch.complex128) -> torch.Tensor:
    """Matrix of the photon number operator n̂ = a†a in the Fock basis.

    n̂|n〉 = n|n〉, so this is a diagonal matrix diag(0, 1, 2, ..., D-1).

    Args:
        cutoff_dim: Fock space truncation D.
        dtype:      Complex dtype for consistency with gate matrices.

    Returns:
        Diagonal tensor of shape (D, D).
    """
    real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
    ns = torch.arange(cutoff_dim, dtype=real_dtype)
    return torch.diag(ns).to(dtype)


def quadrature_x_matrix(cutoff_dim: int, dtype: torch.dtype = torch.complex128) -> torch.Tensor:
    """Matrix of the position quadrature x̂ = (a + a†) / √2 in the Fock basis.

    Matrix elements: 〈m|x̂|n〉 = (√n δ_{m,n-1} + √(n+1) δ_{m,n+1}) / √2.

    Args:
        cutoff_dim: Fock space truncation D.
        dtype:      Complex dtype.

    Returns:
        Hermitian tensor of shape (D, D).
    """
    real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
    ns = torch.arange(cutoff_dim, dtype=real_dtype)
    # Creation operator a†: 〈n+1|a†|n〉 = √(n+1)
    a_dag = torch.diag(torch.sqrt(ns[1:]), diagonal=-1).to(dtype)
    a = a_dag.T.conj()
    return (a + a_dag) / (2.0 ** 0.5)


def quadrature_p_matrix(cutoff_dim: int, dtype: torch.dtype = torch.complex128) -> torch.Tensor:
    """Matrix of the momentum quadrature p̂ = i(a† - a) / √2 in the Fock basis.

    Matrix elements: 〈m|p̂|n〉 = i(√n δ_{m,n-1} - √(n+1) δ_{m,n+1}) / √2.

    Args:
        cutoff_dim: Fock space truncation D.
        dtype:      Complex dtype.

    Returns:
        Hermitian tensor of shape (D, D).
    """
    real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
    ns = torch.arange(cutoff_dim, dtype=real_dtype)
    a_dag = torch.diag(torch.sqrt(ns[1:]), diagonal=-1).to(dtype)
    a = a_dag.T.conj()
    return 1j * (a_dag - a) / (2.0 ** 0.5)

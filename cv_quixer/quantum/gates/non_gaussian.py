"""Non-Gaussian gate matrix factories.

Non-Gaussian gates break the Gaussian character of the state and introduce
the nonlinearity that makes CV quantum circuits computationally universal
(beyond linear optics).

These gates require the Fock basis representation — they cannot be
efficiently simulated using the covariance matrix formalism.

All functions return complex PyTorch tensors differentiable w.r.t. their
parameters via torch.autograd.

Reference:
    Killoran et al., "Continuous-variable quantum neural networks" (2019)
    https://arxiv.org/abs/1806.06871
"""

from __future__ import annotations

import torch


def kerr_matrix(kappa: torch.Tensor, cutoff_dim: int) -> torch.Tensor:
    """Kerr non-linearity K(κ) = exp(i κ n̂²) in the Fock basis.

    The Kerr gate is diagonal in the Fock basis:
        K_{nn} = exp(i κ n²)

    It models the χ⁽³⁾ optical non-linearity (self-phase modulation)
    and is the standard source of non-Gaussian character in CV QNN architectures.

    Args:
        kappa:      Kerr coupling strength (real scalar tensor).
        cutoff_dim: Fock space truncation D.

    Returns:
        Complex diagonal tensor of shape (D, D).
        Differentiable w.r.t. kappa.

    Note:
        The diagonal structure means this gate can be implemented more
        efficiently as an elementwise multiply on the state vector rather
        than a full matrix multiply. CVCircuit.apply_single_mode_gate handles
        the general case; an optimised path can be added later.
    """
    dtype = torch.complex128
    device = kappa.device

    ns = torch.arange(cutoff_dim, dtype=torch.float64, device=device)
    phases = torch.exp(1j * kappa.to(dtype) * ns.to(dtype) ** 2)
    return torch.diag_embed(phases)


def kerr_phases(kappa: torch.Tensor, cutoff_dim: int) -> torch.Tensor:
    """Returns (D,) phase vector for K(κ): phases[n] = exp(i·κ·n²).

    Kerr gate is diagonal in Fock basis: K_{nn} = exp(iκn²).
    Kerr is exactly norm-preserving at any cutoff_dim.

    Args:
        kappa:      Kerr coupling strength (real scalar tensor).
        cutoff_dim: Fock space truncation D.

    Returns:
        Complex tensor of shape (D,). Differentiable w.r.t. kappa.
    """
    ns = torch.arange(cutoff_dim, dtype=torch.float64, device=kappa.device)
    return torch.exp(1j * kappa.to(torch.complex128) * ns.to(torch.complex128) ** 2)


def cubic_phase_matrix(gamma: torch.Tensor, cutoff_dim: int) -> torch.Tensor:
    """Cubic phase gate V(γ) = exp(i γ x̂³ / 6) in the Fock basis.

    The cubic phase gate is a key resource for universal CV quantum computation.
    Unlike the Kerr gate, it is NOT diagonal in the Fock basis and requires
    computing the position operator x̂ = (a + a†)/√2 in the truncated space.

    Matrix elements:
        V_{mn} = 〈m| exp(i γ x̂³ / 6) |n〉

    Computed via matrix exponentiation of (i γ X³ / 6) where X is the D×D
    position quadrature matrix.

    Args:
        gamma:      Cubic phase gate parameter (real scalar tensor).
        cutoff_dim: Fock space truncation D.

    Returns:
        Complex tensor of shape (D, D).
        Differentiable w.r.t. gamma (via torch.linalg.matrix_exp).

    Warning:
        Matrix exponentiation is expensive (O(D³)). For large cutoff_dim
        (D > 20) consider using a Padé approximant or Trotter decomposition.
        Numerical accuracy also degrades for large |gamma|.

    Note:
        CV-Quixer uses exp(i γ x̂³ / 6); SF's Vgate(γ) uses exp(i γ x̂³ / 3).
        Equivalence: cubic_phase_matrix(γ) ≡ SF.Vgate(γ / 2).
    """
    from cv_quixer.quantum.ops import quadrature_x_matrix

    dtype = torch.complex128
    device = gamma.device

    X = quadrature_x_matrix(cutoff_dim, dtype=dtype).to(device)
    X3 = X @ X @ X
    generator = 1j * gamma.to(dtype) / 6.0 * X3
    return torch.linalg.matrix_exp(generator)

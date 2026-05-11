"""Gaussian gate matrix factories.

Each function returns a gate matrix in the Fock basis as a complex PyTorch
tensor. All functions are pure (stateless) and differentiable w.r.t. their
parameters via torch.autograd — no classes, no nn.Parameters.

Implementation strategy: every gate is computed as the matrix exponential of
its generator, using torch.linalg.matrix_exp. This approach is:
  - Mathematically unambiguous (no sign or index convention errors)
  - Fully differentiable through the Padé approximant
  - Free of in-place tensor modifications (which break autograd)

Single-mode gates return shape (D, D).
Two-mode gates return shape (D, D, D, D) indexed [out_i, out_j, in_i, in_j].

References:
    - Serafini, "Quantum Continuous Variables" (2017)
    - Killoran et al., "Continuous-variable quantum neural networks" (2019)
      https://arxiv.org/abs/1806.06871
"""

from __future__ import annotations

import torch


def _ladder_operators(
    cutoff_dim: int,
    device: torch.device,
    dtype: torch.dtype = torch.complex128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the annihilation (a) and creation (a†) operators in Fock basis.

    a|n〉 = sqrt(n)|n-1〉  →  a_{n-1,n} = sqrt(n)   (super-diagonal)
    a†|n〉 = sqrt(n+1)|n+1〉  →  a†_{n+1,n} = sqrt(n+1)  (sub-diagonal)

    Returns:
        (a, a_dag) each of shape (D, D), complex dtype.
    """
    ns = torch.arange(1, cutoff_dim, dtype=torch.float64, device=device)
    a_dag = torch.diag(torch.sqrt(ns), diagonal=-1).to(dtype)
    a = a_dag.mH.contiguous()   # conjugate transpose = annihilation operator
    return a, a_dag


def rotation_matrix(phi: torch.Tensor, cutoff_dim: int) -> torch.Tensor:
    """Phase-space rotation R(φ) = exp(iφ n̂) in the Fock basis.

    Generator: G = iφ n̂  where n̂ = a†a (number operator).
    R is diagonal: R_{nn} = exp(inφ).

    Args:
        phi:        Rotation angle (real scalar tensor), in radians.
        cutoff_dim: Fock space truncation D.

    Returns:
        Complex tensor of shape (D, D). Differentiable w.r.t. phi.
    """
    device = phi.device
    a, a_dag = _ladder_operators(cutoff_dim, device)
    n_op = a_dag @ a
    generator = 1j * phi.to(torch.complex128) * n_op
    return torch.linalg.matrix_exp(generator)


def displacement_matrix(alpha: torch.Tensor, cutoff_dim: int) -> torch.Tensor:
    """Displacement operator D(α) = exp(α a† - α* a) in the Fock basis.

    Generator: G = α a† - α* a.
    Applying D(α) to the vacuum creates a coherent state |α〉 with
    〈x̂〉 = sqrt(2) Re(α) and 〈p̂〉 = sqrt(2) Im(α).

    Args:
        alpha:      Complex displacement amplitude (scalar complex or real tensor).
        cutoff_dim: Fock space truncation D.

    Returns:
        Complex tensor of shape (D, D). Differentiable w.r.t. alpha.
    """
    device = alpha.device
    alpha_c = alpha.to(torch.complex128)
    a, a_dag = _ladder_operators(cutoff_dim, device)
    generator = alpha_c * a_dag - alpha_c.conj() * a
    return torch.linalg.matrix_exp(generator)


def squeezing_matrix(r: torch.Tensor, phi: torch.Tensor, cutoff_dim: int) -> torch.Tensor:
    """Single-mode squeezing S(r, φ) = exp((r/2)(e^{-iφ} a² - e^{iφ} a†²)).

    Generator: G = (r/2)(e^{-iφ} a² - e^{iφ} a†²).
    Squeezing reduces noise in one quadrature at the expense of the other.
    Only connects same-parity Fock states (even↔even, odd↔odd).

    Args:
        r:          Squeezing magnitude (real scalar, r ≥ 0).
        phi:        Squeezing angle in radians (real scalar).
        cutoff_dim: Fock space truncation D.

    Returns:
        Complex tensor of shape (D, D). Differentiable w.r.t. r and phi.
    """
    device = r.device
    a, a_dag = _ladder_operators(cutoff_dim, device)
    e_iphi = torch.exp(1j * phi.to(torch.complex128))
    r_c = r.to(torch.complex128)
    generator = (r_c / 2) * (e_iphi.conj() * a @ a - e_iphi * a_dag @ a_dag)
    return torch.linalg.matrix_exp(generator)


def beamsplitter_matrix(
    theta: torch.Tensor, phi: torch.Tensor, cutoff_dim: int
) -> torch.Tensor:
    """Two-mode beamsplitter BS(θ, φ) = exp(θ(e^{iφ} a†b - e^{-iφ} ab†)).

    Generator in the D²×D² tensor product space:
        G = θ(e^{iφ} a†⊗b - e^{-iφ} a⊗b†) = θ(e^{iφ} kron(a†,a) - e^{-iφ} kron(a,a†))

    The beamsplitter has transmittivity T = cos²(θ) and reflectivity R = sin²(θ).

    Args:
        theta:      Beamsplitter mixing angle (real scalar).
        phi:        Phase angle (real scalar).
        cutoff_dim: Fock space truncation D.

    Returns:
        Complex tensor of shape (D, D, D, D) indexed [out_i, out_j, in_i, in_j].
        Differentiable w.r.t. theta and phi.
    """
    D = cutoff_dim
    device = theta.device
    a, a_dag = _ladder_operators(D, device)

    theta_c = theta.to(torch.complex128)
    e_iphi = torch.exp(1j * phi.to(torch.complex128))

    # Generator in D²×D² space (tensor product of two D-dimensional Fock spaces)
    generator = theta_c * (e_iphi * torch.kron(a_dag, a) - e_iphi.conj() * torch.kron(a, a_dag))
    BS_flat = torch.linalg.matrix_exp(generator)   # (D², D²)
    return BS_flat.reshape(D, D, D, D)             # [out_a, out_b, in_a, in_b]


def two_mode_squeezing_matrix(
    r: torch.Tensor, phi: torch.Tensor, cutoff_dim: int
) -> torch.Tensor:
    """Two-mode squeezing S2(r, φ) = exp(r(e^{-iφ} ab - e^{iφ} a†b†)).

    Generator in D²×D² space:
        G = r(e^{-iφ} kron(a,a) - e^{iφ} kron(a†,a†))

    S2 produces entangled two-mode squeezed (EPR) states. In the limit r→∞
    it creates a perfectly entangled state (ideal EPR pair).

    Args:
        r:          Squeezing parameter (real scalar, r ≥ 0).
        phi:        Phase angle (real scalar).
        cutoff_dim: Fock space truncation D.

    Returns:
        Complex tensor of shape (D, D, D, D) indexed [out_i, out_j, in_i, in_j].
        Differentiable w.r.t. r and phi.
    """
    D = cutoff_dim
    device = r.device
    a, a_dag = _ladder_operators(D, device)

    r_c = r.to(torch.complex128)
    e_iphi = torch.exp(1j * phi.to(torch.complex128))

    generator = r_c * (e_iphi.conj() * torch.kron(a, a) - e_iphi * torch.kron(a_dag, a_dag))
    S2_flat = torch.linalg.matrix_exp(generator)   # (D², D²)
    return S2_flat.reshape(D, D, D, D)             # [out_a, out_b, in_a, in_b]

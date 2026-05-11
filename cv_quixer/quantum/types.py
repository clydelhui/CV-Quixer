"""Shared type vocabulary for the CV quantum simulation engine.

No logic lives here — only types and enums used across every other module.
"""

from enum import Enum, auto

import torch

# A Fock state for N modes with cutoff D is a complex tensor of shape (D,)*N
FockTensor = torch.Tensor   # dtype=complex128 or complex64, shape=(D, D, ..., D)

# A single-mode gate matrix or rank-4 two-mode gate tensor
GateMatrix = torch.Tensor   # shape=(D, D) for single-mode; (D, D, D, D) for two-mode


class GradMode(Enum):
    """Gradient computation strategy for CV quantum circuits."""

    BACKPROP = auto()
    """Standard torch.autograd through Fock-basis gate matrix operations.

    All gate matrices are differentiable functions of their parameters.
    Gradients propagate automatically through the einsum chain.
    Zero overhead vs a classical network — no extra forward passes.
    Use this for most experiments.
    """

    PARAMETER_SHIFT = auto()
    """Parameter shift rule via a custom torch.autograd.Function.

    For each scalar parameter θᵢ, the gradient is estimated as:
        ∂E/∂θᵢ ≈ r · [E(θᵢ + s) − E(θᵢ − s)]
    where s is the shift value (default π/2) and r = 1/(2 sin(s)).

    Requires 2·N_params additional circuit evaluations per backward pass.
    Use when you want to simulate hardware-realistic gradient estimation.
    """

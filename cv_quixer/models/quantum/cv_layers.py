"""Reusable CV quantum circuit primitives.

A single CV transformer layer follows the architecture from Killoran et al.
(2019): Interferometer → Squeezing → Interferometer → Displacement → Kerr.

The Kerr non-linearity is the only non-Gaussian gate; it requires the Fock
backend (strawberryfields.fock). For purely Gaussian experiments, omit Kerr
and use strawberryfields.gaussian instead.

Reference:
    Killoran et al., "Continuous-variable quantum neural networks" (2019)
"""

import pennylane as qml
import torch
import torch.nn as nn


def interferometer(params: torch.Tensor, wires: list[int]) -> None:
    """Apply a general linear-optical interferometer (beam-splitter mesh).

    Implements a rectangular mesh of beam splitters and phase shifters that
    realises an arbitrary N×N unitary on N modes (Clements decomposition).

    Args:
        params: 1-D tensor of length num_modes * (num_modes - 1) / 2 * 3 +
                num_modes containing all beam-splitter angles, phase shifts,
                and final phase shifts.
        wires: Wire indices for the N modes.
    """
    n = len(wires)
    idx = 0
    # Rectangular beam-splitter mesh (Clements-style, row by row)
    for layer in range(n):
        for i in range(layer % 2, n - 1, 2):
            qml.Beamsplitter(params[idx], params[idx + 1], wires=[wires[i], wires[i + 1]])
            idx += 2
    # Per-mode phase shifts
    for i in range(n):
        qml.Rotation(params[idx], wires=wires[i])
        idx += 1


def cv_layer(
    theta: torch.Tensor,
    phi: torch.Tensor,
    varphi: torch.Tensor,
    r: torch.Tensor,
    r_phi: torch.Tensor,
    displacement: torch.Tensor,
    kappa: torch.Tensor,
    wires: list[int],
    use_kerr: bool = True,
) -> None:
    """Apply one complete CV neural network layer to a set of modes.

    Layer structure: Interferometer → Squeezing → Interferometer →
                     Displacement → Kerr (optional)

    Args:
        theta:       Beam-splitter angles for first interferometer.
        phi:         Phase shifts for first interferometer.
        varphi:      All parameters for second interferometer (combined).
        r:           Squeezing magnitude per mode.
        r_phi:       Squeezing angle per mode.
        displacement: Displacement amplitude per mode (real-valued).
        kappa:       Kerr interaction strength per mode.
        wires:       Wire indices.
        use_kerr:    Whether to apply the Kerr non-linearity. Set False when
                     using the Gaussian backend.
    """
    n = len(wires)
    interferometer(torch.cat([theta, phi]), wires)
    for i in range(n):
        qml.Squeezing(r[i], r_phi[i], wires=wires[i])
    interferometer(varphi, wires)
    for i in range(n):
        qml.Displacement(displacement[i], 0.0, wires=wires[i])
    if use_kerr:
        for i in range(n):
            qml.Kerr(kappa[i], wires=wires[i])


def interferometer_param_count(num_modes: int) -> int:
    """Return the number of parameters needed for one interferometer."""
    n = num_modes
    # beam splitter pairs: n*(n-1)/2 pairs × 2 params + n phase shifts
    return n * (n - 1) + n


class CVLayer(nn.Module):
    """Parameterised CV neural network layer with learnable weights.

    Wraps cv_layer() with nn.Parameter tensors so that PyTorch autograd
    propagates gradients through the PennyLane circuit via the
    parameter-shift rule.
    """

    def __init__(self, num_modes: int, use_kerr: bool = True) -> None:
        super().__init__()
        n = num_modes
        ic = interferometer_param_count(n)

        self.theta = nn.Parameter(torch.randn(ic // 2))
        self.phi = nn.Parameter(torch.randn(ic // 2))
        self.varphi = nn.Parameter(torch.randn(ic))
        self.r = nn.Parameter(torch.zeros(n))
        self.r_phi = nn.Parameter(torch.zeros(n))
        self.displacement = nn.Parameter(torch.zeros(n))
        self.kappa = nn.Parameter(torch.zeros(n))
        self.use_kerr = use_kerr
        self.num_modes = num_modes

    def apply(self, wires: list[int]) -> None:
        """Apply this layer's gates inside a QNode context."""
        cv_layer(
            self.theta, self.phi, self.varphi,
            self.r, self.r_phi,
            self.displacement, self.kappa,
            wires, self.use_kerr,
        )

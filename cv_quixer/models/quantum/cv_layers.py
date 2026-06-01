"""Parameterised CV neural network layer.

CVLayer owns the nn.Parameter tensors for one complete CV transformer layer.
The apply() method is the main implementation site — define the gate sequence
for your layer architecture here.

Typical Killoran et al. (2019) structure:
    Interferometer → Squeezing → Interferometer → Displacement → Kerr

The engine tools to use inside apply():
    from cv_quixer.quantum import (
        clements_interferometer,
        displacement_matrix, squeezing_matrix, rotation_matrix,
        beamsplitter_matrix, kerr_matrix, cubic_phase_matrix,
    )
    state = circuit.apply_single_mode_gate(gate_matrix, mode_index, state)
    state = circuit.apply_two_mode_gate(gate_matrix, mode_i, mode_j, state)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from cv_quixer.quantum import CVCircuit, FockState

# Re-export so test_cv_quixer.py::TestInterferometerParamCount passes unchanged
from cv_quixer.quantum import interferometer_param_count  # noqa: F401


class CVLayer(nn.Module):
    """One parameterised CV neural network layer (scaffolding — not used in CVQuixer).

    The current model uses HyperCVAttentionHead with an iterative gate application
    approach. CVLayer exists as a template for a future multi-layer architecture
    where stacked CVLayer blocks replace the single HyperCVAttention block.

    Owns all trainable parameters as nn.Parameter tensors. Implement the
    gate sequence inside apply().

    Parameter layout (Killoran et al. 2019 default — adjust to your design):
        theta, phi   — first interferometer beamsplitter angles / phases
        varphi       — second interferometer (all params combined)
        r, r_phi     — squeezing magnitude and angle per mode
        displacement — displacement amplitude per mode
        kappa        — Kerr coupling per mode

    Args:
        num_modes: Number of bosonic modes.
    """

    def __init__(self, num_modes: int) -> None:
        super().__init__()
        n = num_modes
        ic = interferometer_param_count(n)

        # First interferometer
        self.theta = nn.Parameter(torch.randn(ic // 2))
        self.phi = nn.Parameter(torch.randn(ic // 2))
        # Second interferometer (combined vector)
        self.varphi = nn.Parameter(torch.randn(ic))
        # Squeezing
        self.r = nn.Parameter(torch.zeros(n))
        self.r_phi = nn.Parameter(torch.zeros(n))
        # Displacement
        self.displacement = nn.Parameter(torch.zeros(n))
        # Kerr
        self.kappa = nn.Parameter(torch.zeros(n))

        self.num_modes = num_modes

    def apply(self, circuit: CVCircuit, state: FockState) -> FockState:
        """Apply this layer's gate sequence to state.

        This is the primary implementation site. Use circuit.apply_single_mode_gate()
        and circuit.apply_two_mode_gate() together with gate matrix functions from
        cv_quixer.quantum to build your circuit.

        Args:
            circuit: CVCircuit executor (stateless).
            state:   Input FockState. Not mutated — always return a new state.

        Returns:
            New FockState after the gate sequence.

        Example skeleton:
            from cv_quixer.quantum import clements_interferometer, squeezing_matrix, kerr_matrix

            # 1. First interferometer
            state = clements_interferometer(self.theta[...], self.phi[...], ..., circuit, state)

            # 2. Squeezing
            for i in range(self.num_modes):
                mat = squeezing_matrix(self.r[i], self.r_phi[i], circuit.cutoff_dim)
                state = circuit.apply_single_mode_gate(mat, i, state)

            # 3. Second interferometer, displacement, Kerr ...

            return state
        """
        raise NotImplementedError(
            "Implement the gate sequence for CVLayer.apply(). "
            "See the docstring above for a skeleton."
        )

"""Clements rectangular beamsplitter mesh (interferometer).

An N-mode interferometer implementing an arbitrary N×N unitary via a
brick-wall mesh of beamsplitters and phase shifters. This is the standard
building block for linear optical neural networks and CV quantum circuits.

Reference:
    Clements et al., "Optimal design for universal multiport interferometers"
    Optica 3, 1460 (2016). https://doi.org/10.1364/OPTICA.3.001460
"""

from __future__ import annotations

import torch

from cv_quixer.quantum.circuit import CVCircuit
from cv_quixer.quantum.gates.gaussian import beamsplitter_matrix, rotation_phases
from cv_quixer.quantum.state import FockState


def interferometer_param_count(num_modes: int) -> int:
    """Number of parameters required for one full interferometer.

    A rectangular Clements mesh on N modes has:
        - N*(N-1)/2 beamsplitter pairs, each with 2 parameters (θ, φ) → N*(N-1) params
        - N per-mode phase shifts → N params
    Total: N*(N-1) + N = N²

    This matches the existing cv_layers.py convention and is re-exported here
    so that the old import path (from cv_quixer.models.quantum.cv_layers import
    interferometer_param_count) continues to work after that file is refactored.

    Args:
        num_modes: Number of bosonic modes N.

    Returns:
        Total parameter count for one interferometer.
    """
    return num_modes * (num_modes - 1) + num_modes


def clements_interferometer(
    bs_angles: torch.Tensor,
    bs_phases: torch.Tensor,
    mode_phases: torch.Tensor,
    circuit: CVCircuit,
    state: FockState,
) -> FockState:
    """Apply a Clements rectangular beamsplitter mesh to a FockState.

    The mesh is a layer-by-layer brick-wall topology:
      - Layer 0: BS on pairs (0,1), (2,3), (4,5), ...
      - Layer 1: BS on pairs (1,2), (3,4), (5,6), ...
      - Layer 2: same as layer 0, etc.

    After all BS layers, a per-mode phase shift (rotation) is applied.

    Args:
        bs_angles:   Beamsplitter θ angles. Shape: (N*(N-1)//2,).
                     Each pair (θ, φ) defines one BS.
        bs_phases:   Beamsplitter φ phases. Shape: (N*(N-1)//2,).
        mode_phases: Per-mode rotation angles. Shape: (N,).
        circuit:     CVCircuit executor providing apply_* methods.
        state:       Input FockState. Not mutated.

    Returns:
        New FockState after applying the full interferometer.
    """
    n = circuit.num_modes
    D = circuit.cutoff_dim

    bs_idx = 0
    for layer in range(n):
        for i in range(layer % 2, n - 1, 2):
            theta = bs_angles[bs_idx]
            phi = bs_phases[bs_idx]
            bs_idx += 1
            gate = beamsplitter_matrix(theta, phi, D).to(state.device)
            state = circuit.apply_two_mode_gate(gate, i, i + 1, state)

    # Per-mode phase shifts
    for i in range(n):
        phases = rotation_phases(mode_phases[i], D).to(state.device)
        state = circuit.apply_single_mode_phases(phases, i, state)

    return state

"""CV Quantum Simulation Engine — public API.

This package provides a pure PyTorch Fock-basis simulation engine for
continuous-variable (CV) quantum circuits. It has no dependency on PennyLane
or any other quantum framework.

Typical usage:
    from cv_quixer.quantum import FockState, CVCircuit, GradMode
    from cv_quixer.quantum import displacement_matrix, kerr_matrix

    circuit = CVCircuit(num_modes=4, cutoff_dim=10)
    state = FockState.vacuum(4, 10)

    alpha = torch.tensor(0.5 + 0.3j, requires_grad=True)
    D = displacement_matrix(alpha, cutoff_dim=10)
    state = circuit.apply_single_mode_gate(D, mode=0, state=state)
    x_exp = circuit.measure_quadrature_x(0, state)   # <x> should be ~ sqrt(2) * Re(alpha)
    x_exp.backward()                                  # gradient flows into alpha
"""

from cv_quixer.quantum.circuit import CVCircuit
from cv_quixer.quantum.gates import (
    beamsplitter_matrix,
    cubic_phase_matrix,
    displacement_matrix,
    kerr_matrix,
    rotation_matrix,
    squeezing_matrix,
    two_mode_squeezing_matrix,
)
from cv_quixer.quantum.grad import ParameterShiftFunction
from cv_quixer.quantum.interferometer import clements_interferometer, interferometer_param_count
from cv_quixer.quantum.ops import (
    number_operator_matrix,
    quadrature_p_matrix,
    quadrature_x_matrix,
)
from cv_quixer.quantum.state import FockState
from cv_quixer.quantum.types import GradMode

__all__ = [
    # Core types
    "GradMode",
    "FockState",
    "CVCircuit",
    # Gate matrix factories — Gaussian
    "displacement_matrix",
    "squeezing_matrix",
    "rotation_matrix",
    "beamsplitter_matrix",
    "two_mode_squeezing_matrix",
    # Gate matrix factories — non-Gaussian
    "kerr_matrix",
    "cubic_phase_matrix",
    # Interferometer
    "clements_interferometer",
    "interferometer_param_count",
    # Observable operators
    "quadrature_x_matrix",
    "quadrature_p_matrix",
    "number_operator_matrix",
    # Gradient utilities
    "ParameterShiftFunction",
]

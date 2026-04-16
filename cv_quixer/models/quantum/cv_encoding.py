"""Encoding classical image patches into continuous-variable (CV) quantum states.

The standard approach for CV quantum machine learning is displacement encoding:
each element of the input vector sets the displacement amplitude of a bosonic
mode. This places the input information in the first moment (mean field) of the
Wigner function, which is accessible to subsequent Gaussian operations.

Reference:
    Killoran et al., "Continuous-variable quantum neural networks" (2019)
    https://arxiv.org/abs/1806.06871
"""

import pennylane as qml
import torch
import torch.nn as nn


class DisplacementEncoding(nn.Module):
    """Encode a patch vector into bosonic modes via displacement gates.

    Each scalar element x_i of the input patch is mapped to a displacement
    amplitude alpha_i = scale * x_i applied to mode i. A learned scale
    parameter allows the model to adapt the encoding magnitude during training.

    Args:
        patch_dim: Dimension of each input patch vector.
        num_modes: Number of bosonic modes to encode into. If num_modes <
                   patch_dim the patch is split across repeated encoding layers;
                   if num_modes > patch_dim the remaining modes receive zero
                   displacement.
    """

    def __init__(self, patch_dim: int, num_modes: int) -> None:
        super().__init__()
        self.patch_dim = patch_dim
        self.num_modes = num_modes
        # Learnable per-mode scale factors (initialised to 1)
        self.scale = nn.Parameter(torch.ones(min(patch_dim, num_modes)))

    def encode(self, patch: torch.Tensor, wires: list[int]) -> None:
        """Apply displacement gates to encode a single patch (called inside a QNode).

        Args:
            patch: 1-D tensor of length patch_dim (classical values).
            wires: List of wire indices to encode onto.
        """
        n = min(self.patch_dim, self.num_modes)
        for i in range(n):
            alpha = self.scale[i] * patch[i]
            qml.Displacement(alpha, 0.0, wires=wires[i])

"""Encoding classical image patches into continuous-variable (CV) quantum states.

The standard approach for CV quantum machine learning is displacement encoding:
each element of the input vector sets the displacement amplitude of a bosonic
mode. This places the input information in the first moment (mean field) of the
Wigner function, which is accessible to subsequent Gaussian operations.

Reference:
    Killoran et al., "Continuous-variable quantum neural networks" (2019)
    https://arxiv.org/abs/1806.06871
"""

import torch
import torch.nn as nn


class DisplacementEncoding(nn.Module):
    """Encode a patch vector into bosonic modes via displacement gates.

    Each scalar element x_i of the input patch is mapped to a displacement
    amplitude alpha_i = scale_i * x_i applied to mode i. A learned scale
    parameter allows the model to adapt the encoding magnitude during training.

    Args:
        patch_dim: Dimension of each input patch vector.
        num_modes: Number of bosonic modes to encode into.
    """

    def __init__(self, patch_dim: int, num_modes: int) -> None:
        super().__init__()
        self.patch_dim = patch_dim
        self.num_modes = num_modes
        self.scale = nn.Parameter(torch.ones(min(patch_dim, num_modes)))

    def encoding_alphas(self, patch: torch.Tensor) -> torch.Tensor:
        """Return the displacement amplitude vector for a single patch.

        Args:
            patch: 1-D real tensor of length patch_dim.

        Returns:
            Complex tensor of shape (min(patch_dim, num_modes),) where
            alpha_i = scale_i * patch_i. The circuit owner (CVAttention)
            passes each element to displacement_matrix() and applies the gate.
        """
        n = min(self.patch_dim, self.num_modes)
        alphas = self.scale * patch[:n]
        return alphas.to(torch.complex128)

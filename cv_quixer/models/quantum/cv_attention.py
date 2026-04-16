"""CV Quantum Attention Block — the core novelty of the CV-Quixer model.

Classical self-attention computes:
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

In the CV quantum formulation (the 'Quixer' design), the query-key similarity
is computed by a beam-splitter network (interferometer) on bosonic modes, and
the readout is performed by homodyne measurement of the quadrature operators.
Non-Gaussian (Kerr) interactions introduce the nonlinearity that allows the
model to go beyond purely linear transformations.

This module defines one CV attention layer that:
1. Encodes each patch into bosonic modes via DisplacementEncoding.
2. Applies a stack of CVLayer circuits to perform attention-like mixing.
3. Reads out the quadrature expectation values <x_i> via homodyne measurement.
4. Projects the readout back to the required output dimension.

Note on simulation cost:
    The Fock backend truncates at cutoff_dim Fock states per mode, so memory
    scales as cutoff_dim^num_modes. Keep num_modes small (≤8) for tractability.

Reference:
    Killoran et al., "Continuous-variable quantum neural networks" (2019)
    https://arxiv.org/abs/1806.06871
"""

from __future__ import annotations

import pennylane as qml
import torch
import torch.nn as nn

from cv_quixer.config.schema import QuantumConfig
from cv_quixer.models.quantum.cv_encoding import DisplacementEncoding
from cv_quixer.models.quantum.cv_layers import CVLayer


class CVAttention(nn.Module):
    """One CV quantum attention block.

    Args:
        patch_dim: Dimensionality of each input patch vector.
        config: QuantumConfig controlling num_modes, num_layers, cutoff_dim,
                and the PennyLane backend device string.
    """

    def __init__(self, patch_dim: int, config: QuantumConfig) -> None:
        super().__init__()
        self.num_modes = config.num_modes
        self.patch_dim = patch_dim
        self.wires = list(range(config.num_modes))
        self.use_kerr = "fock" in config.backend

        self.encoding = DisplacementEncoding(patch_dim, config.num_modes)
        self.cv_layers = nn.ModuleList([
            CVLayer(config.num_modes, use_kerr=self.use_kerr)
            for _ in range(config.num_layers)
        ])

        dev = qml.device(config.backend, wires=config.num_modes, cutoff_dim=config.cutoff_dim)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def _circuit(patch: torch.Tensor, layer_params: list) -> list:
            self.encoding.encode(patch, self.wires)
            for lp in layer_params:
                lp.apply(self.wires)
            return [qml.expval(qml.QuadX(w)) for w in self.wires]

        self._circuit = _circuit

        # Project the num_modes readout values back to patch_dim
        self.readout_proj = nn.Linear(config.num_modes, patch_dim)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """Apply CV attention to a sequence of patches.

        Args:
            patches: Tensor of shape (batch_size, num_patches, patch_dim).

        Returns:
            Tensor of shape (batch_size, num_patches, patch_dim).
        """
        B, N, D = patches.shape
        outputs = []
        for b in range(B):
            patch_outputs = []
            for n in range(N):
                quadratures = self._circuit(patches[b, n], list(self.cv_layers))
                patch_outputs.append(torch.stack(quadratures))   # (num_modes,)
            outputs.append(torch.stack(patch_outputs))            # (N, num_modes)
        readout = torch.stack(outputs)                            # (B, N, num_modes)
        return self.readout_proj(readout)                         # (B, N, patch_dim)

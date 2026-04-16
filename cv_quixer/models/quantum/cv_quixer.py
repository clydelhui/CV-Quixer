"""Top-level CV Quantum Transformer (CV-Quixer) model.

Architecture overview:
    1. Linear patch embedding: project flat patch vectors to embed_dim.
    2. Learnable [CLS] token prepended to the patch sequence.
    3. Learnable positional embeddings added to all tokens.
    4. Stack of CV attention blocks (the quantum component).
    5. LayerNorm + linear classifier on the [CLS] token.

The CV attention blocks are the novel contribution. Everything else mirrors
a standard ViT to keep the comparison with the classical baseline fair.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from cv_quixer.config.schema import DataConfig, QuantumConfig
from cv_quixer.models.base import BaseVisionTransformer
from cv_quixer.models.quantum.cv_attention import CVAttention


class CVQuixer(BaseVisionTransformer):
    """Continuous-Variable Quantum Vision Transformer.

    Args:
        quantum_config: QuantumConfig with circuit hyperparameters.
        data_config: DataConfig for patch/image dimensions and num_classes.
    """

    def __init__(self, quantum_config: QuantumConfig, data_config: DataConfig) -> None:
        super().__init__()
        image_size = data_config.image_size
        patch_size = data_config.patch_size
        num_patches = (image_size // patch_size) ** 2
        patch_dim = patch_size * patch_size          # single-channel MNIST

        embed_dim = quantum_config.num_modes         # readout dim matches num_modes

        # 1. Patch embedding
        self.patch_embed = nn.Linear(patch_dim, embed_dim)

        # 2. [CLS] token and positional embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # 3. CV quantum attention blocks
        self.cv_attention = CVAttention(embed_dim, quantum_config)

        # 4. LayerNorm + classifier head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, data_config.num_classes)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patches: (batch_size, num_patches, patch_dim)

        Returns:
            Logits: (batch_size, num_classes)
        """
        B = patches.shape[0]

        x = self.patch_embed(patches)                           # (B, N, embed_dim)

        cls = self.cls_token.expand(B, -1, -1)                 # (B, 1, embed_dim)
        x = torch.cat([cls, x], dim=1)                         # (B, N+1, embed_dim)
        x = x + self.pos_embed                                  # (B, N+1, embed_dim)

        x = self.cv_attention(x)                                # (B, N+1, embed_dim)

        x = self.norm(x[:, 0])                                  # (B, embed_dim) — CLS
        return self.head(x)                                     # (B, num_classes)

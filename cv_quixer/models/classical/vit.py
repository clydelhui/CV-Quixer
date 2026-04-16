"""Classical Vision Transformer (ViT) baseline.

Uses the `timm` library for a lightweight, well-tested ViT implementation.
The wrapper conforms to BaseVisionTransformer so the Trainer and evaluation
code treat it identically to the quantum model.

Patches are pre-extracted by the data pipeline, so we bypass timm's internal
patching and feed the patch sequence directly to the transformer encoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from cv_quixer.config.schema import ClassicalConfig, DataConfig
from cv_quixer.models.base import BaseVisionTransformer


class ClassicalViT(BaseVisionTransformer):
    """Standard Vision Transformer classifier built from scratch.

    A minimal ViT matching the architecture used by the CV-Quixer:
    patch embedding → [CLS] token + positional embedding →
    transformer encoder → LayerNorm → linear head.

    Args:
        classical_config: Embed dim, number of heads/layers, MLP ratio.
        data_config:      Patch/image dimensions and number of output classes.
    """

    def __init__(self, classical_config: ClassicalConfig, data_config: DataConfig) -> None:
        super().__init__()
        image_size = data_config.image_size
        patch_size = data_config.patch_size
        num_patches = (image_size // patch_size) ** 2
        patch_dim = patch_size * patch_size          # single-channel MNIST

        D = classical_config.embed_dim

        # Patch embedding
        self.patch_embed = nn.Linear(patch_dim, D)

        # [CLS] token and positional embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, D))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, D))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D,
            nhead=classical_config.num_heads,
            dim_feedforward=int(D * classical_config.mlp_ratio),
            dropout=classical_config.dropout,
            batch_first=True,
            norm_first=True,     # pre-norm (more stable)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=classical_config.num_layers,
        )

        # Classifier head
        self.norm = nn.LayerNorm(D)
        self.head = nn.Linear(D, data_config.num_classes)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patches: (batch_size, num_patches, patch_dim)

        Returns:
            Logits: (batch_size, num_classes)
        """
        B = patches.shape[0]

        x = self.patch_embed(patches)                   # (B, N, D)

        cls = self.cls_token.expand(B, -1, -1)          # (B, 1, D)
        x = torch.cat([cls, x], dim=1)                  # (B, N+1, D)
        x = x + self.pos_embed

        x = self.transformer(x)                         # (B, N+1, D)

        x = self.norm(x[:, 0])                          # (B, D) — CLS token
        return self.head(x)                             # (B, num_classes)

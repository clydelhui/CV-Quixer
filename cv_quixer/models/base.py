from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseVisionTransformer(nn.Module, ABC):
    """Abstract base class shared by the classical ViT and CV Quantum Transformer.

    Both models must implement this interface so that the Trainer, evaluation
    code, and experiment scripts are fully model-agnostic.
    """

    @abstractmethod
    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """Run a forward pass.

        Args:
            patches: Tensor of shape (batch_size, num_patches, patch_dim).

        Returns:
            Logits tensor of shape (batch_size, num_classes).
        """
        ...

    def predict(self, patches: torch.Tensor) -> torch.Tensor:
        """Return predicted class indices for a batch of patch sequences.

        Args:
            patches: Tensor of shape (batch_size, num_patches, patch_dim).

        Returns:
            Predicted class indices of shape (batch_size,).
        """
        with torch.no_grad():
            logits = self.forward(patches)
        return logits.argmax(dim=-1)

    def get_num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

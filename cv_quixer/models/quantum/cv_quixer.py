"""CV Quantum Vision Transformer (CV-Quixer).

Architecture:
    PatchEmbed       — linear projection + learnable positional embeddings
    HyperCVAttention — M independent heads, each driven by a PatchHypernetwork
    CVDecoder        — classical MLP: concatenated head readouts → class logits

Each head encodes all N patches sequentially into a shared quantum register,
applying a hypernetwork-generated Gaussian unitary per patch. After all
patches, quadrature expectation values are measured. This gives a
whole-sequence readout of shape (num_modes,) per head.

An optional Fock truncation penalty can be added to the training loss via
`forward(patches, return_trunc_loss=True)`.

If QuantumConfig.target_params > 0, hyper_hidden_dim is auto-scaled at
__init__ time so the total trainable parameter count is within 5% of the
target.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from cv_quixer.config.schema import DataConfig, QuantumConfig
from cv_quixer.models.base import BaseVisionTransformer
from cv_quixer.models.quantum.cv_attention import (
    HyperCVAttention,
    _gate_param_count,
    norm_truncation_penalty,
    photon_number_penalty,
)
from cv_quixer.quantum import CVCircuit


# ---------------------------------------------------------------------------
# PatchEmbed
# ---------------------------------------------------------------------------


class PatchEmbed(nn.Module):
    """Linear patch projection with learnable positional embeddings.

    Applies a shared linear projection to every patch, then adds per-position
    embeddings. No [CLS] token — the quantum attention produces a single
    whole-sequence readout rather than a token-level output.

    Args:
        patch_dim:   Flat patch dimensionality (patch_size² for greyscale).
        embed_dim:   Output embedding width (hypernetwork input size).
        num_patches: Total number of patches per image.
    """

    def __init__(self, patch_dim: int, embed_dim: int, num_patches: int) -> None:
        super().__init__()
        self.proj = nn.Linear(patch_dim, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """Project and add positional embeddings.

        Args:
            patches: Tensor of shape (B, N, patch_dim).

        Returns:
            Tensor of shape (B, N, embed_dim).
        """
        return self.proj(patches) + self.pos_embed


# ---------------------------------------------------------------------------
# CVDecoder
# ---------------------------------------------------------------------------


class CVDecoder(nn.Module):
    """Two-layer MLP: concatenated head readouts → class logits.

    Args:
        in_dim:      num_heads × num_modes (total readout width).
        hidden_dim:  Width of the hidden layer.
        num_classes: Number of output classes.
    """

    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Decode readouts to logits.

        Args:
            x: Tensor of shape (B, in_dim).

        Returns:
            Logits tensor of shape (B, num_classes).
        """
        return self.net(x)


# ---------------------------------------------------------------------------
# Auto-scaling helper
# ---------------------------------------------------------------------------


def _param_count_formula(
    patch_dim: int,
    num_patches: int,
    embed_dim: int,
    num_heads: int,
    num_modes: int,
    hyper_hidden: int,
    decoder_hidden: int,
    num_classes: int,
    bs_topology: str,
    poly_degree: int,
) -> int:
    """Closed-form parameter count for CVQuixer."""
    patch_embed = patch_dim * embed_dim + embed_dim + num_patches * embed_dim
    gate_params = _gate_param_count(num_modes, bs_topology)
    # PatchHypernetwork: two Linear layers
    hypernetwork = embed_dim * hyper_hidden + hyper_hidden + hyper_hidden * gate_params + gate_params
    # LCUSumCoefficients: b_real + b_imag, one per patch
    lcu = 2 * num_patches
    # PolynomialCoefficients: c_0 … c_d
    poly = poly_degree + 1
    per_head = hypernetwork + lcu + poly
    heads = num_heads * per_head
    # CVDecoder
    decoder_in = num_heads * num_modes
    decoder = (decoder_in * decoder_hidden + decoder_hidden
               + decoder_hidden * num_classes + num_classes)
    return patch_embed + heads + decoder


def _resolve_hyper_hidden_dim(
    config: QuantumConfig,
    patch_dim: int,
    num_patches: int,
    num_classes: int,
) -> QuantumConfig:
    """Binary-search hyper_hidden_dim so total params ≈ target_params."""
    import dataclasses

    target = config.target_params

    def count(h: int) -> int:
        return _param_count_formula(
            patch_dim, num_patches, config.embed_dim,
            config.num_heads, config.num_modes,
            h, config.decoder_hidden_dim, num_classes,
            config.bs_topology, config.poly_degree,
        )

    lo, hi = 1, 4096
    # Expand hi until count(hi) >= target
    while count(hi) < target:
        hi *= 2

    while lo < hi:
        mid = (lo + hi) // 2
        if count(mid) < target:
            lo = mid + 1
        else:
            hi = mid

    best = lo
    # Pick whichever of best-1 / best is closer to target
    if best > 1 and abs(count(best - 1) - target) < abs(count(best) - target):
        best -= 1

    achieved = count(best)
    if abs(achieved - target) / max(target, 1) > 0.10:
        import warnings
        warnings.warn(
            f"target_params={target} but closest achievable is {achieved} "
            f"(hyper_hidden_dim={best}). "
            "Adjust num_modes, num_heads, or embed_dim for a tighter match.",
            stacklevel=3,
        )

    return dataclasses.replace(config, hyper_hidden_dim=best)


# ---------------------------------------------------------------------------
# CVQuixer
# ---------------------------------------------------------------------------


class CVQuixer(BaseVisionTransformer):
    """CV Quantum Vision Transformer.

    Components:
        patch_embed:  PatchEmbed (linear projection + positional embeddings)
        cv_attention: HyperCVAttention (M parallel heads, each hypernetwork-driven)
        decoder:      CVDecoder (readout MLP → class logits)

    The forward pass optionally returns a Fock truncation penalty term that
    should be added (scaled by trunc_lambda) to the cross-entropy loss during
    training:

        logits, trunc = model(patches, return_trunc_loss=True)
        loss = F.cross_entropy(logits, labels) + config.trunc_lambda * trunc

    Args:
        quantum_config: QuantumConfig — circuit and architecture hyperparameters.
        data_config:    DataConfig — patch/image dimensions and num_classes.
    """

    def __init__(
        self, quantum_config: QuantumConfig, data_config: DataConfig
    ) -> None:
        super().__init__()

        patch_size = data_config.patch_size
        image_size = data_config.image_size
        num_patches = (image_size // patch_size) ** 2
        patch_dim = patch_size * patch_size    # single-channel (greyscale)

        # Auto-scale hyper_hidden_dim if requested
        config = quantum_config
        if config.target_params > 0:
            config = _resolve_hyper_hidden_dim(
                config, patch_dim, num_patches, data_config.num_classes
            )

        self.config = config
        self.patch_embed = PatchEmbed(patch_dim, config.embed_dim, num_patches)
        self.cv_attention = HyperCVAttention(config.embed_dim, num_patches, config)
        self.decoder = CVDecoder(
            in_dim=config.num_heads * config.num_modes,
            hidden_dim=config.decoder_hidden_dim,
            num_classes=data_config.num_classes,
        )

        # Kept for truncation loss computation in forward()
        self._circuit = CVCircuit(config.num_modes, config.cutoff_dim)
        self.trunc_penalty = config.trunc_penalty
        self.trunc_lambda = config.trunc_lambda

    def forward(
        self,
        patches: torch.Tensor,
        return_trunc_loss: bool = False,
        return_success_prob: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Run the full CV-Quixer model.

        Args:
            patches:            Tensor of shape (B, N, patch_dim).
            return_trunc_loss:  If True and trunc_penalty != "none", also
                                return the mean truncation penalty across all
                                batch elements and heads.
            return_success_prob: Not yet implemented. Raises NotImplementedError.

        Returns:
            logits:     (B, num_classes) — always returned.
            trunc_loss: scalar — appended when return_trunc_loss=True and
                        trunc_penalty != "none".
        """
        if return_success_prob:
            raise NotImplementedError(
                "return_success_prob is not yet implemented. "
                "Per-head success_prob tensors are computed inside "
                "HyperCVAttentionHead but aggregation and exposure from "
                "CVQuixer is pending."
            )

        x = self.patch_embed(patches)                          # (B, N, embed_dim)
        readouts, states, _ = self.cv_attention(x)             # (B, M×n), [B][M], [B][M]
        logits = self.decoder(readouts)                        # (B, num_classes)

        if return_trunc_loss and self.trunc_penalty != "none":
            penalties: list[torch.Tensor] = []
            for b_states in states:
                for state in b_states:
                    if self.trunc_penalty == "norm":
                        penalties.append(norm_truncation_penalty(state))
                    else:
                        penalties.append(
                            photon_number_penalty(state, self._circuit)
                        )
            trunc_loss = torch.stack(penalties).mean()
            return logits, trunc_loss.to(logits.device)

        return logits

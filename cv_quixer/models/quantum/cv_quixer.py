"""CV Quantum Vision Transformer (CV-Quixer).

Architecture:
    HyperCVAttention — M independent heads, each driven by a CNNHypernetwork
    CVDecoder        — classical MLP: concatenated head readouts → class logits

Each head encodes all N patches sequentially into a shared quantum register,
applying a hypernetwork-generated Gaussian unitary per patch. After all
patches, quadrature expectation values are measured. This gives a
whole-sequence readout of shape (num_modes,) per head.

An optional Fock truncation penalty can be added to the training loss via
`forward(patches, return_trunc_loss=True)`.

If QuantumConfig.target_params > 0, cnn_channels_2 is auto-scaled at
__init__ time so the total trainable parameter count is within 5% of the
target.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import torch
import torch.nn as nn

from cv_quixer.config.schema import DataConfig, QuantumConfig
from cv_quixer.models.base import BaseVisionTransformer
from cv_quixer.models.quantum.cv_attention import (
    HyperCVAttention,
    _gate_param_count,
    _readout_total_dim,
    norm_truncation_penalty,
    photon_number_penalty,
)
from cv_quixer.quantum import CVCircuit


class CVQuixerOut(NamedTuple):
    """Structured return for ``CVQuixer.forward`` when any ``return_*`` flag
    is set. Fields not requested are ``None``. Name-based access means adding
    or reordering optional outputs can never silently mis-bind a caller
    (audit M5). With no flags set, ``forward`` returns a plain ``logits``
    tensor instead — preserving the ``BaseVisionTransformer`` contract.
    """

    logits: torch.Tensor
    trunc_loss: Optional[torch.Tensor] = None
    readouts: Optional[torch.Tensor] = None
    states: Optional[list[torch.Tensor]] = None
    success_probs: Optional[list[torch.Tensor]] = None


# ---------------------------------------------------------------------------
# CVDecoder
# ---------------------------------------------------------------------------


class CVDecoder(nn.Module):
    """Two-layer MLP: concatenated head readouts → class logits.

    Args:
        in_dim:      Total readout width = num_heads × len(observable_plan),
                     where observable_plan is the expanded per-head sequence
                     of (type, mode, n) entries derived from QuantumConfig.
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
    patch_size: int,
    num_patches: int,
    num_heads: int,
    num_modes: int,
    cnn_channels_1: int,
    cnn_channels_2: int,
    cnn_kernel_size: int,
    decoder_hidden: int,
    num_classes: int,
    bs_topology: str,
    poly_degree: int,
    observable_plan: list,
) -> int:
    """Closed-form parameter count for CVQuixer."""
    h_out = patch_size - 2 * (cnn_kernel_size - 1)
    feature_dim = cnn_channels_2 * h_out * h_out
    gate_params = _gate_param_count(num_modes, bs_topology)
    # CNNHypernetwork: Conv1 + Conv2 + Linear
    conv1 = cnn_channels_1 * cnn_kernel_size ** 2 + cnn_channels_1
    conv2 = cnn_channels_2 * cnn_channels_1 * cnn_kernel_size ** 2 + cnn_channels_2
    linear = feature_dim * gate_params + gate_params
    # LCUSumCoefficients: b_real + b_imag, one per patch
    lcu = 2 * num_patches
    # PolynomialCoefficients: c_0 … c_d
    poly = poly_degree + 1
    per_head = conv1 + conv2 + linear + lcu + poly
    heads = num_heads * per_head
    # CVDecoder — input width = num_heads × len(observable_plan)
    per_head_dim = _readout_total_dim(observable_plan)
    decoder_in = num_heads * per_head_dim
    decoder = (decoder_in * decoder_hidden + decoder_hidden
               + decoder_hidden * num_classes + num_classes)
    return heads + decoder


def _resolve_cnn_channels(
    config: QuantumConfig,
    patch_size: int,
    num_patches: int,
    num_classes: int,
) -> QuantumConfig:
    """Binary-search cnn_channels_2 so total params ≈ target_params."""
    import dataclasses

    target = config.target_params

    def count(c2: int) -> int:
        return _param_count_formula(
            patch_size, num_patches,
            config.num_heads, config.num_modes,
            config.cnn_channels_1, c2, config.cnn_kernel_size,
            config.decoder_hidden_dim, num_classes,
            config.bs_topology, config.poly_degree,
            config._observable_plan,
        )

    lo, hi = 1, 4096
    while count(hi) < target:
        hi *= 2

    while lo < hi:
        mid = (lo + hi) // 2
        if count(mid) < target:
            lo = mid + 1
        else:
            hi = mid

    best = lo
    if best > 1 and abs(count(best - 1) - target) < abs(count(best) - target):
        best -= 1

    achieved = count(best)
    if abs(achieved - target) / max(target, 1) > 0.10:
        import warnings
        warnings.warn(
            f"target_params={target} but closest achievable is {achieved} "
            f"(cnn_channels_2={best}). "
            "Adjust num_modes, num_heads, or cnn_channels_1 for a tighter match.",
            stacklevel=3,
        )

    return dataclasses.replace(config, cnn_channels_2=best)


# ---------------------------------------------------------------------------
# CVQuixer
# ---------------------------------------------------------------------------


class CVQuixer(BaseVisionTransformer):
    """CV Quantum Vision Transformer.

    Components:
        cv_attention: HyperCVAttention (M parallel heads, each CNN-hypernetwork-driven)
        decoder:      CVDecoder (readout MLP → class logits)

    The forward pass optionally returns a Fock truncation penalty term that
    should be added (scaled by trunc_lambda) to the cross-entropy loss during
    training:

        out = model(patches, return_trunc_loss=True)
        loss = F.cross_entropy(out.logits, labels)
        if out.trunc_loss is not None:
            loss = loss + config.trunc_lambda * out.trunc_loss

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

        # Auto-scale cnn_channels_2 if requested
        config = quantum_config
        if config.target_params > 0:
            config = _resolve_cnn_channels(
                config, patch_size, num_patches, data_config.num_classes
            )

        self.config = config
        self.cv_attention = HyperCVAttention(patch_size, num_patches, config)
        per_head_dim = _readout_total_dim(config._observable_plan)
        self.decoder = CVDecoder(
            in_dim=config.num_heads * per_head_dim,
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
        return_states: bool = False,
        return_readouts: bool = False,
    ) -> torch.Tensor | CVQuixerOut:
        """Run the full CV-Quixer model.

        Args:
            patches:            Tensor of shape (B, N, patch_dim).
            return_trunc_loss:  If True and trunc_penalty != "none", also
                                return the mean truncation penalty across all
                                batch elements and heads.
            return_success_prob: Not yet implemented. Raises NotImplementedError.
            return_states:      If True, also return ``(states, success_probs)``
                                as appended trailing tuple entries —
                                ``states`` is ``list[Tensor]`` (one per head,
                                shape ``(B, cutoff_dim, ..., cutoff_dim)``),
                                ``success_probs`` is ``list[Tensor]`` (one
                                ``(B,)`` per head). Diagnostic-only;
                                training path leaves this False.
            return_readouts:    If True, also return the pre-decoder readouts
                                ``(B, num_heads * len(observable_plan))``.

        Returns:
            With no return_* flag set: the plain ``logits`` tensor of shape
            ``(B, num_classes)`` (preserves the BaseVisionTransformer
            contract). With any flag set: a ``CVQuixerOut`` namedtuple whose
            fields are ``None`` unless requested —

              logits:        (B, num_classes) — always populated.
              trunc_loss:    scalar — populated when return_trunc_loss=True
                             and trunc_penalty != "none" (else None).
              readouts:      (B, num_heads * len(observable_plan)) — populated
                             when return_readouts=True.
              states:        list[Tensor] — populated when return_states=True.
              success_probs: list[Tensor] — populated when return_states=True.
        """
        if return_success_prob:
            raise NotImplementedError(
                "return_success_prob is not yet implemented. "
                "Per-head success_prob tensors are computed inside "
                "HyperCVAttentionHead but aggregation and exposure from "
                "CVQuixer is pending."
            )

        readouts, states, success_probs, trunc_loss = self.cv_attention(patches)
        logits = self.decoder(readouts)                           # (B, num_classes)

        if not (return_trunc_loss or return_readouts or return_states):
            return logits

        return CVQuixerOut(
            logits=logits,
            trunc_loss=(
                trunc_loss.to(logits.device)
                if return_trunc_loss and self.trunc_penalty != "none"
                else None
            ),
            readouts=readouts if return_readouts else None,
            states=states if return_states else None,
            success_probs=success_probs if return_states else None,
        )

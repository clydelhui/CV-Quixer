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

import dataclasses
from typing import NamedTuple, Optional

import torch
import torch.nn as nn

from cv_quixer.config.schema import DataConfig, QuantumConfig
from cv_quixer.models.base import BaseVisionTransformer
from cv_quixer.models.quantum.cv_attention import (
    HyperCVAttention,
    SharedCVAttention,
    _readout_total_dim,
    norm_truncation_penalty,
    photon_number_penalty,
)
from cv_quixer.quantum import CVCircuit
from cv_quixer.utils.params import autoscale_to_target


class CVQuixerOut(NamedTuple):
    """Structured return for ``CVQuixer.forward`` when any ``return_*`` flag
    is set. Fields not requested are ``None``. Name-based access means adding
    or reordering optional outputs can never silently mis-bind a caller
    (audit M5). With no flags set, ``forward`` returns a plain ``logits``
    tensor instead — preserving the ``BaseVisionTransformer`` contract.
    """

    logits: torch.Tensor
    trunc_loss: Optional[torch.Tensor] = None
    cvqnn_trunc_loss: Optional[torch.Tensor] = None
    readouts: Optional[torch.Tensor] = None
    states: Optional[list[torch.Tensor]] = None
    success_probs: Optional[list[torch.Tensor]] = None
    # Query-unitary truncation leakage mean_i(1 − ‖U_{q,i}|0⟩‖²) — only the
    # seq-to-seq stacked model has query unitaries, so the canonical models
    # leave this None (the stream does not exist for them, ADR-0003).
    query_trunc_loss: Optional[torch.Tensor] = None
    # NaN-forensics debug reductions (all detached; populated alongside
    # trunc_loss for the canonical models, None for the stacked model). Per-head
    # versions of the scalar streams plus per-param-type gate stats — see
    # cv_attention.HeadDebugStats. ``gate_param_*`` columns are ordered by the
    # model's ``gate_stat_labels`` property.
    trunc_loss_per_head: Optional[torch.Tensor] = None        # (H,)
    cvqnn_trunc_loss_per_head: Optional[torch.Tensor] = None  # (H,)
    gate_param_min_abs: Optional[torch.Tensor] = None         # (H, T)
    gate_param_zero_count: Optional[torch.Tensor] = None      # (H, T)


# ---------------------------------------------------------------------------
# CVDecoder
# ---------------------------------------------------------------------------


class CVDecoder(nn.Module):
    """MLP: concatenated head readouts → class logits.

    Args:
        in_dim:      Total readout width = num_heads × len(observable_plan),
                     where observable_plan is the expanded per-head sequence
                     of (type, mode, n) entries derived from QuantumConfig.
        hidden_dim:  Width of the hidden layer(s).
        num_classes: Number of output classes.
        num_layers:  Total Linear layers. 2 (default) = the historic
                     Linear(in→h)→ReLU→Linear(h→classes), whose state-dict keys
                     (``net.0``/``net.2``) are preserved. >2 inserts
                     (num_layers-2) extra hidden h→h ReLU blocks before the
                     output layer (appending ``net.4`` ...).
    """

    def __init__(
        self, in_dim: int, hidden_dim: int, num_classes: int, num_layers: int = 2
    ) -> None:
        super().__init__()
        if num_layers < 2:
            raise ValueError(f"decoder num_layers must be >= 2, got {num_layers}")
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Decode readouts to logits.

        Args:
            x: Tensor of shape (B, in_dim).

        Returns:
            Logits tensor of shape (B, num_classes).
        """
        return self.net(x)


# ---------------------------------------------------------------------------
# Shared forward base
# ---------------------------------------------------------------------------


class _CVQuixerBase(BaseVisionTransformer):
    """Shared forward for the CV-Quixer model family.

    Both ``CVQuixer`` (per-head CNN hypernetworks) and ``SharedCVQuixer`` (shared
    patch CNN + per-head linears) build a ``cv_attention`` module exposing the
    same 4-tuple forward contract and a ``decoder``; the forward below depends
    only on ``self.cv_attention``, ``self.decoder``, and ``self.trunc_penalty``,
    so it is identical for both and lives here.
    """

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
            return_success_prob: If True, also return the per-head raw
                                LCU/QSVT post-selection norms
                                ``‖P(M)|ψ⟩‖²`` as ``success_probs`` —
                                ``list[Tensor]``, one ``(B,)`` per head.
                                Raw, i.e. NOT divided by the block-encoding
                                subnormalisation λ² (derive λ from the LCU /
                                polynomial coefficients; see ADR-0002).
                                Essentially free — the values are already
                                computed by the attention forward.
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
              cvqnn_trunc_loss: scalar — the CVQNN block's truncation leakage
                             (1 − ‖W|ψ⟩‖²), populated when return_trunc_loss=True
                             (0 when cvqnn_num_layers == 0). Independent of
                             trunc_penalty; weighted by cvqnn_trunc_lambda in
                             the training loss.
              readouts:      (B, num_heads * len(observable_plan)) — populated
                             when return_readouts=True.
              states:        list[Tensor] — populated when return_states=True.
              success_probs: list[(B,) Tensor] — populated when
                             return_success_prob=True or return_states=True.
        """
        readouts, states, success_probs, trunc_loss, cvqnn_trunc_loss, debug = (
            self.cv_attention(patches)
        )
        logits = self.decoder(readouts)                           # (B, num_classes)

        if not (return_trunc_loss or return_readouts or return_states
                or return_success_prob):
            return logits

        return CVQuixerOut(
            logits=logits,
            trunc_loss=(
                trunc_loss.to(logits.device)
                if return_trunc_loss and self.trunc_penalty != "none"
                else None
            ),
            # W's penalty is its own concern, independent of trunc_penalty — it is
            # always computed (it is the norm used for the post-W renorm) and is 0
            # when cvqnn_num_layers == 0. Surfaced whenever trunc loss is requested.
            cvqnn_trunc_loss=(
                cvqnn_trunc_loss.to(logits.device)
                if return_trunc_loss
                else None
            ),
            readouts=readouts if return_readouts else None,
            states=states if return_states else None,
            success_probs=(
                success_probs if (return_success_prob or return_states)
                else None
            ),
            # NaN-forensics reductions: detached at source (HeadDebugStats), so
            # surfacing them alongside trunc_loss costs nothing and never holds
            # the autograd graph alive.
            trunc_loss_per_head=(
                debug.trunc_per_head.to(logits.device)
                if return_trunc_loss else None
            ),
            cvqnn_trunc_loss_per_head=(
                debug.cvqnn_trunc_per_head.to(logits.device)
                if return_trunc_loss else None
            ),
            gate_param_min_abs=(
                debug.gate_min_abs.to(logits.device)
                if return_trunc_loss else None
            ),
            gate_param_zero_count=(
                debug.gate_zero_count.to(logits.device)
                if return_trunc_loss else None
            ),
        )

    @property
    def gate_stat_labels(self) -> list[str]:
        """Column labels for the ``gate_param_*`` debug outputs, in order."""
        return list(self.cv_attention.heads[0]._gate_stat_labels)


# ---------------------------------------------------------------------------
# Shared config resolution
# ---------------------------------------------------------------------------


def _resolve_decoder_hidden(config: QuantumConfig) -> QuantumConfig:
    """Resolve ``decoder_hidden_dim`` from ``decoder_hidden_mult`` if set.

    With ``decoder_hidden_mult = c``, the decoder hidden width is sized relative
    to its input ``in_dim = num_heads × readout_width`` as
    ``decoder_hidden_dim = max(1, round(c × in_dim))``. Called *after* any
    ``target_params`` auto-scaling so ``num_heads`` is final; returns the config
    unchanged when the multiplier is None. Deterministic ⇒ idempotent on reload.
    """
    if config.decoder_hidden_mult is None:
        return config
    in_dim = config.num_heads * _readout_total_dim(config._observable_plan)
    hidden = max(1, round(config.decoder_hidden_mult * in_dim))
    return dataclasses.replace(config, decoder_hidden_dim=hidden)


# ---------------------------------------------------------------------------
# CVQuixer
# ---------------------------------------------------------------------------


class CVQuixer(_CVQuixerBase):
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

        # Auto-scale the configured knob to the parameter budget if requested.
        # Counting is done by actually building trial models and summing their
        # nn.Parameters (autoscale_to_target), so the budget is matched against
        # the real architecture rather than a hand-maintained formula.
        config = quantum_config
        if config.target_params > 0:
            # The 2D sinusoidal PE in CNNHypernetwork needs an even
            # feature_dim = cnn_channels_2 * h_out**2. When scaling cnn_channels_2
            # and h_out is odd (e.g. patch_size=7, kernel=3 → h_out=3), the knob
            # must step in 2s; every other knob steps in 1s.
            h_out = patch_size - 2 * (config.cnn_kernel_size - 1)
            step = (
                2
                if config.scaling_knob == "cnn_channels_2" and (h_out * h_out) % 2 == 1
                else 1
            )
            config = autoscale_to_target(
                build_fn=lambda c: CVQuixer(c, data_config),
                config=config,
                knob=config.scaling_knob,
                target=config.target_params,
                step=step,
            )

        # Size the decoder hidden dim relative to its input, if requested (after
        # auto-scaling, so num_heads — and thus the decoder input width — is final).
        config = _resolve_decoder_hidden(config)

        self.config = config
        self.cv_attention = HyperCVAttention(patch_size, num_patches, config)
        per_head_dim = _readout_total_dim(config._observable_plan)
        self.decoder = CVDecoder(
            in_dim=config.num_heads * per_head_dim,
            hidden_dim=config.decoder_hidden_dim,
            num_classes=data_config.num_classes,
            num_layers=config.decoder_num_layers,
        )

        # Kept for truncation loss computation in forward()
        self._circuit = CVCircuit(config.num_modes, config.cutoff_dim)
        self.trunc_penalty = config.trunc_penalty
        self.trunc_lambda = config.trunc_lambda
        self.cvqnn_trunc_lambda = config.cvqnn_trunc_lambda

    # forward() is inherited from _CVQuixerBase.


# ---------------------------------------------------------------------------
# SharedCVQuixer
# ---------------------------------------------------------------------------


class SharedCVQuixer(_CVQuixerBase):
    """CV Quantum Vision Transformer with a shared patch CNN (model="quantum_shared").

    Identical to ``CVQuixer`` except the per-head ``CNNHypernetwork`` is replaced
    by a single ``SharedPatchCNN`` (run once per patch) feeding per-head
    ``Linear`` projections. The shared CNN emits flattened conv features (+ 2D PE)
    of width ``cnn_channels_2 × h_out²`` directly — no extra projection. See
    ``SharedCVAttention``.

    Components:
        cv_attention: SharedCVAttention (shared CNN + M linear heads)
        decoder:      CVDecoder (readout MLP → class logits)

    Args:
        quantum_config: QuantumConfig — circuit and architecture hyperparameters.
                        ``scaling_knob`` is typically "num_heads" for this model.
        data_config:    DataConfig — patch/image dimensions and num_classes.
    """

    def __init__(
        self, quantum_config: QuantumConfig, data_config: DataConfig
    ) -> None:
        super().__init__()

        patch_size = data_config.patch_size
        image_size = data_config.image_size
        num_patches = (image_size // patch_size) ** 2

        # Auto-scale the configured knob to the parameter budget if requested.
        # Counting builds trial models and sums their nn.Parameters
        # (autoscale_to_target), so the budget is matched against the real
        # architecture. For this model the knob is typically num_heads.
        config = quantum_config
        if config.target_params > 0:
            config = autoscale_to_target(
                build_fn=lambda c: SharedCVQuixer(c, data_config),
                config=config,
                knob=config.scaling_knob,
                target=config.target_params,
                step=1,
            )

        # Size the decoder hidden dim relative to its input, if requested (after
        # auto-scaling, so num_heads — and thus the decoder input width — is final).
        config = _resolve_decoder_hidden(config)

        self.config = config
        self.cv_attention = SharedCVAttention(patch_size, num_patches, config)
        per_head_dim = _readout_total_dim(config._observable_plan)
        self.decoder = CVDecoder(
            in_dim=config.num_heads * per_head_dim,
            hidden_dim=config.decoder_hidden_dim,
            num_classes=data_config.num_classes,
            num_layers=config.decoder_num_layers,
        )

        # Kept for truncation loss computation in forward()
        self._circuit = CVCircuit(config.num_modes, config.cutoff_dim)
        self.trunc_penalty = config.trunc_penalty
        self.trunc_lambda = config.trunc_lambda
        self.cvqnn_trunc_lambda = config.cvqnn_trunc_lambda

    # forward() is inherited from _CVQuixerBase.

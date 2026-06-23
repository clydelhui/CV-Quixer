"""Seq-to-seq stacked CV-Quixer (model="quantum_stacked", ADR-0003).

A **seq-to-seq block** maps an N-token sequence to an N-token sequence. Within
a block, every position shares each head's LCU `M = Σ_i b_i U_i`, polynomial
`P`, and CVQNN block `W`; position i differs only in its query state — its
output token is the readout of

    W · P(M) · U_{q,i}|0⟩

with the query state renormalised before `P(M)` (its leakage is the separate
`query_trunc_loss` stream) and the usual post-selection / post-W renorms after.
The query unitary `U_{q,i}` is a second slice of the same hypernet output that
emits the LCU term `U_i` (key/query projections of one shared patch embedding).

Blocks stack: block 1 consumes raw patches via per-head CNN hypernetworks;
blocks >= 2 consume the previous block's (H×R)-wide tokens via per-head
linears, with an optional identity residual. The stack ends in mean-pooling
over positions or a canonical seq-to-one **aggregator block**
(pooling="quixer"), either way feeding the same (H×R)-wide CVDecoder as the
canonical models.

Engine: iterative, no LCU matrix materialised — the vmap nesting gains a query
axis (head → batch → query → patch), costing ~d·N² gate-plan applications per
head per block.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.func import functional_call, vmap

from cv_quixer.config.schema import DataConfig, QuantumConfig
from cv_quixer.models.base import BaseVisionTransformer
from cv_quixer.models.quantum.cv_attention import (
    _SUCCESS_PROB_FLOOR,
    CNNHypernetwork,
    LinearCVHead,
    _CVHeadBase,
    _readout_total_dim,
    _run_heads_vmap,
    _stack_head_state,
    _warn_failed_postselection,
)
from cv_quixer.models.quantum.cv_quixer import (
    CVDecoder,
    CVQuixerOut,
    _resolve_decoder_hidden,
)
from cv_quixer.quantum import FockState


# ---------------------------------------------------------------------------
# Seq-to-seq heads
# ---------------------------------------------------------------------------


class _Seq2SeqHeadBase(_CVHeadBase):
    """One seq-to-seq CV attention head: per-position readouts.

    ``_features_to_params`` must emit ``(N, 2 × gate_param_width)`` — the key
    slice (LCU terms ``U_i``) followed by the query slice (``U_{q,i}``).
    """

    def forward(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the head on one batch element's token sequence.

        Args:
            features: Real tensor of shape (N, in_dim).

        Returns:
            readouts:         (N, R) — one readout vector per position.
            states:           (N, D, ..., D) — final per-position states.
            success_probs:    (N,) — ‖P(M)|q_i⟩‖² per position.
            patch_trunc_loss: scalar — LCU-term leakage on the actual query
                              states, mean over positions (ADR-0003).
            query_trunc_loss: scalar — mean_i(1 − ‖U_{q,i}|0⟩‖²).
            w_trunc_loss:     scalar — W leakage, mean over positions.
        """
        classical_device = features.device
        quantum_device = (
            classical_device if classical_device.type != "mps"
            else torch.device("cpu")
        )
        D, n = self.cutoff_dim, self.num_modes
        dtype = self.torch_dtype
        N = features.shape[0]
        gp = self._gate_param_width

        # 1. One hypernet pass emits both slices for all positions.
        all_params = self._features_to_params(features).to(quantum_device)
        key_params = all_params[:, :gp]      # (N, gp) — LCU terms U_i
        query_params = all_params[:, gp:]    # (N, gp) — query unitaries U_{q,i}

        # 2. Query states U_{q,i}|0⟩, vmapped over positions. Leakage is the
        # separate query_trunc stream; states are renormalised before P(M) so
        # success_prob keeps its canonical meaning (pure polynomial
        # post-selection probability on a unit-norm input).
        vac = FockState.vacuum(n, D, quantum_device, dtype).data
        apply_query = lambda p: self._apply_patch_gates_to_data(
            p, vac, quantum_device, dtype
        )
        q_data = vmap(apply_query)(query_params)            # (N, D, ..., D)
        q_flat = q_data.reshape(N, -1)
        q_norm_sq = (q_flat.abs() ** 2).sum(dim=-1)         # (N,) real
        query_trunc_loss = (1.0 - q_norm_sq).mean()
        safe_qn = q_norm_sq.clamp(min=_SUCCESS_PROB_FLOOR)
        q_scaled = q_flat / safe_qn.sqrt().unsqueeze(-1)
        q_unit = torch.where(
            (q_norm_sq > _SUCCESS_PROB_FLOOR).unsqueeze(-1),
            q_scaled,
            torch.zeros_like(q_flat),
        )

        # 3. Per position: P(M)|q_i⟩ → post-select renorm → W → renorm →
        # readout. key_params is closed over (shared across the position vmap);
        # the trace-time branches mirror _CVHeadBase.forward exactly.
        def _one_position(q_vec: torch.Tensor):
            if not self._trunc_enabled:
                out_unnorm, sp = self._apply_polynomial_iterative_params(
                    key_params, q_vec, quantum_device, dtype
                )
                pos_trunc = torch.zeros(
                    (), device=quantum_device, dtype=self._real_dtype
                )
            elif len(self.poly_coeffs()) >= 2:
                out_unnorm, sp, pos_trunc = self._apply_polynomial_iterative_params(
                    key_params, q_vec, quantum_device, dtype, want_trunc=True
                )
            else:
                # poly_degree == 0: no LCU pass exists to fuse into — standalone
                # leakage pass of the LCU terms on this query state.
                q_grid = q_vec.reshape((D,) * n)
                outs = vmap(
                    lambda p: self._apply_patch_gates_to_data(
                        p, q_grid, quantum_device, dtype
                    )
                )(key_params)
                norm_sq = (outs.reshape(N, -1).abs() ** 2).sum(dim=-1)
                pos_trunc = (1.0 - norm_sq).mean()
                out_unnorm, sp = self._apply_polynomial_iterative_params(
                    key_params, q_vec, quantum_device, dtype
                )

            out_norm = self._postselect_renorm(out_unnorm, sp)
            out_w, w_trunc = self._apply_cvqnn(out_norm, quantum_device, dtype)
            state = FockState(out_w.reshape((D,) * n), n, D)
            readout = self._measure_plan(state)
            return readout, out_w.reshape((D,) * n), sp, pos_trunc, w_trunc

        readouts, states, sps, pos_truncs, w_truncs = vmap(_one_position)(q_unit)

        return (
            readouts.to(classical_device),   # (N, R)
            states,                          # (N, D, ..., D)
            sps,                             # (N,)
            pos_truncs.mean(),
            query_trunc_loss,
            w_truncs.mean(),
        )


class Seq2SeqCNNHead(_Seq2SeqHeadBase):
    """Block-1 seq-to-seq head: per-head CNN hypernetwork over raw patches.

    The hypernet's final linear emits 2× the op-plan width (key + query slices).

    Args:
        patch_size:  Side length of each raw image patch in pixels.
        num_patches: Sequence length N.
        config:      QuantumConfig.
    """

    def __init__(
        self, patch_size: int, num_patches: int, config: QuantumConfig
    ) -> None:
        super().__init__(num_patches, config)
        self.hypernetwork = CNNHypernetwork(
            patch_size, num_patches, config.num_modes,
            config.cnn_channels_1, config.cnn_channels_2, config.cnn_kernel_size,
            config.bs_topology, config.num_layers,
            cnn_num_conv_layers=config.cnn_num_conv_layers,
            hypernet_num_linear_layers=config.hypernet_num_linear_layers,
            param_count_multiplier=2,
            positional_encoding=config.positional_encoding,
        )

    def _features_to_params(self, features: torch.Tensor) -> torch.Tensor:
        return self.hypernetwork.forward_all(features)


class Seq2SeqLinearHead(_Seq2SeqHeadBase):
    """Deeper-block seq-to-seq head: per-head Linear over incoming tokens.

    Args:
        num_patches: Sequence length N.
        in_dim:      Incoming token width (num_heads × R of the block below).
        config:      QuantumConfig.
    """

    def __init__(
        self, num_patches: int, in_dim: int, config: QuantumConfig
    ) -> None:
        super().__init__(num_patches, config)
        self.hidden_linears = nn.ModuleList(
            nn.Linear(in_dim, in_dim)
            for _ in range(config.hypernet_num_linear_layers - 1)
        )
        self.linear = nn.Linear(in_dim, 2 * self._gate_param_width)

    def _features_to_params(self, features: torch.Tensor) -> torch.Tensor:
        for lin in self.hidden_linears:
            features = torch.tanh(lin(features))
        return self.linear(features)


# ---------------------------------------------------------------------------
# Multi-head seq-to-seq driver + block module
# ---------------------------------------------------------------------------


def _run_seq2seq_heads_vmap(
    heads: nn.ModuleList,
    num_heads: int,
    readout_total_dim: int,
    features: torch.Tensor,
) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor],
           torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run all seq-to-seq heads over a batch via nested vmap.

    Same stacked-params pattern as ``_run_heads_vmap`` (plain differentiable
    ``torch.stack``, head → batch nesting; each head internally vmaps positions
    then patches), but the per-element output is a token *sequence*.

    Args:
        heads:             ModuleList of homogeneous ``_Seq2SeqHeadBase``.
        num_heads:         Number of heads.
        readout_total_dim: Per-head readout width R.
        features:          Real tensor of shape (B, N, in_dim).

    Returns:
        tokens:           (B, N, num_heads × R).
        states:           list[(B, N, D, ..., D)] — one per head.
        success_probs:    list[(B, N)] — one per head.
        patch_trunc:      scalar — mean over heads × batch (already
                          position-meaned per head).
        query_trunc:      scalar — mean over heads × batch.
        w_trunc:          scalar — mean over heads × batch.
    """
    base = heads[0]
    stacked_p, stacked_b = _stack_head_state(heads)

    def _one_element(p, b, single_features):
        return functional_call(base, (p, b), (single_features,))

    def _one_head(p, b):
        return vmap(_one_element, in_dims=(None, None, 0))(p, b, features)

    rd_hb, st_hb, sp_hb, pt_hb, qt_hb, wt_hb = vmap(_one_head, in_dims=(0, 0))(
        stacked_p, stacked_b
    )
    # rd_hb: (H, B, N, R); st_hb: (H, B, N, D, ..., D); sp_hb: (H, B, N)
    # pt_hb, qt_hb, wt_hb: (H, B)

    _warn_failed_postselection(sp_hb, "head×batch×position")

    B, N = features.shape[0], features.shape[1]
    # (H, B, N, R) → (B, N, H·R): per-position [head0_R | head1_R | ...].
    tokens = rd_hb.permute(1, 2, 0, 3).reshape(B, N, num_heads * readout_total_dim)
    return (
        tokens,
        list(st_hb.unbind(0)),
        list(sp_hb.unbind(0)),
        pt_hb.mean(),
        qt_hb.mean(),
        wt_hb.mean(),
    )


class Seq2SeqCVAttention(nn.Module):
    """One seq-to-seq block: num_heads parallel seq-to-seq heads.

    Exactly one of ``patch_size`` / ``in_dim`` must be given: block 1 takes raw
    patches (CNN heads), deeper blocks take tokens (linear heads).

    Args:
        num_patches: Sequence length N.
        config:      QuantumConfig.
        patch_size:  Raw-patch side length (block 1 only).
        in_dim:      Incoming token width (blocks >= 2 only).
    """

    def __init__(
        self,
        num_patches: int,
        config: QuantumConfig,
        *,
        patch_size: int | None = None,
        in_dim: int | None = None,
    ) -> None:
        super().__init__()
        if (patch_size is None) == (in_dim is None):
            raise ValueError("Give exactly one of patch_size / in_dim")
        self.num_heads = config.num_heads
        self.readout_total_dim = _readout_total_dim(config._observable_plan)
        if patch_size is not None:
            self.heads = nn.ModuleList([
                Seq2SeqCNNHead(patch_size, num_patches, config)
                for _ in range(config.num_heads)
            ])
        else:
            self.heads = nn.ModuleList([
                Seq2SeqLinearHead(num_patches, in_dim, config)
                for _ in range(config.num_heads)
            ])

    def forward(self, features: torch.Tensor):
        """(B, N, in_dim) → 6-tuple from ``_run_seq2seq_heads_vmap``."""
        return _run_seq2seq_heads_vmap(
            self.heads, self.num_heads, self.readout_total_dim, features
        )

    def gate_params_grid(self, features: torch.Tensor) -> list[torch.Tensor]:
        """Per-head gate-parameter samples over a batch (diagnostics accessor).

        Args:
            features: Real tensor of shape (B, N, in_dim).

        Returns:
            list of ``num_heads`` tensors, each (B, N, 2 × gate_param_width) —
            key slice first, query slice second.
        """
        return [
            h.hypernetwork.forward_grid(features)
            if isinstance(h, Seq2SeqCNNHead)
            else h._features_to_params(features)
            for h in self.heads
        ]


# ---------------------------------------------------------------------------
# StackedCVQuixer
# ---------------------------------------------------------------------------


class StackedCVQuixer(BaseVisionTransformer):
    """Stacked seq-to-seq CV-Quixer (model="quantum_stacked").

    ``num_seq2seq_blocks`` uniform seq-to-seq blocks (identity residual from
    block 2 onward when ``block_residual``), then mean-pooling over positions
    (``pooling="mean"``) or a canonical seq-to-one aggregator block
    (``pooling="quixer"``), then the same (H×R)-input CVDecoder as the
    canonical models.

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

        config = _resolve_decoder_hidden(quantum_config)
        self.config = config

        per_head_dim = _readout_total_dim(config._observable_plan)
        token_dim = config.num_heads * per_head_dim

        blocks = [Seq2SeqCVAttention(num_patches, config, patch_size=patch_size)]
        for _ in range(config.num_seq2seq_blocks - 1):
            blocks.append(Seq2SeqCVAttention(num_patches, config, in_dim=token_dim))
        self.blocks = nn.ModuleList(blocks)
        self.block_residual = config.block_residual
        self.pooling = config.pooling
        # Aggregator block (pooling="quixer"): a canonical seq-to-one head per
        # head — vacuum input, no query slice — consuming the final token
        # sequence via per-head linears (LinearCVHead). Not counted by
        # num_seq2seq_blocks (it is seq-to-one, ADR-0003).
        if config.pooling == "quixer":
            self.aggregator_heads = nn.ModuleList([
                LinearCVHead(num_patches, token_dim, config)
                for _ in range(config.num_heads)
            ])
        else:
            self.aggregator_heads = None

        self.decoder = CVDecoder(
            in_dim=token_dim,
            hidden_dim=config.decoder_hidden_dim,
            num_classes=data_config.num_classes,
            num_layers=config.decoder_num_layers,
        )

        self.trunc_penalty = config.trunc_penalty
        self.trunc_lambda = config.trunc_lambda
        self.cvqnn_trunc_lambda = config.cvqnn_trunc_lambda
        self.query_trunc_lambda = config.query_trunc_lambda

    def forward(
        self,
        patches: torch.Tensor,
        return_trunc_loss: bool = False,
        return_readouts: bool = False,
        return_states: bool = False,
        return_success_prob: bool = False,
    ) -> torch.Tensor | CVQuixerOut:
        """Run the stacked model.

        Args:
            patches:           Tensor of shape (B, N, patch_dim).
            return_trunc_loss: If True, also return the three truncation
                               streams (flat means over blocks × heads × batch
                               × positions, ADR-0003).
            return_readouts:   If True, also return the pre-decoder readout
                               vector ``(B, num_heads × R)`` (the pooled tokens
                               or the aggregator readouts).
            return_states:     If True, also return ``(states, success_probs)``
                               for the **decoder-input stage** (ADR-0003): the
                               aggregator's per-head ``(B, D, ..., D)`` states
                               under pooling="quixer", or the last seq-to-seq
                               block's per-head ``(B, N, D, ..., D)`` states
                               under pooling="mean" (success_probs ``(B,)`` /
                               ``(B, N)`` accordingly). Diagnostic-only.
            return_success_prob: If True, also return ``success_probs`` alone —
                               the decoder-input stage's raw per-head
                               post-selection norms ``‖P(M)|ψ⟩‖²`` (same
                               definition as under ``return_states``; per-head
                               shape is ``(B,)`` for pooling="quixer" or
                               ``(B, N)`` for pooling="mean", unlike the
                               canonical models' ``(B,)``).

        Returns:
            With no return_* flag set: the plain ``logits`` tensor of shape
            ``(B, num_classes)``. Otherwise a ``CVQuixerOut`` namedtuple with
            the requested fields populated (``trunc_loss`` is None when
            trunc_penalty == "none").
        """
        tokens: torch.Tensor | None = None
        patch_truncs: list[torch.Tensor] = []
        query_truncs: list[torch.Tensor] = []
        w_truncs: list[torch.Tensor] = []
        stage_states: list[torch.Tensor] = []
        stage_sps: list[torch.Tensor] = []
        for i, block in enumerate(self.blocks):
            out, states, sps, pt, qt, wt = block(
                patches if i == 0 else tokens
            )
            patch_truncs.append(pt)
            query_truncs.append(qt)
            w_truncs.append(wt)
            if i == len(self.blocks) - 1:
                stage_states, stage_sps = states, sps
            if i > 0 and self.block_residual:
                tokens = tokens + out
            else:
                tokens = out

        if self.pooling == "quixer":
            # Aggregator: canonical seq-to-one heads on the final tokens. Its
            # patch-trunc and W-trunc streams join the block means; it has no
            # query stream (vacuum input, ADR-0003).
            # The aggregator's per-head debug stats are dropped: the stacked
            # model's NaN-forensics stream is out of scope (canonical models
            # only) and CVQuixerOut's debug fields default to None.
            pooled, agg_states, agg_sps, agg_pt, agg_wt, _agg_debug = (
                _run_heads_vmap(
                    self.aggregator_heads, self.config.num_heads,
                    len(self.config._observable_plan), tokens,
                )
            )
            patch_truncs.append(agg_pt)
            w_truncs.append(agg_wt)
            stage_states, stage_sps = agg_states, agg_sps
        else:
            pooled = tokens.mean(dim=1)        # (B, H×R)
        logits = self.decoder(pooled)

        if not (return_trunc_loss or return_readouts or return_states
                or return_success_prob):
            return logits

        # Flat means over blocks (each block scalar is already the mean over
        # its heads × batch × positions).
        trunc_loss = torch.stack(patch_truncs).mean()
        query_trunc_loss = torch.stack(query_truncs).mean()
        cvqnn_trunc_loss = torch.stack(w_truncs).mean()
        return CVQuixerOut(
            logits=logits,
            trunc_loss=(
                trunc_loss.to(logits.device)
                if return_trunc_loss and self.trunc_penalty != "none"
                else None
            ),
            cvqnn_trunc_loss=(
                cvqnn_trunc_loss.to(logits.device) if return_trunc_loss else None
            ),
            query_trunc_loss=(
                query_trunc_loss.to(logits.device) if return_trunc_loss else None
            ),
            readouts=pooled if return_readouts else None,
            states=stage_states if return_states else None,
            success_probs=(
                stage_sps if (return_states or return_success_prob) else None
            ),
        )

    @torch.no_grad()
    def block_inputs(self, patches: torch.Tensor) -> list[torch.Tensor]:
        """The input token sequence of every stage (diagnostics accessor).

        Entry 0 is the raw patches (block 1's input); entry i is block i+1's
        input; with pooling="quixer" a final entry carries the aggregator's
        input. Used with each stage's ``gate_params_grid`` to sample the
        per-block gate parameters.

        Args:
            patches: Tensor of shape (B, N, patch_dim).

        Returns:
            list of ``num_seq2seq_blocks`` (+1 with the aggregator) tensors.
        """
        inputs = [patches]
        tokens: torch.Tensor | None = None
        for i, block in enumerate(self.blocks):
            out, _states, _sps, _pt, _qt, _wt = block(
                patches if i == 0 else tokens
            )
            if i > 0 and self.block_residual:
                tokens = tokens + out
            else:
                tokens = out
            if i < len(self.blocks) - 1 or self.pooling == "quixer":
                inputs.append(tokens)
        return inputs

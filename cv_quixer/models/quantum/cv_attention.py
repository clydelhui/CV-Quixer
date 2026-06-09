"""CV Quantum Attention — iterative LCU + polynomial implementation.

Each attention head processes a patch sequence in three stages:

  1. Per-patch gate application U_i|v⟩: the hypernetwork maps embedded patch i
     to gate parameters (Killoran-style sequence: S → BS → R → D → K), applied
     directly to the state tensor via circuit.apply_single/two_mode_gate.
     No D^m × D^m unitary matrix is ever assembled.

  2. LCU application M|v⟩ = Σ_i b_i (U_i|v⟩): trainable complex scalars b_i
     (LCUSumCoefficients) weight each patch's contribution. Computed iteratively
     over patches; no LCU matrix is materialised.

  3. Polynomial P(M)|ψ_in⟩ = Σ_j c_j M^j|ψ_in⟩: real trainable scalars c_j
     (PolynomialCoefficients). Each M^j|ψ⟩ derived from M^{j-1}|ψ⟩ by one LCU
     application. Models post-selected QSVT — success probability ‖P(M)|ψ_in⟩‖²
     is returned for later use.

Cost: O(degree × N × (m + n_bs) × D^{m+2}) — no D^{3m} matrix powers.
Gradients flow through the einsum chain via standard autograd (loop unrolled
at runtime, analogous to RNN backprop).

Multiple heads run in parallel (HyperCVAttention) via an outer head-axis
vmap that nests over the batch and patch vmaps; readouts concatenated and
passed to CVDecoder in cv_quixer.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
from torch.func import functional_call, vmap

from cv_quixer.config.schema import QuantumConfig
from cv_quixer.quantum import (
    CVCircuit,
    FockState,
    beamsplitter_matrix,
    displacement_matrix,
    kerr_phases,
    rotation_phases,
    squeezing_matrix,
)


# Post-selection floor. When ‖P(M)|ψ⟩‖² drops at/below this, the QSVT
# post-selection is treated as failed: the renormalised state is forced to
# exactly zero so the failing batch element contributes a zero (not exploded)
# gradient. Shared by the renormalisation and the diagnostic warning so the
# two thresholds can never drift apart.
_SUCCESS_PROB_FLOOR = 1e-6


# Std of the small-Gaussian init for the CVQNN block W's gate params. Small
# enough that W ≈ I at start, but non-zero so the displacement/beamsplitter gate
# gradients stay finite (their analytic Fock formulas are NaN-singular at exactly
# zero) and W-symmetry across heads is broken.
_CVQNN_INIT_STD = 1e-2


# ---------------------------------------------------------------------------
# Gate-set helpers
# ---------------------------------------------------------------------------


def _readout_total_dim(observable_plan: list) -> int:
    """Per-head readout width = number of scalars in the normalised plan."""
    return len(observable_plan)


# ---------------------------------------------------------------------------
# Declarative gate-op list
#
# `_GATE_SEQUENCE` is the single source of truth for one layer of the per-patch
# unitary U_i. For num_layers > 1, `_build_op_plan` interleaves L copies of it
# with L-1 `_INTERFEROMETER_SEQUENCE` blocks into the head's `_op_plan`. Both the
# hypernetwork output width (_op_plan_param_count) and the slicing/dispatch logic
# in _apply_patch_gates_to_state derive from that one plan, so adding or removing
# a gate is a one-line edit and the two pieces of code cannot drift.
#
# Layout convention (parameter-name-major): for each op with k param names
# and N sites, the hypernetwork output contains N values for param_names[0],
# then N values for param_names[1], etc. This matches the original manual
# layout and preserves checkpoint compatibility.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GateOp:
    """One gate in the per-patch unitary U_i.

    Attributes:
        name:        Short identifier (also used to build slice-name keys
                     in diagnostic snapshots, e.g. f"{name}_{param_name}").
        param_names: Names of the real scalars consumed per site, in order
                     (e.g. ("r", "phi") for squeeze).
        site_kind:   "mode" — gate acts on each of the m modes; or "pair"
                     — gate acts on each beamsplitter pair.
        apply:       (circuit, state, slc, site, D, device, dtype) -> FockState
                     where slc is a dict {param_name: scalar_tensor}.
    """

    name: str
    param_names: tuple[str, ...]
    site_kind: str
    apply: Callable


def _apply_squeeze(circuit, state, slc, k, D, device, dtype):
    S = squeezing_matrix(slc["r"], slc["phi"], D).to(device=device, dtype=dtype)
    return circuit.apply_single_mode_gate(S, k, state)


def _apply_bs(circuit, state, slc, pair, D, device, dtype):
    a, b = pair
    BS = beamsplitter_matrix(slc["theta"], slc["phi"], D).to(device=device, dtype=dtype)
    return circuit.apply_two_mode_gate(BS, a, b, state)


def _apply_rot(circuit, state, slc, k, D, device, dtype):
    phases = rotation_phases(slc["phi"], D).to(device=device, dtype=dtype)
    return circuit.apply_single_mode_phases(phases, k, state)


def _apply_disp(circuit, state, slc, k, D, device, dtype):
    alpha = torch.complex(slc["re"], slc["im"])
    Dk = displacement_matrix(alpha, D).to(device=device, dtype=dtype)
    return circuit.apply_single_mode_gate(Dk, k, state)


def _apply_kerr(circuit, state, slc, k, D, device, dtype):
    phases = kerr_phases(slc["kappa"], D).to(device=device, dtype=dtype)
    return circuit.apply_single_mode_phases(phases, k, state)


_GATE_SEQUENCE: tuple[GateOp, ...] = (
    GateOp("squeeze", ("r", "phi"),     "mode", _apply_squeeze),
    GateOp("bs",      ("theta", "phi"), "pair", _apply_bs),
    GateOp("rot",     ("phi",),         "mode", _apply_rot),
    GateOp("disp",    ("re", "im"),     "mode", _apply_disp),
    GateOp("kerr",    ("kappa",),       "mode", _apply_kerr),
)

# Gate params whose MAGNITUDE can overflow the analytic Fock matrices in
# complex64 (squeeze r → cosh/sinh r; displacement re/im → exp(-|α|²/2)·αⁿ → 0·∞).
# When QuantumConfig.gate_param_bound is set, these are soft-clipped to (-b, b);
# angles (phi, theta) are periodic and Kerr kappa is a diagonal phase, so neither
# overflows and both are left untouched. See `_apply_patch_gates_to_state`.
_BOUNDED_GATE_PARAMS: frozenset[str] = frozenset({"r", "re", "im"})


# Interferometer inserted *between* consecutive layers when num_layers > 1: a
# beamsplitter column + per-mode rotation (the same `bs`/`rot` ops the in-layer
# sequence uses — reused here, no new gate code). Layer 1 has no leading
# interferometer because it would act trivially on the vacuum input state.
_INTERFEROMETER_SEQUENCE: tuple[GateOp, ...] = (
    _GATE_SEQUENCE[1],  # bs
    _GATE_SEQUENCE[2],  # rot
)


def _build_op_plan(num_layers: int) -> tuple[GateOp, ...]:
    """Ordered gate-op plan for the depth-``num_layers`` per-patch unitary U_i.

    Interleaves ``num_layers`` copies of ``_GATE_SEQUENCE`` (one full Killoran
    layer each) with ``num_layers - 1`` ``_INTERFEROMETER_SEQUENCE`` blocks:

        layer_1, interferometer_1, layer_2, interferometer_2, ..., layer_L

    At ``num_layers == 1`` this returns exactly ``_GATE_SEQUENCE`` (byte-for-byte
    identical behaviour to the single-layer model), so existing checkpoints load
    unchanged. The plan is the single source of truth for both the hypernetwork
    output width and the slicing/dispatch in ``_apply_patch_gates_to_state``.

    Args:
        num_layers: Per-patch circuit depth L (number of stacked gate sequences).

    Returns:
        Flat tuple of GateOps in application order.
    """
    if num_layers < 1:
        raise ValueError(f"num_layers must be >= 1, got {num_layers}")
    plan: list[GateOp] = []
    for layer in range(num_layers):
        plan.extend(_GATE_SEQUENCE)
        if layer < num_layers - 1:
            plan.extend(_INTERFEROMETER_SEQUENCE)
    return tuple(plan)


# One layer of the CVQNN block W (model="quantum"/"quantum_shared", added after
# the polynomial, before readout). The *canonical* two-interferometer Killoran
# form `(BS→R) → S → (BS→R) → D → K`: the per-patch `_GATE_SEQUENCE` with the
# leading interferometer restored. `U_i` drops that leading interferometer
# because it acts trivially on the vacuum `U_i` first sees; `W` acts on the
# non-vacuum post-polynomial state, so it is included. Reuses existing GateOps.
_CVQNN_LAYER: tuple[GateOp, ...] = _INTERFEROMETER_SEQUENCE + _GATE_SEQUENCE


def _build_cvqnn_plan(num_cvqnn_layers: int) -> tuple[GateOp, ...]:
    """Ordered gate-op plan for the depth-``num_cvqnn_layers`` CVQNN block W.

    Stacks ``num_cvqnn_layers`` copies of ``_CVQNN_LAYER`` (each a full canonical
    Killoran layer). Returns the empty tuple at ``num_cvqnn_layers == 0`` — the W
    block is then absent and the head's state_dict is byte-identical to a pre-W
    model. The plan drives both the W parameter-vector width
    (``_op_plan_param_count``) and the gate dispatch in ``_apply_gate_plan``.

    Args:
        num_cvqnn_layers: CVQNN block depth L_W (>= 0).

    Returns:
        Flat tuple of GateOps in application order (empty when L_W == 0).
    """
    if num_cvqnn_layers < 0:
        raise ValueError(
            f"num_cvqnn_layers must be >= 0, got {num_cvqnn_layers}"
        )
    return _CVQNN_LAYER * num_cvqnn_layers


def _bs_pair_count(num_modes: int, bs_topology: str) -> int:
    """Number of beamsplitter pairs given the mode count and topology."""
    if num_modes <= 1:
        return 0
    return num_modes if bs_topology == "ring" else num_modes - 1


def _gate_param_count(num_modes: int, bs_topology: str) -> int:
    """Total hypernetwork output size for the configured gate set and topology.

    Derived from `_GATE_SEQUENCE`: sums `len(op.param_names) * n_sites` over
    each op, where n_sites is `num_modes` for "mode" gates and the number of
    beamsplitter pairs for "pair" gates.
    """
    m = num_modes
    n_bs = _bs_pair_count(m, bs_topology)
    return sum(
        len(op.param_names) * (m if op.site_kind == "mode" else n_bs)
        for op in _GATE_SEQUENCE
    )


def _op_plan_param_count(
    op_plan: tuple[GateOp, ...], num_modes: int, bs_topology: str
) -> int:
    """Total hypernetwork output size for an arbitrary op-plan.

    Generalises ``_gate_param_count`` (which is the ``num_layers == 1`` special
    case, ``op_plan == _GATE_SEQUENCE``) to the interleaved multi-layer plan
    built by ``_build_op_plan``. Sums ``len(op.param_names) * n_sites`` over the
    plan, where ``n_sites`` is ``num_modes`` for "mode" gates and the
    beamsplitter-pair count for "pair" gates.
    """
    m = num_modes
    n_bs = _bs_pair_count(m, bs_topology)
    return sum(
        len(op.param_names) * (m if op.site_kind == "mode" else n_bs)
        for op in op_plan
    )


# ---------------------------------------------------------------------------
# Truncation loss helpers
# ---------------------------------------------------------------------------


def norm_truncation_penalty(state: FockState) -> torch.Tensor:
    """1 - ‖ψ‖² — leakage outside the Fock cutoff.

    Zero when the state is exactly normalised. Positive values indicate
    photon population at or beyond cutoff_dim. Minimising this drives the
    circuit toward states that fit within the chosen cutoff.

    Note: FockState.norm() returns ‖ψ‖² (the squared norm), so the penalty
    is simply 1 - state.norm().

    Args:
        state: Current FockState.

    Returns:
        Scalar tensor in [0, 1].
    """
    return 1.0 - state.norm()


def photon_number_penalty(state: FockState, circuit: CVCircuit) -> torch.Tensor:
    """Mean ⟨n̂_i⟩ / (cutoff_dim - 1) averaged over modes.

    Penalises circuits that drive photon occupancy toward the truncation
    boundary. Normalised so that the penalty is 1 when every mode sits at
    the highest Fock level (cutoff_dim - 1).

    Args:
        state:   Current FockState.
        circuit: CVCircuit owning cutoff_dim (used for normalisation only).

    Returns:
        Scalar tensor in [0, 1].
    """
    D = state.data.shape[0]
    mean_n = sum(
        circuit.measure_photon_number(i, state) for i in range(state.num_modes)
    )
    return mean_n / (state.num_modes * (D - 1))


# ---------------------------------------------------------------------------
# 2D sinusoidal positional encoding helper
# ---------------------------------------------------------------------------


def _compute_2d_sinusoidal_pe(num_patches: int, feature_dim: int) -> torch.Tensor:
    """Precompute additive 2D sinusoidal positional encodings for a patch grid.

    The patch grid is assumed square (grid_size × grid_size = num_patches).
    The feature_dim is split in half: first half encodes row position, second
    half encodes column position, using alternating sin/cos pairs.

    Args:
        num_patches: Total number of patches (must be a perfect square).
        feature_dim: Dimensionality of the CNN feature vector (must be even).

    Returns:
        Float tensor of shape (num_patches, feature_dim).
    """
    grid = int(num_patches ** 0.5)
    assert grid * grid == num_patches, (
        f"num_patches={num_patches} must be a perfect square for 2D PE"
    )
    assert feature_dim % 2 == 0, (
        f"feature_dim={feature_dim} must be even for 2D sinusoidal PE"
    )
    d_half = feature_dim // 2
    pe = torch.zeros(num_patches, feature_dim)
    for idx in range(num_patches):
        row, col = divmod(idx, grid)
        for k in range(d_half // 2):
            div = 10000.0 ** (2 * k / max(d_half, 1))
            pe[idx, 2 * k]     = math.sin(row / div)
            pe[idx, 2 * k + 1] = math.cos(row / div)
        for k in range(d_half // 2):
            div = 10000.0 ** (2 * k / max(d_half, 1))
            pe[idx, d_half + 2 * k]     = math.sin(col / div)
            pe[idx, d_half + 2 * k + 1] = math.cos(col / div)
    return pe


# ---------------------------------------------------------------------------
# CNNHypernetwork
# ---------------------------------------------------------------------------


class CNNHypernetwork(nn.Module):
    """CNN that maps one raw image patch to quantum gate parameters.

    Architecture:
        Conv2d(1, C1, k) → Tanh → Conv2d(C1, C2, k) → Tanh →
        flatten → + 2D sinusoidal PE → Linear(feature_dim, gate_params)

    No padding is used; spatial output size h_out = patch_size - 2*(k-1).
    The 2D PE encodes each patch's (row, col) grid position and is added to
    the flattened feature vector before the final linear projection.

    Args:
        patch_size:    Side length of each square patch in pixels.
        num_patches:   Total number of patches (must be a perfect square).
        num_modes:     Number of bosonic modes.
        cnn_channels_1: Output channels of first conv layer.
        cnn_channels_2: Output channels of second conv layer.
        cnn_kernel_size: Kernel size for both conv layers.
        bs_topology:   "linear" | "ring".
        num_layers:    Per-patch circuit depth L. The output width is the full
                       op-plan size (L gate sequences + L-1 BS->Rot
                       interferometers); L=1 reproduces the single-layer width.
    """

    def __init__(
        self,
        patch_size: int,
        num_patches: int,
        num_modes: int,
        cnn_channels_1: int,
        cnn_channels_2: int,
        cnn_kernel_size: int,
        bs_topology: str,
        num_layers: int = 1,
        cnn_num_conv_layers: int = 2,
        hypernet_num_linear_layers: int = 1,
    ) -> None:
        super().__init__()
        if cnn_num_conv_layers < 2:
            raise ValueError(
                f"cnn_num_conv_layers must be >= 2, got {cnn_num_conv_layers}"
            )
        if hypernet_num_linear_layers < 1:
            raise ValueError(
                f"hypernet_num_linear_layers must be >= 1, got "
                f"{hypernet_num_linear_layers}"
            )
        h_out = patch_size - 2 * (cnn_kernel_size - 1)
        assert h_out > 0, (
            f"patch_size={patch_size} is too small for cnn_kernel_size={cnn_kernel_size}; "
            f"need patch_size > {2 * (cnn_kernel_size - 1)}"
        )
        self.patch_size = patch_size
        feature_dim = cnn_channels_2 * h_out * h_out
        gate_params = _op_plan_param_count(
            _build_op_plan(num_layers), num_modes, bs_topology
        )

        self.conv1 = nn.Conv2d(1, cnn_channels_1, cnn_kernel_size)
        self.conv2 = nn.Conv2d(cnn_channels_1, cnn_channels_2, cnn_kernel_size)
        # Extra depth: same-padding C2→C2 convs preserve h_out / feature_dim, so
        # the PE buffer and final-linear width are unchanged. Empty at the default
        # of 2 ⇒ no new state-dict keys (checkpoint-compatible).
        self.extra_convs = nn.ModuleList(
            nn.Conv2d(cnn_channels_2, cnn_channels_2, cnn_kernel_size, padding="same")
            for _ in range(cnn_num_conv_layers - 2)
        )
        # Extra depth: feature_dim→feature_dim Tanh blocks before the final
        # projection. Empty at the default of 1 ⇒ no new keys.
        self.hidden_linears = nn.ModuleList(
            nn.Linear(feature_dim, feature_dim)
            for _ in range(hypernet_num_linear_layers - 1)
        )
        self.linear = nn.Linear(feature_dim, gate_params)

        pe = _compute_2d_sinusoidal_pe(num_patches, feature_dim)
        self.register_buffer('pos_enc', pe)

    def _conv_features(self, x: torch.Tensor) -> torch.Tensor:
        """Run the conv stack on a 4-D (M, 1, ps, ps) tensor → (M, C2, h, h)."""
        x = torch.tanh(self.conv1(x))
        x = torch.tanh(self.conv2(x))
        for conv in self.extra_convs:
            x = torch.tanh(conv(x))
        return x

    def _project(self, features: torch.Tensor) -> torch.Tensor:
        """Map flattened (+PE) features (..., feature_dim) → (..., gate_params)."""
        for lin in self.hidden_linears:
            features = torch.tanh(lin(features))
        return self.linear(features)

    def forward(self, patch: torch.Tensor, patch_idx: int) -> torch.Tensor:
        """Map one raw patch to gate parameters.

        Args:
            patch:     1-D tensor of shape (patch_size²,) — flattened greyscale patch.
            patch_idx: Position index of this patch in the sequence (0-indexed).

        Returns:
            Tensor of shape (op-plan param count,) — the full depth-L gate-param
            vector (= _gate_param_count(num_modes, bs_topology) when num_layers==1).
        """
        x = patch.reshape(1, 1, self.patch_size, self.patch_size)
        x = self._conv_features(x)
        x = x.flatten(start_dim=1).squeeze(0) + self.pos_enc[patch_idx]
        return self._project(x)

    def forward_all(self, patches: torch.Tensor) -> torch.Tensor:
        """Map a full patch sequence to gate parameters in one batched call.

        Equivalent to stacking ``forward(patches[i], i)`` for i in 0..N-1, but
        runs the conv stack and final linear in a single batched call instead
        of N Python iterations.

        Args:
            patches: Real tensor of shape (N, patch_size²) — flattened patches
                     in positional order (row-major over the patch grid).

        Returns:
            Tensor of shape (N, op-plan param count) — full depth-L gate params.
        """
        N = patches.shape[0]
        x = patches.reshape(N, 1, self.patch_size, self.patch_size)
        x = self._conv_features(x)
        x = x.flatten(start_dim=1) + self.pos_enc   # (N, feature_dim)
        return self._project(x)                     # (N, gate_params)

    def forward_grid(self, patches: torch.Tensor) -> torch.Tensor:
        """Map a batch of patch sequences to gate parameters in one call.

        Equivalent to running ``forward(patches[b, i], i)`` for every
        (b, i), but uses a single batched Conv2d pass and broadcasts the
        positional encoding over the batch axis.

        Args:
            patches: Real tensor of shape (B, N, patch_size²).

        Returns:
            Tensor of shape (B, N, op-plan param count) — full depth-L gate params.
        """
        B, N, _ = patches.shape
        x = patches.reshape(B * N, 1, self.patch_size, self.patch_size)
        x = self._conv_features(x)
        # Flatten conv output then add the (N, feature_dim) PE broadcast
        # across the batch axis. The PE is per patch-position, identical
        # for every batch element.
        x = x.flatten(start_dim=1).reshape(B, N, -1) + self.pos_enc
        return self._project(x)                     # (B, N, gate_params)


# ---------------------------------------------------------------------------
# LCUSumCoefficients
# ---------------------------------------------------------------------------


class LCUSumCoefficients(nn.Module):
    """Trainable complex scalar b_i per patch position for M = Σ_i b_i U_i.

    Stored as two separate real parameter vectors (b_real, b_imag) rather than
    a single complex parameter. Standard PyTorch optimisers (Adam, AdamW, SGD)
    are designed for real-valued parameters. While nn.Parameter supports
    complex dtypes, optimiser behaviour on complex tensors is implementation-
    dependent and inconsistent across third-party gradient utilities (gradient
    clipping, schedulers, etc.). Two real parameters are universally safe
    across the full optimisation stack.

    One instance per head; shared across all batch elements (b_i depends only
    on position i, not on patch content).

    Args:
        num_patches: Sequence length N (fixed for a given dataset/patch size).
    """

    def __init__(self, num_patches: int) -> None:
        super().__init__()
        # Init: b_i = 1/N + 0j → M ≈ (1/N) Σ U_i (uniform real average)
        self.b_real = nn.Parameter(torch.full((num_patches,), 1.0 / num_patches))
        self.b_imag = nn.Parameter(torch.zeros(num_patches))

    def forward(self) -> torch.Tensor:
        """Return complex tensor of shape (num_patches,)."""
        return torch.complex(self.b_real, self.b_imag)


# ---------------------------------------------------------------------------
# PolynomialCoefficients
# ---------------------------------------------------------------------------


class PolynomialCoefficients(nn.Module):
    """Trainable real scalars c_j for P(M) = Σ_{j=0}^{d} c_j M^j.

    Real coefficients are guaranteed at construction: torch.zeros returns a
    float tensor and nn.Parameter preserves that dtype. Assertions in both
    __init__ and forward guard against accidental complexification. Real
    coefficients are required for QSVT compatibility, where the polynomial
    applied to singular values must be real-valued.

    Args:
        degree: Polynomial degree d (total d+1 coefficients).
    """

    def __init__(self, degree: int) -> None:
        super().__init__()
        # Init: c_0 = 1, rest = 0 → P(M) = I at start of training
        init = torch.zeros(degree + 1)
        init[0] = 1.0
        assert not init.is_complex(), (
            "PolynomialCoefficients must be initialised with a real tensor; "
            "got complex — check the init code above."
        )
        self.c = nn.Parameter(init)

    def forward(self) -> torch.Tensor:
        """Return real tensor [c_0, c_1, ..., c_d] of shape (d+1,)."""
        assert not self.c.is_complex(), (
            "Polynomial coefficients c_j must be real. "
            "The parameter has been modified to a complex dtype — this is not allowed."
        )
        return self.c


# ---------------------------------------------------------------------------
# HyperCVAttentionHead
# ---------------------------------------------------------------------------


class _CVHeadBase(nn.Module):
    """Shared circuit core for a single CV attention head.

    Holds the LCU + polynomial machinery and the quantum circuit. Subclasses
    supply the per-patch input → gate-parameter map via ``_features_to_params``:

      - ``HyperCVAttentionHead`` — a per-head ``CNNHypernetwork`` over raw patches.
      - ``LinearCVHead``         — a single ``Linear`` over shared patch embeddings.

    Parameters owned here:
        lcu_coeffs:    LCUSumCoefficients (per-position complex b_i)
        poly_coeffs:   PolynomialCoefficients (real c_j)

    CVCircuit is stateless — it holds no nn.Parameters.

    Args:
        num_patches: Sequence length N — needed to size LCUSumCoefficients.
        config:      QuantumConfig.
    """

    def __init__(self, num_patches: int, config: QuantumConfig) -> None:
        super().__init__()
        self.num_modes = config.num_modes
        self.cutoff_dim = config.cutoff_dim
        self._observable_plan = config._observable_plan
        self.torch_dtype = (
            torch.complex128 if config.dtype == "complex128" else torch.complex64
        )
        # Trace-time constants for the truncation-loss path. Branching on these
        # under torch.func.vmap is safe (resolved once at trace time). The real
        # dtype is the avg_trunc_loss slot dtype in *every* forward branch so
        # the vmapped 4-tuple return signature stays invariant.
        self._trunc_enabled: bool = config.trunc_penalty != "none"
        self._real_dtype = (
            torch.float64 if config.dtype == "complex128" else torch.float32
        )

        m = config.num_modes
        if m <= 1:
            self._bs_pairs: list[tuple[int, int]] = []
        elif config.bs_topology == "ring":
            self._bs_pairs = [(k, (k + 1) % m) for k in range(m)]
        else:
            self._bs_pairs = [(k, k + 1) for k in range(m - 1)]

        # Per-patch unitary depth: L stacked gate sequences interleaved with
        # L-1 BS->Rot interferometers. _op_plan is the ordered gate list driving
        # both _apply_patch_gates_to_state and the hypernetwork output width.
        self.num_layers = config.num_layers
        self._op_plan = _build_op_plan(config.num_layers)

        # Optional soft-clip on magnitude gate params (overflow guard). Trace-time
        # constant (plain float / None), so the branch in _apply_patch_gates_to_state
        # resolves once under vmap.
        self._gate_bound = config.gate_param_bound

        self.lcu_coeffs = LCUSumCoefficients(num_patches)
        self.poly_coeffs = PolynomialCoefficients(config.poly_degree)
        self.circuit = CVCircuit(config.num_modes, config.cutoff_dim)

        # CVQNN block W: a fixed, per-image, trainable Killoran circuit applied to
        # the post-polynomial (post-selected) state before readout. Owned params
        # (input-independent). Initialised with small Gaussian noise (W ≈ I, near
        # the identity) — NOT exact zero: the displacement (α=0) and beamsplitter
        # (θ=0) analytic Fock formulas have a NaN-gradient singularity at exactly
        # zero (the off-diagonal complex power exp(s·log α) / sin(θ)**0). Small
        # noise sits off that singularity (like the always-nonzero hypernet params)
        # and breaks W-symmetry across heads. At L_W == 0 the block is absent —
        # `cvqnn_params` stays a plain None attribute (NOT a registered
        # nn.Parameter), so it never appears in named_parameters()/state_dict() and
        # the head is byte-identical to a pre-W model (checkpoint compat).
        self.cvqnn_num_layers = config.cvqnn_num_layers
        self._cvqnn_plan = _build_cvqnn_plan(config.cvqnn_num_layers)
        cvqnn_param_count = _op_plan_param_count(
            self._cvqnn_plan, config.num_modes, config.bs_topology
        )
        if cvqnn_param_count > 0:
            self.cvqnn_params = nn.Parameter(torch.randn(cvqnn_param_count) * _CVQNN_INIT_STD)
        else:
            self.cvqnn_params = None

    # ------------------------------------------------------------------
    # Per-patch input → gate parameters (subclass hook)
    # ------------------------------------------------------------------

    def _features_to_params(self, features: torch.Tensor) -> torch.Tensor:
        """Map a per-patch input sequence to gate parameters.

        Args:
            features: Real tensor of shape (N, in_dim) — one row per patch.
                      ``in_dim`` is the raw patch dimension for the CNN head or
                      the shared embedding dimension for the linear head.

        Returns:
            Tensor of shape (N, _gate_param_count(num_modes, bs_topology)).
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _apply_gate_plan(
        self,
        op_plan: tuple[GateOp, ...],
        params: torch.Tensor,
        state: FockState,
        device: torch.device,
        dtype: torch.dtype,
    ) -> FockState:
        """Apply an arbitrary ordered gate-op plan directly to a FockState.

        Iterates over ``op_plan`` and dispatches through each op's apply
        callback. Parameter slicing follows the parameter-name-major layout: for
        each op, the full vector for each param name is read in order before
        iterating over sites. Uses circuit.apply_single_mode_gate /
        apply_two_mode_gate / apply_single_mode_phases via the per-op adapters —
        no D^m × D^m matrix is ever assembled.

        Shared by the per-patch unitary U_i (``self._op_plan``) and the CVQNN
        block W (``self._cvqnn_plan``); the ``gate_param_bound`` soft-clip applies
        to both for free.

        Args:
            op_plan: Ordered tuple of GateOps to apply.
            params:  Real tensor of shape
                     (_op_plan_param_count(op_plan, num_modes, topology),).
            state:   Input FockState. Not mutated.
            device:  Target device.
            dtype:   Complex dtype.

        Returns:
            New FockState after the plan is applied.
        """
        m = self.num_modes
        D = self.cutoff_dim
        bs_pairs = self._bs_pairs
        idx = 0

        for op in op_plan:
            sites = list(range(m)) if op.site_kind == "mode" else bs_pairs
            n_sites = len(sites)
            param_vectors: dict[str, torch.Tensor] = {}
            for p in op.param_names:
                param_vectors[p] = params[idx:idx + n_sites]
                idx += n_sites
            for site_idx, site in enumerate(sites):
                slc = {p: param_vectors[p][site_idx] for p in op.param_names}
                # Soft-clip magnitude params to (-b, b) so squeeze/displacement
                # can't overflow the Fock matrices: b·tanh(x/b) (≈ x near 0, slope 1
                # at 0). Off when gate_param_bound is None (trace-time constant).
                if self._gate_bound is not None:
                    b = self._gate_bound
                    for p in op.param_names:
                        if p in _BOUNDED_GATE_PARAMS:
                            slc[p] = b * torch.tanh(slc[p] / b)
                state = op.apply(self.circuit, state, slc, site, D, device, dtype)

        return state

    def _apply_patch_gates_to_state(
        self,
        params: torch.Tensor,
        state: FockState,
        device: torch.device,
        dtype: torch.dtype,
    ) -> FockState:
        """Apply the depth-L per-patch unitary U_i directly to a FockState.

        Thin wrapper over ``_apply_gate_plan`` bound to ``self._op_plan`` (the
        ordered op list built by `_build_op_plan` — L stacked `_GATE_SEQUENCE`
        blocks interleaved with L-1 `_INTERFEROMETER_SEQUENCE` blocks; equals
        `_GATE_SEQUENCE` when num_layers == 1).

        Args:
            params: Real tensor of shape
                    (_op_plan_param_count(self._op_plan, num_modes, topology),).
            state:  Input FockState. Not mutated.
            device: Target device.
            dtype:  Complex dtype.

        Returns:
            New FockState after U_i applied.
        """
        return self._apply_gate_plan(self._op_plan, params, state, device, dtype)

    def _apply_patch_gates_to_data(
        self,
        params: torch.Tensor,
        state_data: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Tensor-in / tensor-out wrapper around ``_apply_patch_gates_to_state``.

        ``torch.func.vmap`` cannot trace a function whose return type is the
        custom ``FockState`` dataclass, so the LCU and truncation-loss paths
        go through this helper instead. The wrapped FockState is constructed
        on the fly using static ``num_modes`` / ``cutoff_dim`` (trace-time
        constants under vmap) and immediately unwrapped on return.

        Args:
            params:     Real tensor of shape (_gate_param_count(num_modes, topology),).
            state_data: Complex tensor of shape (D,)*num_modes.
            device:     Target device.
            dtype:      Complex dtype.

        Returns:
            Complex tensor of shape (D,)*num_modes after U_i applied.
        """
        state = FockState(state_data, self.num_modes, self.cutoff_dim)
        out = self._apply_patch_gates_to_state(params, state, device, dtype)
        return out.data

    def _apply_lcu_to_vector(
        self,
        patches: torch.Tensor,
        v: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
        *,
        accumulate_norm_sq: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Compute M|v⟩ = Σ_i b_i (U_i|v⟩) without building M explicitly.

        Args:
            patches: Real tensor of shape (N, embed_dim).
            v:       Complex flat tensor of shape (D^num_modes,). Need not be normalised.
            device:  Target device.
            dtype:   Complex dtype.
            accumulate_norm_sq: If True, also accumulate Σ_i ‖U_i|v⟩‖² over all
                         patches and return it alongside the LCU result. Used by
                         the polynomial's first (vacuum) pass to derive the
                         truncation loss for free (see _apply_polynomial_iterative).
                         This kwarg is a trace-time constant resolved per call
                         site, so the dual return type is vmap-safe.

        Returns:
            If ``accumulate_norm_sq`` is False: complex flat tensor of shape
            (D^num_modes,). If True: a tuple ``(result, norm_sq_sum)`` where
            ``norm_sq_sum`` is a real 0-dim tensor equal to Σ_i ‖U_i|v⟩‖².
        """
        D, m = self.cutoff_dim, self.num_modes
        N = patches.shape[0]
        b = self.lcu_coeffs().to(device)                       # (N,) complex

        # 1. Batched per-patch param net — all N patches in one call.
        all_params = self._features_to_params(patches).to(device)  # (N, gp)

        # 2. Vmap per-patch gate application over N. The shared input vector
        #    `v_data` is broadcast (in_dims=None); device/dtype are trace-time
        #    constants. Nests inside the batch vmap, which itself nests inside
        #    the head vmap in HyperCVAttention (head → batch → patch).
        v_data = v.reshape((D,) * m)
        apply_one = lambda p: self._apply_patch_gates_to_data(
            p, v_data, device, dtype
        )
        out_data_N = vmap(apply_one)(all_params)               # (N, D, ..., D)

        # 3. Reduce to M|v⟩ = Σ_i b_i U_i|v⟩.
        out_flat_N = out_data_N.reshape(N, -1)                 # (N, D^m)
        result = (b.to(dtype).unsqueeze(-1) * out_flat_N).sum(dim=0)  # (D^m,)

        if accumulate_norm_sq:
            norm_sq_sum = (out_flat_N.abs() ** 2).sum()
            return result, norm_sq_sum
        return result

    def _compute_patch_trunc_loss(
        self,
        patches: torch.Tensor,
        v: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Mean norm-based truncation loss across all N patches applied to the vacuum.

        For each patch i, applies the hypernetwork gate sequence U_i to the
        vacuum and computes 1 − ‖U_i|v⟩‖², i.e. the probability that the true
        infinite-dimensional state has amplitude at photon number ≥ cutoff_dim.
        This is non-zero because analytic Fock-basis gate matrices are true
        sub-isometries (column norms ≤ 1).

        Not on the primary hot path: ``forward`` fuses this computation into the
        first LCU pass for poly_degree ≥ 1 (audit M1/M2). This standalone method
        is retained as (a) the isolated oracle for the direct unit tests in
        ``tests/test_cv_quixer.py::TestPatchTruncLoss`` and (b) the
        poly_degree == 0 fallback (no LCU pass exists to fuse into). Do not
        delete without migrating both.

        Args:
            patches: Real tensor of shape (N, embed_dim).
            v:       Complex flat tensor of shape (D^num_modes,) — unit-norm vacuum.
            device:  Target device.
            dtype:   Complex dtype.

        Returns:
            Scalar tensor in [0, 1] — mean truncation loss over i = 0..N-1.
        """
        D, m = self.cutoff_dim, self.num_modes
        N = patches.shape[0]

        all_params = self._features_to_params(patches).to(device)  # (N, gp)
        v_data = v.reshape((D,) * m)
        apply_one = lambda p: self._apply_patch_gates_to_data(
            p, v_data, device, dtype
        )
        out_data_N = vmap(apply_one)(all_params)                # (N, D, ..., D)

        # Mean over patches of 1 − ‖U_i|v⟩‖². Equivalent to
        # 1 − Σ_i ‖U_i|v⟩‖² / N once expanded, matching the LCU fused path
        # in _apply_polynomial_iterative (want_trunc=True).
        norm_sq_per_patch = (out_data_N.reshape(N, -1).abs() ** 2).sum(dim=-1)
        return (1.0 - norm_sq_per_patch).mean()

    def _apply_polynomial_iterative(
        self,
        patches: torch.Tensor,
        state_flat: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
        *,
        want_trunc: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute P(M)|ψ⟩ = Σ_j c_j M^j|ψ⟩ without materialising M or M^j.

        Each M^j|ψ⟩ is derived from M^{j-1}|ψ⟩ via _apply_lcu_to_vector.
        Autograd unrolls the loop and builds the correct computation graph.
        Intermediate states are not renormalised — the polynomial structure
        from QSVT requires exact matrix-polynomial evaluation.

        Args:
            patches:    Real tensor of shape (N, embed_dim).
            state_flat: Complex flat tensor of shape (D^num_modes,) — input statevector.
            device:     Target device.
            dtype:      Complex dtype.
            want_trunc: If True, also return the mean norm-based truncation loss
                        1 − Σ_i ‖U_i|ψ⟩‖² / N. This is computed for free from the
                        *first* LCU pass only (j == 0), where v is still the input
                        vacuum — exactly where the truncation loss is defined.
                        Requires len(poly_coeffs) ≥ 2 so that first pass exists;
                        the caller guarantees this (poly_degree == 0 uses the
                        standalone _compute_patch_trunc_loss fallback instead).

        Returns:
            out_unnorm:   Complex (D^num_modes,) — unnormalised output state.
            success_prob: Scalar — ‖P(M)|ψ⟩‖² (QSVT post-selection probability).
            avg_trunc_loss: Scalar in [0, 1] — only when ``want_trunc`` is True.
        """
        c = self.poly_coeffs().to(device)   # (d+1,) real — move to quantum device
        result = torch.zeros_like(state_flat)
        vacuum_norm_sq = torch.zeros((), device=device, dtype=self._real_dtype)
        v = state_flat
        for j in range(len(c)):
            result = result + c[j].to(dtype) * v
            if j < len(c) - 1:
                if want_trunc and j == 0:
                    # First pass: v is the vacuum. Fuse the trunc summary here
                    # so U_i|0⟩ is computed once, not twice (audit M2).
                    v, vacuum_norm_sq = self._apply_lcu_to_vector(
                        patches, v, device, dtype, accumulate_norm_sq=True
                    )
                else:
                    v = self._apply_lcu_to_vector(patches, v, device, dtype)
        success_prob = (result.abs() ** 2).sum()
        if want_trunc:
            avg_trunc_loss = 1.0 - vacuum_norm_sq / float(patches.shape[0])
            return result, success_prob, avg_trunc_loss
        return result, success_prob

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        patches: torch.Tensor,
        input_state: FockState | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the LCU + polynomial + CVQNN head on one batch element.

        Args:
            patches:     Real tensor of shape (N, embed_dim).
            input_state: Optional FockState to use instead of vacuum. The
                         state must have the same num_modes and cutoff_dim as
                         this head. Defaults to vacuum |0,...,0⟩.

        Returns:
            readout:        Real tensor of shape ``(len(config._observable_plan),)``
                            — one scalar per entry in the expanded observable plan,
                            measured on the final (post-W, renormalised) state.
            output_state:   Tensor — final renormalised output state data (post-W).
            success_prob:   Scalar tensor — ‖P(M)|ψ_in⟩‖² (post-selection
                            probability; 1.0 for a unitary P(M)). Defined on the
                            *polynomial* output, unchanged by W.
            avg_trunc_loss: Scalar tensor in [0, 1] — mean norm-based truncation
                            loss (1 − ‖U_i|ψ⟩‖²) across all patches.
            w_trunc_loss:   Scalar tensor in [0, 1] — the CVQNN block's own
                            truncation leakage 1 − ‖W|ψ⟩‖² (0 when L_W == 0).
        """
        classical_device = patches.device
        # CUDA supports float64/complex128 natively; MPS does not.
        # Quantum circuit (analytic gate matrices, complex128 arithmetic) runs on CPU for MPS.
        quantum_device = (
            classical_device if classical_device.type != "mps"
            else torch.device("cpu")
        )
        D, n = self.cutoff_dim, self.num_modes
        dtype = self.torch_dtype

        # 1. Input state
        if input_state is None:
            state = FockState.vacuum(n, D, quantum_device, dtype)
        else:
            state = input_state
        state_flat = state.data.reshape(-1)   # (D^n,)

        # 2-3. LCU + polynomial (iterative — no matrix materialised), plus the
        # norm-based truncation loss. Both branch variables are trace-time
        # constants so exactly one branch is traced under vmap; avg_trunc_loss
        # is always a real 0-dim tensor (self._real_dtype, quantum_device).
        #   - trunc disabled: skip all per-patch vacuum work entirely (M1).
        #   - poly_degree ≥ 1: fuse the trunc summary into the first (vacuum)
        #     LCU pass so U_i|0⟩ is computed once, not twice (M2).
        #   - poly_degree == 0: no LCU pass exists, so fall back to the
        #     standalone per-patch vacuum pass (preserves prior behaviour).
        if not self._trunc_enabled:
            out_unnorm, success_prob = self._apply_polynomial_iterative(
                patches, state_flat, quantum_device, dtype
            )
            avg_trunc_loss = torch.zeros(
                (), device=quantum_device, dtype=self._real_dtype
            )
        elif len(self.poly_coeffs()) >= 2:
            out_unnorm, success_prob, avg_trunc_loss = self._apply_polynomial_iterative(
                patches, state_flat, quantum_device, dtype, want_trunc=True
            )
        else:
            avg_trunc_loss = self._compute_patch_trunc_loss(
                patches, state_flat, quantum_device, dtype
            )
            out_unnorm, success_prob = self._apply_polynomial_iterative(
                patches, state_flat, quantum_device, dtype
            )

        # 4. Post-selection renormalisation: divide by ‖P(M)|ψ⟩‖ to give a unit-norm state.
        # success_prob = ‖out_unnorm‖²; deviation from 1 is the QSVT post-selection cost.
        # Clamp the *divisor* (not the result) so the division is always finite —
        # this keeps out_scaled's gradient finite even when success_prob is
        # exactly 0 (0/0). torch.where then forces a failed post-selection to an
        # exactly-zero state; because out_scaled is finite, the where-branch
        # gradient is 0·grad = 0 (not 0·NaN), so a near-zero-norm batch element
        # contributes a zero gradient instead of a ~1/√ε explosion.
        safe_sp = success_prob.clamp(min=_SUCCESS_PROB_FLOOR)
        out_scaled = out_unnorm / safe_sp.sqrt()
        out_norm = torch.where(
            success_prob > _SUCCESS_PROB_FLOOR,
            out_scaled,
            torch.zeros_like(out_unnorm),
        )

        # 5. CVQNN block W: apply the fixed per-image Killoran circuit to the
        # heralded (post-selected, unit-norm) state, then renormalise again
        # before readout. W is unitary in exact CV but its truncated
        # squeeze/displace gates are sub-isometries that leak norm, and the
        # observable measurements are raw Tr(ρÔ) with no internal normalisation —
        # so an un-normalised post-W state would scale every readout by the leaked
        # norm (a truncation confound). We capture that leakage (1 − ‖W|ψ⟩‖²) as
        # `w_trunc_loss` *before* renormalising so it can feed the separate CVQNN
        # truncation penalty. `cvqnn_num_layers` is a trace-time constant, so
        # exactly one branch is traced under vmap and the return signature (incl.
        # the real 0-dim `w_trunc_loss`) is invariant. At L_W == 0 this is the
        # identity: out_w == out_norm and w_trunc_loss == 0 (byte-identical to a
        # pre-W model).
        if self.cvqnn_num_layers == 0:
            out_w = out_norm
            w_trunc_loss = torch.zeros(
                (), device=quantum_device, dtype=self._real_dtype
            )
        else:
            pre_w_state = FockState(out_norm.reshape((D,) * n), n, D)
            # Move W params to the quantum device (CPU under MPS, which lacks the
            # float64 the analytic gate matrices use) — mirrors the lcu/poly
            # `.to(quantum_device)` above. vmap-safe (per-head slice).
            cvqnn_params = self.cvqnn_params.to(quantum_device)
            w_state = self._apply_gate_plan(
                self._cvqnn_plan, cvqnn_params, pre_w_state,
                quantum_device, dtype,
            )
            w_flat = w_state.data.reshape(-1)
            w_norm_sq = (w_flat.abs() ** 2).sum()
            w_trunc_loss = 1.0 - w_norm_sq
            # Same clamp-divisor + zero-on-failure guard as the post-selection
            # renorm above: finite gradient even at a (near-)zero-norm element.
            safe_wn = w_norm_sq.clamp(min=_SUCCESS_PROB_FLOOR)
            w_scaled = w_flat / safe_wn.sqrt()
            out_w = torch.where(
                w_norm_sq > _SUCCESS_PROB_FLOOR,
                w_scaled,
                torch.zeros_like(w_flat),
            )

        # 6. Wrap final state as FockState
        output_state = FockState(out_w.reshape((D,) * n), n, D)

        # 7. Measure observable plan in order; move result back to classical device for decoder.
        # Plan order is the source of truth for the readout vector layout —
        # legacy alias translation preserves the pre-refactor ordering so
        # checkpoints remain bit-compatible.
        readout_values: list[torch.Tensor] = []
        for spec in self._observable_plan:
            if spec.type == "x":
                readout_values.append(self.circuit.measure_quadrature_x(spec.mode, output_state))
            elif spec.type == "p":
                readout_values.append(self.circuit.measure_quadrature_p(spec.mode, output_state))
            elif spec.type == "x_squared":
                readout_values.append(self.circuit.measure_quadrature_x_squared(spec.mode, output_state))
            elif spec.type == "p_squared":
                readout_values.append(self.circuit.measure_quadrature_p_squared(spec.mode, output_state))
            elif spec.type == "n":
                readout_values.append(self.circuit.measure_photon_number(spec.mode, output_state))
            elif spec.type == "prob_n":
                readout_values.append(
                    self.circuit.measure_prob_n_photons(spec.mode, spec.n, output_state)
                )
            else:
                raise ValueError(f"Unknown observable type {spec.type!r}")
        readout = torch.stack(readout_values).to(classical_device)
        return readout, output_state.data, success_prob, avg_trunc_loss, w_trunc_loss


# ---------------------------------------------------------------------------
# Concrete heads
# ---------------------------------------------------------------------------


class HyperCVAttentionHead(_CVHeadBase):
    """CV attention head driven by a per-head CNN hypernetwork over raw patches.

    The canonical CV-Quixer head: each head owns a full ``CNNHypernetwork`` that
    maps a raw image patch to gate parameters. Forward input is the raw patch
    sequence ``(N, patch_size²)``.

    Args:
        patch_size:  Side length of each raw image patch in pixels.
        num_patches: Sequence length N — needed to size LCUSumCoefficients and PE.
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
        )

    def _features_to_params(self, features: torch.Tensor) -> torch.Tensor:
        return self.hypernetwork.forward_all(features)


class SharedPatchCNN(nn.Module):
    """Conv feature extractor shared across all heads (model="quantum_shared").

    Runs once per patch and produces a patch embedding consumed by every head's
    per-head ``Linear``. Architecture is ``CNNHypernetwork``'s conv path without
    its final gate-parameter linear — the flattened conv features (+ 2D PE) are
    the embedding directly:

        Conv2d(1, C1, k) → Tanh → Conv2d(C1, C2, k) → Tanh →
        flatten → + 2D sinusoidal PE

    The embedding width is ``out_dim = cnn_channels_2 × h_out²`` (= the canonical
    CNNHypernetwork's final-linear *input* width), so per-head capacity matches
    the canonical model. The 2D positional encoding is added once here, so all
    heads see the same PE-augmented embedding.

    Args:
        patch_size:    Side length of each square patch in pixels.
        num_patches:   Total number of patches (must be a perfect square).
        cnn_channels_1: Output channels of first conv layer.
        cnn_channels_2: Output channels of second conv layer.
        cnn_kernel_size: Kernel size for both conv layers.
    """

    def __init__(
        self,
        patch_size: int,
        num_patches: int,
        cnn_channels_1: int,
        cnn_channels_2: int,
        cnn_kernel_size: int,
        cnn_num_conv_layers: int = 2,
    ) -> None:
        super().__init__()
        if cnn_num_conv_layers < 2:
            raise ValueError(
                f"cnn_num_conv_layers must be >= 2, got {cnn_num_conv_layers}"
            )
        h_out = patch_size - 2 * (cnn_kernel_size - 1)
        assert h_out > 0, (
            f"patch_size={patch_size} is too small for cnn_kernel_size={cnn_kernel_size}; "
            f"need patch_size > {2 * (cnn_kernel_size - 1)}"
        )
        self.patch_size = patch_size
        self.out_dim = cnn_channels_2 * h_out * h_out   # embedding width

        self.conv1 = nn.Conv2d(1, cnn_channels_1, cnn_kernel_size)
        self.conv2 = nn.Conv2d(cnn_channels_1, cnn_channels_2, cnn_kernel_size)
        # Extra depth: same-padding C2→C2 convs preserve h_out / out_dim. Empty at
        # the default of 2 ⇒ no new state-dict keys (checkpoint-compatible).
        self.extra_convs = nn.ModuleList(
            nn.Conv2d(cnn_channels_2, cnn_channels_2, cnn_kernel_size, padding="same")
            for _ in range(cnn_num_conv_layers - 2)
        )

        pe = _compute_2d_sinusoidal_pe(num_patches, self.out_dim)
        self.register_buffer('pos_enc', pe)

    def forward_grid(self, patches: torch.Tensor) -> torch.Tensor:
        """Embed a batch of patch sequences in one batched conv pass.

        Args:
            patches: Real tensor of shape (B, N, patch_size²).

        Returns:
            Tensor of shape (B, N, out_dim).
        """
        B, N, _ = patches.shape
        x = patches.reshape(B * N, 1, self.patch_size, self.patch_size)
        x = torch.tanh(self.conv1(x))
        x = torch.tanh(self.conv2(x))
        for conv in self.extra_convs:
            x = torch.tanh(conv(x))
        return x.flatten(start_dim=1).reshape(B, N, -1) + self.pos_enc  # (B,N,out_dim)


class LinearCVHead(_CVHeadBase):
    """CV attention head driven by a single Linear over shared patch embeddings.

    Used by model="quantum_shared". A ``SharedPatchCNN`` (owned by
    ``SharedCVAttention``) produces patch embeddings once; each head maps them to
    gate parameters with a single ``Linear(in_dim, gate_params)`` — no
    activation. Forward input is the embedding sequence ``(N, in_dim)``.

    Args:
        num_patches: Sequence length N — needed to size LCUSumCoefficients.
        in_dim:      Width of the shared patch embedding (SharedPatchCNN.out_dim).
        config:      QuantumConfig.
    """

    def __init__(
        self, num_patches: int, in_dim: int, config: QuantumConfig
    ) -> None:
        super().__init__(num_patches, config)
        if config.hypernet_num_linear_layers < 1:
            raise ValueError(
                f"hypernet_num_linear_layers must be >= 1, got "
                f"{config.hypernet_num_linear_layers}"
            )
        gate_params = _op_plan_param_count(
            self._op_plan, config.num_modes, config.bs_topology
        )
        # Extra depth: in_dim→in_dim Tanh blocks before the final projection.
        # Empty at the default of 1 ⇒ no new state-dict keys (checkpoint-compatible).
        self.hidden_linears = nn.ModuleList(
            nn.Linear(in_dim, in_dim)
            for _ in range(config.hypernet_num_linear_layers - 1)
        )
        self.linear = nn.Linear(in_dim, gate_params)

    def _features_to_params(self, features: torch.Tensor) -> torch.Tensor:
        for lin in self.hidden_linears:
            features = torch.tanh(lin(features))
        return self.linear(features)


# ---------------------------------------------------------------------------
# Multi-head vmap driver (shared by both attention modules)
# ---------------------------------------------------------------------------


def _run_heads_vmap(
    heads: nn.ModuleList,
    num_heads: int,
    readout_total_dim: int,
    features: torch.Tensor,
) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor], torch.Tensor, torch.Tensor]:
    """Run all heads over a batch of per-patch feature sequences via nested vmap.

    The heads are independent, so their params/buffers are stacked along a new
    leading head axis and vmapped over (head → batch → patch nesting) rather than
    looped in Python. Plain ``torch.stack`` (NOT ``torch.func.stack_module_state``,
    which ends in ``.detach()`` and would strand head gradients) keeps the stacked
    tensors connected to each head's leaf parameter, so gradients flow back through
    StackBackward into the head parameters the optimizer owns.

    Args:
        heads:             ModuleList of homogeneous ``_CVHeadBase`` instances.
        num_heads:         Number of heads (== len(heads)).
        readout_total_dim: Per-head readout width R.
        features:          Real tensor of shape (B, N, in_dim) — raw patches for
                           CNN heads, shared embeddings for linear heads.

    Returns:
        readouts:         (B, num_heads × R).
        states:           list[(B, D, ..., D)] — one per head.
        success_probs:    list[(B,)] — one per head.
        trunc_loss:       scalar — mean per-patch truncation loss over heads × batch.
        cvqnn_trunc_loss: scalar — mean CVQNN-block (W) truncation leakage over
                          heads × batch (0 when cvqnn_num_layers == 0).
    """
    base = heads[0]
    per_head_p = [dict(h.named_parameters()) for h in heads]
    per_head_b = [dict(h.named_buffers()) for h in heads]
    stacked_p = {
        n: torch.stack([d[n] for d in per_head_p]) for n in per_head_p[0]
    }
    stacked_b = {
        n: torch.stack([d[n] for d in per_head_b]) for n in per_head_b[0]
    }

    def _one_element(p, b, single_features):
        return functional_call(base, (p, b), (single_features,))

    def _one_head(p, b):
        # Inner vmap over the batch axis of `features` (closed over, so the outer
        # head vmap treats it as shared — every head sees the full sequence).
        # This itself nests the patch vmap inside _apply_lcu_to_vector:
        # head → batch → patch.
        return vmap(_one_element, in_dims=(None, None, 0))(p, b, features)

    readout_hb, state_hb, sp_hb, tl_hb, wtl_hb = vmap(_one_head, in_dims=(0, 0))(
        stacked_p, stacked_b
    )
    # readout_hb          : (num_heads, B, R)
    # state_hb            : (num_heads, B, cutoff_dim, ..., cutoff_dim)
    # sp_hb, tl_hb, wtl_hb: (num_heads, B)

    if (sp_hb < _SUCCESS_PROB_FLOOR).any().item():
        import warnings
        bad = sp_hb[sp_hb < _SUCCESS_PROB_FLOOR]
        warnings.warn(
            f"{len(bad)} head×batch element(s) have "
            f"success_prob < {_SUCCESS_PROB_FLOOR:.0e} "
            f"(min={bad.min().item():.2e}); post-selection failed — "
            "those elements forced to a zero state (zero gradient).",
            RuntimeWarning, stacklevel=2,
        )

    B = features.shape[0]
    R = readout_total_dim
    # permute(1,0,2).reshape reproduces a torch.cat(dim=-1) per-batch layout
    # [head0_R | head1_R | ...] exactly (pure reshape).
    readouts = readout_hb.permute(1, 0, 2).reshape(B, num_heads * R)
    all_states = list(state_hb.unbind(0))          # list[(B, D,...,D)] per head
    all_success_probs = list(sp_hb.unbind(0))      # list[(B,)] per head
    trunc_loss = tl_hb.mean()                       # scalar over heads × batch
    cvqnn_trunc_loss = wtl_hb.mean()                # scalar over heads × batch
    return readouts, all_states, all_success_probs, trunc_loss, cvqnn_trunc_loss


# ---------------------------------------------------------------------------
# HyperCVAttention
# ---------------------------------------------------------------------------


class HyperCVAttention(nn.Module):
    """Multi-head CV attention with per-head LCU + polynomial circuits.

    Runs num_heads independent HyperCVAttentionHead instances. The heads are
    fully independent, so forward stacks their parameters along a new leading
    head axis and vmaps over it (head → batch → patch nesting) rather than
    looping in Python. Each head processes the full patch sequence and produces
    a (num_modes,) readout. Readouts from all heads are concatenated into a
    (num_heads × num_modes,) vector per batch element.

    Args:
        patch_size:  Side length of each raw image patch in pixels.
        num_patches: Sequence length N.
        config:      QuantumConfig.
    """

    def __init__(
        self, patch_size: int, num_patches: int, config: QuantumConfig
    ) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.num_modes = config.num_modes
        self.cutoff_dim = config.cutoff_dim
        self._observable_plan = config._observable_plan
        self.readout_total_dim = _readout_total_dim(config._observable_plan)

        self.heads = nn.ModuleList([
            HyperCVAttentionHead(patch_size, num_patches, config)
            for _ in range(config.num_heads)
        ])

    def forward(
        self,
        patches: torch.Tensor,
        input_state: FockState | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor], torch.Tensor, torch.Tensor]:
        """Apply all heads to a batch of patch sequences.

        Args:
            patches:     Real tensor of shape (B, N, patch_dim).
            input_state: Not supported under vmap; must be None.

        Returns:
            readouts:         Float tensor of shape
                              (B, num_heads × len(config._observable_plan)).
            states:           list[Tensor] — one (B, D, ..., D) state tensor per head.
            success_probs:    list[Tensor] — one (B,) success probability tensor per head.
            trunc_loss:       Scalar — mean per-patch truncation loss across all heads
                              and batch elements.
            cvqnn_trunc_loss: Scalar — mean CVQNN-block (W) truncation leakage across
                              heads × batch (0 when cvqnn_num_layers == 0).
        """
        if input_state is not None:
            raise NotImplementedError("input_state is not supported under vmap")

        return _run_heads_vmap(
            self.heads, self.num_heads, self.readout_total_dim, patches
        )

    def gate_params_grid(self, patches: torch.Tensor) -> list[torch.Tensor]:
        """Per-head gate-parameter samples over a batch (diagnostics accessor).

        Args:
            patches: Real tensor of shape (B, N, patch_dim).

        Returns:
            list of ``num_heads`` tensors, each (B, N, _gate_param_count).
        """
        return [h.hypernetwork.forward_grid(patches) for h in self.heads]


# ---------------------------------------------------------------------------
# SharedCVAttention
# ---------------------------------------------------------------------------


class SharedCVAttention(nn.Module):
    """Multi-head CV attention with a shared patch CNN + per-head linear heads.

    Powers model="quantum_shared". A single ``SharedPatchCNN`` embeds every patch
    once (shared across heads); each ``LinearCVHead`` then maps the embedding to
    its own gate parameters via a single ``Linear``. The conv stack therefore runs
    once per forward pass instead of once per head. Head execution reuses the same
    head → batch → patch vmap driver as ``HyperCVAttention`` (``_run_heads_vmap``),
    closing over the shared embeddings rather than the raw patches.

    Args:
        patch_size:  Side length of each raw image patch in pixels.
        num_patches: Sequence length N.
        config:      QuantumConfig.
    """

    def __init__(
        self,
        patch_size: int,
        num_patches: int,
        config: QuantumConfig,
    ) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.num_modes = config.num_modes
        self.cutoff_dim = config.cutoff_dim
        self._observable_plan = config._observable_plan
        self.readout_total_dim = _readout_total_dim(config._observable_plan)

        self.patch_cnn = SharedPatchCNN(
            patch_size, num_patches,
            config.cnn_channels_1, config.cnn_channels_2, config.cnn_kernel_size,
            cnn_num_conv_layers=config.cnn_num_conv_layers,
        )
        in_dim = self.patch_cnn.out_dim
        self.heads = nn.ModuleList([
            LinearCVHead(num_patches, in_dim, config)
            for _ in range(config.num_heads)
        ])

    def forward(
        self,
        patches: torch.Tensor,
        input_state: FockState | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor], torch.Tensor, torch.Tensor]:
        """Apply the shared CNN then all heads to a batch of patch sequences.

        Args:
            patches:     Real tensor of shape (B, N, patch_dim).
            input_state: Not supported under vmap; must be None.

        Returns:
            Same 5-tuple as ``HyperCVAttention.forward`` (readouts, states,
            success_probs, trunc_loss, cvqnn_trunc_loss).
        """
        if input_state is not None:
            raise NotImplementedError("input_state is not supported under vmap")

        # Shared conv runs once (outside the head vmap); only the per-head linear
        # is vmapped over the head axis inside _run_heads_vmap.
        embeddings = self.patch_cnn.forward_grid(patches)   # (B, N, embed_dim)
        return _run_heads_vmap(
            self.heads, self.num_heads, self.readout_total_dim, embeddings
        )

    def gate_params_grid(self, patches: torch.Tensor) -> list[torch.Tensor]:
        """Per-head gate-parameter samples over a batch (diagnostics accessor).

        Embeds the patches once via the shared CNN, then applies each head's
        linear projection.

        Args:
            patches: Real tensor of shape (B, N, patch_dim).

        Returns:
            list of ``num_heads`` tensors, each (B, N, _gate_param_count).
        """
        embeddings = self.patch_cnn.forward_grid(patches)   # (B, N, embed_dim)
        return [h.linear(embeddings) for h in self.heads]

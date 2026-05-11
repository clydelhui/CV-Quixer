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

Multiple heads run in parallel (HyperCVAttention); readouts concatenated and
passed to CVDecoder in cv_quixer.py.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from cv_quixer.config.schema import QuantumConfig
from cv_quixer.quantum import (
    CVCircuit,
    FockState,
    beamsplitter_matrix,
    displacement_matrix,
    kerr_matrix,
    rotation_matrix,
    squeezing_matrix,
)


# ---------------------------------------------------------------------------
# Gate-set helpers
# ---------------------------------------------------------------------------


def _gate_param_count(num_modes: int, bs_topology: str) -> int:
    """Total hypernetwork output size for the given gate set and topology.

    Layout: squeeze_r(m) + squeeze_phi(m) + bs_theta(n_bs) + bs_phi(n_bs)
            + rot_phi(m) + disp_re(m) + disp_im(m) + kerr_kappa(m)
    """
    m = num_modes
    if m <= 1:
        n_bs = 0
    elif bs_topology == "ring":
        n_bs = m
    else:
        n_bs = m - 1
    return 2 * m + 2 * n_bs + m + 2 * m + m  # = 6m + 2*n_bs


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
    ) -> None:
        super().__init__()
        h_out = patch_size - 2 * (cnn_kernel_size - 1)
        assert h_out > 0, (
            f"patch_size={patch_size} is too small for cnn_kernel_size={cnn_kernel_size}; "
            f"need patch_size > {2 * (cnn_kernel_size - 1)}"
        )
        self.patch_size = patch_size
        feature_dim = cnn_channels_2 * h_out * h_out
        gate_params = _gate_param_count(num_modes, bs_topology)

        self.conv1 = nn.Conv2d(1, cnn_channels_1, cnn_kernel_size)
        self.conv2 = nn.Conv2d(cnn_channels_1, cnn_channels_2, cnn_kernel_size)
        self.linear = nn.Linear(feature_dim, gate_params)

        pe = _compute_2d_sinusoidal_pe(num_patches, feature_dim)
        self.register_buffer('pos_enc', pe)

    def forward(self, patch: torch.Tensor, patch_idx: int) -> torch.Tensor:
        """Map one raw patch to gate parameters.

        Args:
            patch:     1-D tensor of shape (patch_size²,) — flattened greyscale patch.
            patch_idx: Position index of this patch in the sequence (0-indexed).

        Returns:
            Tensor of shape (_gate_param_count(num_modes, bs_topology),).
        """
        x = patch.reshape(1, 1, self.patch_size, self.patch_size)
        x = torch.tanh(self.conv1(x))
        x = torch.tanh(self.conv2(x))
        x = x.flatten(start_dim=1).squeeze(0) + self.pos_enc[patch_idx]
        return self.linear(x)


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


class HyperCVAttentionHead(nn.Module):
    """Single CV attention head: per-patch unitary assembly → LCU → polynomial.

    Parameters owned by this module:
        hypernetwork:  PatchHypernetwork (patch → gate params)
        lcu_coeffs:    LCUSumCoefficients (per-position complex b_i)
        poly_coeffs:   PolynomialCoefficients (real c_j)

    CVCircuit is stateless — it holds no nn.Parameters.

    Args:
        patch_size:  Side length of each raw image patch in pixels.
        num_patches: Sequence length N — needed to size LCUSumCoefficients and PE.
        config:      QuantumConfig.
    """

    def __init__(
        self, patch_size: int, num_patches: int, config: QuantumConfig
    ) -> None:
        super().__init__()
        self.num_modes = config.num_modes
        self.cutoff_dim = config.cutoff_dim
        self.torch_dtype = (
            torch.complex128 if config.dtype == "complex128" else torch.complex64
        )

        m = config.num_modes
        if m <= 1:
            self._bs_pairs: list[tuple[int, int]] = []
        elif config.bs_topology == "ring":
            self._bs_pairs = [(k, (k + 1) % m) for k in range(m)]
        else:
            self._bs_pairs = [(k, k + 1) for k in range(m - 1)]

        self.hypernetwork = CNNHypernetwork(
            patch_size, num_patches, config.num_modes,
            config.cnn_channels_1, config.cnn_channels_2, config.cnn_kernel_size,
            config.bs_topology,
        )
        self.lcu_coeffs = LCUSumCoefficients(num_patches)
        self.poly_coeffs = PolynomialCoefficients(config.poly_degree)
        self.circuit = CVCircuit(config.num_modes, config.cutoff_dim)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _apply_patch_gates_to_state(
        self,
        params: torch.Tensor,
        state: FockState,
        device: torch.device,
        dtype: torch.dtype,
    ) -> FockState:
        """Apply the full Killoran gate sequence U_i directly to a FockState.

        Gate sequence (squeeze acts on state first, Kerr last):
            S(r_k, φ_k) per mode →
            BS(θ_i, φ_i) on adjacent pairs →
            R(φ_k) per mode →
            D(α_k) per mode →
            K(κ_k) per mode

        Uses circuit.apply_single_mode_gate / apply_two_mode_gate directly —
        no D^m × D^m matrix is ever assembled.

        Args:
            params: Real tensor of shape (_gate_param_count(num_modes, topology),).
            state:  Input FockState. Not mutated.
            device: Target device.
            dtype:  Complex dtype.

        Returns:
            New FockState after U_i applied.
        """
        m = self.num_modes
        D = self.cutoff_dim
        n_bs = len(self._bs_pairs)
        idx = 0

        squeeze_r   = params[idx:idx + m];     idx += m
        squeeze_phi = params[idx:idx + m];     idx += m
        bs_theta    = params[idx:idx + n_bs];  idx += n_bs
        bs_phi      = params[idx:idx + n_bs];  idx += n_bs
        rot_phi     = params[idx:idx + m];     idx += m
        disp_re     = params[idx:idx + m];     idx += m
        disp_im     = params[idx:idx + m];     idx += m
        kerr_kappa  = params[idx:idx + m];     idx += m

        for k in range(m):
            S = squeezing_matrix(squeeze_r[k], squeeze_phi[k], D).to(device=device, dtype=dtype)
            state = self.circuit.apply_single_mode_gate(S, k, state)

        for i, (a, b) in enumerate(self._bs_pairs):
            BS = beamsplitter_matrix(bs_theta[i], bs_phi[i], D).to(device=device, dtype=dtype)
            state = self.circuit.apply_two_mode_gate(BS, a, b, state)

        for k in range(m):
            R = rotation_matrix(rot_phi[k], D).to(device=device, dtype=dtype)
            state = self.circuit.apply_single_mode_gate(R, k, state)

        for k in range(m):
            alpha = torch.complex(disp_re[k], disp_im[k])
            Dk = displacement_matrix(alpha, D).to(device=device, dtype=dtype)
            state = self.circuit.apply_single_mode_gate(Dk, k, state)

        for k in range(m):
            K = kerr_matrix(kerr_kappa[k], D).to(device=device, dtype=dtype)
            state = self.circuit.apply_single_mode_gate(K, k, state)

        return state

    def _apply_lcu_to_vector(
        self,
        patches: torch.Tensor,
        v: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Compute M|v⟩ = Σ_i b_i (U_i|v⟩) without building M explicitly.

        Args:
            patches: Real tensor of shape (N, embed_dim).
            v:       Complex flat tensor of shape (D^num_modes,). Need not be normalised.
            device:  Target device.
            dtype:   Complex dtype.

        Returns:
            Complex flat tensor of shape (D^num_modes,).
        """
        D, m = self.cutoff_dim, self.num_modes
        b = self.lcu_coeffs().to(device)   # (N,) complex — move to quantum device
        result = torch.zeros_like(v)
        for i in range(patches.shape[0]):
            params = self.hypernetwork(patches[i], i).to(device)  # classical → quantum device
            state_i = FockState(v.reshape((D,) * m), m, D)
            out_i = self._apply_patch_gates_to_state(params, state_i, device, dtype)
            result = result + b[i].to(dtype) * out_i.data.reshape(-1)
        return result

    def _apply_polynomial_iterative(
        self,
        patches: torch.Tensor,
        state_flat: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute P(M)|ψ⟩ = Σ_j c_j M^j|ψ⟩ without materialising M or M^j.

        Each M^j|ψ⟩ is derived from M^{j-1}|ψ⟩ via _apply_lcu_to_vector.
        Autograd unrolls the loop and builds the correct computation graph.

        Args:
            patches:    Real tensor of shape (N, embed_dim).
            state_flat: Complex flat tensor of shape (D^num_modes,) — input statevector.
            device:     Target device.
            dtype:      Complex dtype.

        Returns:
            out_unnorm:   Complex (D^num_modes,) — unnormalised output state.
            success_prob: Scalar — ‖P(M)|ψ⟩‖² (post-selection probability).
        """
        c = self.poly_coeffs().to(device)   # (d+1,) real — move to quantum device
        result = torch.zeros_like(state_flat)
        v = state_flat
        for j in range(len(c)):
            result = result + c[j].to(dtype) * v
            if j < len(c) - 1:
                v = self._apply_lcu_to_vector(patches, v, device, dtype)
        success_prob = (result.abs() ** 2).sum()
        return result, success_prob

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        patches: torch.Tensor,
        input_state: FockState | None = None,
    ) -> tuple[torch.Tensor, FockState, torch.Tensor]:
        """Run the LCU + polynomial head on one batch element.

        Args:
            patches:     Real tensor of shape (N, embed_dim).
            input_state: Optional FockState to use instead of vacuum. The
                         state must have the same num_modes and cutoff_dim as
                         this head. Defaults to vacuum |0,...,0⟩.

        Returns:
            readout:      Real tensor of shape (num_modes,) — ⟨x̂_i⟩ for each
                          mode, measured on the normalised post-selected state.
            output_state: FockState — normalised post-selected output.
            success_prob: Scalar tensor — ‖P(M)|ψ_in⟩‖² (post-selection
                          probability; 1.0 for a unitary P(M)).
        """
        classical_device = patches.device
        # CUDA supports float64/complex128 natively; MPS does not.
        # Quantum circuit (gate matrix_exp, complex128 arithmetic) runs on CPU for MPS.
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

        # 2. LCU + polynomial (iterative — no matrix materialised)
        out_unnorm, success_prob = self._apply_polynomial_iterative(
            patches, state_flat, quantum_device, dtype
        )

        # 3. Normalise (post-selection)
        out_norm = out_unnorm / success_prob.sqrt().clamp(min=1e-8)

        # 4. Wrap as FockState
        output_state = FockState(out_norm.reshape((D,) * n), n, D)

        # 5. Measure quadratures; move result back to classical device for decoder
        readout = torch.stack([
            self.circuit.measure_quadrature_x(i, output_state)
            for i in range(n)
        ]).to(classical_device)
        return readout, output_state, success_prob


# ---------------------------------------------------------------------------
# HyperCVAttention
# ---------------------------------------------------------------------------


class HyperCVAttention(nn.Module):
    """Multi-head CV attention with per-head LCU + polynomial circuits.

    Runs num_heads independent HyperCVAttentionHead instances. Each head
    processes the full patch sequence and produces a (num_modes,) readout.
    Readouts from all heads are concatenated into a (num_heads × num_modes,)
    vector per batch element.

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

        self.heads = nn.ModuleList([
            HyperCVAttentionHead(patch_size, num_patches, config)
            for _ in range(config.num_heads)
        ])

    def forward(
        self,
        patches: torch.Tensor,
        input_state: FockState | None = None,
    ) -> tuple[torch.Tensor, list[list[FockState]], list[list[torch.Tensor]]]:
        """Apply all heads to a batch of patch sequences.

        Args:
            patches:     Real tensor of shape (B, N, embed_dim).
            input_state: Optional FockState passed to every head. If None,
                         each head initialises from vacuum independently.

        Returns:
            readouts:      Float tensor of shape (B, num_heads × num_modes).
            states:        states[b][h] — final FockState for batch element b,
                           head h.
            success_probs: success_probs[b][h] — scalar success probability
                           for batch element b, head h.
        """
        B = patches.shape[0]
        all_readouts: list[torch.Tensor] = []
        all_states: list[list[FockState]] = []
        all_success_probs: list[list[torch.Tensor]] = []

        for b in range(B):
            head_readouts: list[torch.Tensor] = []
            head_states: list[FockState] = []
            head_success_probs: list[torch.Tensor] = []

            for head in self.heads:
                readout, state, sp = head(patches[b], input_state)
                head_readouts.append(readout)
                head_states.append(state)
                head_success_probs.append(sp)

            all_readouts.append(torch.cat(head_readouts))
            all_states.append(head_states)
            all_success_probs.append(head_success_probs)

        readouts = torch.stack(all_readouts).to(patches.dtype)
        return readouts, all_states, all_success_probs

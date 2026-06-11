"""Gaussian gate matrix factories — analytic Fock-basis formulas.

Each function returns the exact truncated Fock-basis representation of the gate.
Unlike matrix_exp of the truncated generator, these are true sub-isometries:
column norms satisfy ‖U_col_n‖ ≤ 1, and the deficit 1 − ‖U|ψ⟩‖² equals the
probability of finding the system at photon number ≥ cutoff_dim.

Diagonal gates (Rotation, Kerr) return a (D,) phase vector for use with
CVCircuit.apply_single_mode_phases. Non-diagonal gates return (D, D) or
(D, D, D, D) tensors for use with apply_single_mode_gate / apply_two_mode_gate.

All functions are pure, stateless, and differentiable w.r.t. their parameters
via torch.autograd. All tensor ops are out-of-place for vmap compatibility.

References:
    - Serafini, "Quantum Continuous Variables" (2017)
    - Killoran et al., "Continuous-variable quantum neural networks" (2019)
      https://arxiv.org/abs/1806.06871
"""

from __future__ import annotations

import torch


def rotation_phases(phi: torch.Tensor, cutoff_dim: int) -> torch.Tensor:
    """Returns (D,) phase vector for R(φ): phases[n] = exp(i·n·φ).

    R(φ) is diagonal in the Fock basis: R_{nn} = exp(inφ).
    Rotation is exactly norm-preserving at any cutoff_dim.

    Args:
        phi:        Rotation angle (real scalar tensor), in radians.
        cutoff_dim: Fock space truncation D.

    Returns:
        Complex tensor of shape (D,). Differentiable w.r.t. phi.
    """
    ns = torch.arange(cutoff_dim, device=phi.device, dtype=torch.float64)
    return torch.exp(1j * phi.to(torch.complex128) * ns)


def rotation_matrix(phi: torch.Tensor, cutoff_dim: int) -> torch.Tensor:
    """Phase-space rotation R(φ) in the Fock basis — backward-compat wrapper.

    Returns torch.diag(rotation_phases(phi, cutoff_dim)). Prefer
    rotation_phases + apply_single_mode_phases for efficiency.

    Args:
        phi:        Rotation angle (real scalar tensor), in radians.
        cutoff_dim: Fock space truncation D.

    Returns:
        Complex tensor of shape (D, D). Differentiable w.r.t. phi.
    """
    return torch.diag(rotation_phases(phi, cutoff_dim))


def _laguerre_all(
    k_grid: torch.Tensor,
    l_grid: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    """Compute L_{k_grid}^{l_grid}(x) for all (k, l) pairs in parallel.

    Uses the 3-term recurrence:
        L_0^l(x) = 1
        L_1^l(x) = 1 + l - x
        L_k^l(x) = ((2k+l-1-x)·L_{k-1} - (k+l-1)·L_{k-2}) / k

    k_grid and l_grid are non-batched integer tensors of the same shape.
    x is a scalar (may be batched). Returns a tensor of the same shape as k_grid.
    """
    shape = k_grid.shape
    L_prev2 = torch.ones(shape, dtype=torch.float64, device=k_grid.device)
    L_prev1 = 1.0 + l_grid.double() - x.double()
    result = torch.where(k_grid == 0, L_prev2, L_prev1)
    max_k = int(k_grid.max().item())   # k_grid is non-batched → .item() is safe
    for step in range(2, max_k + 1):
        l = l_grid.double()
        L_curr = ((2 * step + l - 1 - x.double()) * L_prev1
                  - (step + l - 1) * L_prev2) / step
        result = torch.where(k_grid == step, L_curr, result)
        L_prev2, L_prev1 = L_prev1, L_curr
    return result


def displacement_matrix(alpha: torch.Tensor, cutoff_dim: int) -> torch.Tensor:
    """Displacement D(α) in Fock basis using exact matrix elements.

    Uses the closed-form formula via associated Laguerre polynomials:
        D_{mn}(α) = exp(−|α|²/2) · α^{n−m} · sqrt(m!/n!) · L_m^{n−m}(|α|²)  for n ≥ m
        D_{mn}(α) = exp(−|α|²/2) · (−α*)^{m−n} · sqrt(n!/m!) · L_n^{m−n}(|α|²) for n < m

    The resulting matrix is a sub-isometry: column norms ≤ 1.

    Args:
        alpha:      Complex displacement amplitude.
        cutoff_dim: Fock space truncation D.

    Returns:
        Complex tensor of shape (D, D). Differentiable w.r.t. alpha.
    """
    D = cutoff_dim
    alpha_c = alpha.to(torch.complex128)
    abs2 = (alpha_c * alpha_c.conj()).real       # |α|²  (scalar)

    m_idx = torch.arange(D, device=alpha.device)
    n_idx = torch.arange(D, device=alpha.device)
    m_grid, n_grid = torch.meshgrid(m_idx, n_idx, indexing='ij')   # (D, D)

    s_grid = (n_grid - m_grid).abs()             # |n - m|  (non-batched integer tensor)
    k_grid = torch.minimum(m_grid, n_grid)       # min(m, n)

    # log sqrt(min(m,n)! / max(m,n)!) = 0.5*(lgamma(k+1) - lgamma(k+s+1))
    log_fact_k = torch.lgamma(k_grid.double() + 1)
    log_fact_ks = torch.lgamma((k_grid + s_grid).double() + 1)
    log_sqrt_ratio = 0.5 * (log_fact_k - log_fact_ks)              # (D, D)

    # Associated Laguerre L_{min(m,n)}^{|n-m|}(|α|²)
    L = _laguerre_all(k_grid, s_grid, abs2)                         # (D, D)

    # Phase factor (Cahill-Glauber convention): (-α*)^(n-m) for n>m (upper), α^(m-n) for n<m (lower).
    #
    # Complex pow z**s routes through exp(s·log z), whose *backward* is NaN at
    # z == 0 exactly when s == 1 (it evaluates exp(0·log 0)); s == 0 is NaN
    # already in the forward. α == 0 is a legitimate gate value (the identity,
    # "no push"), so feed the pow only exponents ≥ 2 — its backward is then
    # s·z^(s−1) with exponent ≥ 1, finite by construction — and supply the
    # s ∈ {0, 1} entries by their closed forms (1 and ±α). Values are
    # bit-identical to the pow route for s ≥ 2; the s == 1 entries become
    # literal ±α (more accurate than exp(log α)). The broken set {s=0 fwd,
    # s=1 grad} is pinned exhaustively by tests/test_gate_gradients.py
    # (ADR-0004).
    upper = (n_grid >= m_grid)
    s_pow = s_grid.clamp(min=2).to(torch.complex128)        # pow never sees s < 2
    alpha_pow = torch.where(upper, (-alpha_c.conj()) ** s_pow, alpha_c ** s_pow)
    # s == 1: the value IS the base — ±α directly, no pow involved.
    alpha_pow = torch.where(
        s_grid == 1,
        torch.where(upper, -alpha_c.conj(), alpha_c),
        alpha_pow,
    )
    # For diagonal (s=0): α^0 = 1 by definition; complex 0^0 would give NaN.
    alpha_pow = torch.where(s_grid > 0, alpha_pow,
                            torch.ones(D, D, dtype=torch.complex128, device=alpha.device))

    mat = (torch.exp(-abs2 / 2)
           * alpha_pow
           * torch.exp(log_sqrt_ratio).to(torch.complex128)
           * L.to(torch.complex128))
    return mat


def squeezing_matrix(r: torch.Tensor, phi: torch.Tensor, cutoff_dim: int) -> torch.Tensor:
    """Single-mode squeezing S(r, φ) in the Fock basis via the exact closed form.

    Uses the disentangling identity
        S(r) = exp(-tanh(r)/2 · a†²) · (cosh r)^{-(N + 1/2)} · exp(tanh(r)/2 · a²)
    together with the rotation conjugation
        S(r, φ) = R(φ/2) · S(r, 0) · R(-φ/2)
    to derive the closed-form matrix element (for (m+n) even, else 0):
        ⟨m|S(r, 0)|n⟩ = (cosh r)^{-1/2} · √(m! n!)
                          · Σ_l (-1)^{(m-l)/2}
                                · (tanh r / 2)^{(m+n)/2 - l}
                                · (cosh r)^{-l}
                                / ( l! · ((m-l)/2)! · ((n-l)/2)! )
    where the sum runs over l ∈ {0, 1, …, min(m, n)} with the same parity as m.
    For φ ≠ 0, multiply S_{mn}(r, 0) by exp(iφ(m - n)/2).

    Unlike the older column-by-column Bogoliubov recurrence, this expression
    is numerically stable for any r — each term in the sum is bounded and
    there is no error propagation between columns. The resulting matrix is a
    true sub-isometry: column norms ≤ 1, with the deficit
    1 − ‖S(r, φ) · e_n‖² equal to the probability of finding photon number
    ≥ cutoff_dim in the un-truncated state.

    All tensor ops are out-of-place for vmap compatibility. Factorials are
    evaluated in log-space via `torch.lgamma` to avoid overflow at large D.

    Args:
        r:          Squeezing magnitude (real scalar, r ≥ 0).
        phi:        Squeezing angle in radians (real scalar).
        cutoff_dim: Fock space truncation D.

    Returns:
        Complex tensor of shape (D, D). Differentiable w.r.t. r and phi.
    """
    D = cutoff_dim
    device = r.device
    r64 = r.to(torch.float64)
    cosh_r = r64.cosh()
    tanh_r = r64.tanh()
    half_tanh = tanh_r / 2.0   # signed; r < 0 gives half_tanh < 0

    # --- Build (D, D) meshgrid of integer indices ---
    idx = torch.arange(D, dtype=torch.int64, device=device)
    m_grid, n_grid = torch.meshgrid(idx, idx, indexing='ij')   # (D, D) int64

    # log(k!) for k = 0..D  — used to evaluate factorial ratios via lgamma
    log_fact = torch.lgamma(
        torch.arange(D + 1, dtype=torch.float64, device=device) + 1.0
    )

    # √(m! n!) (real, (D, D))
    sqrt_mn_fact = torch.exp(0.5 * (log_fact[m_grid] + log_fact[n_grid]))

    # Accumulate the sum over l in real arithmetic
    parity_match = ((m_grid + n_grid) % 2 == 0)
    half_sum = (m_grid + n_grid) // 2                                 # (D, D) int64
    accum = torch.zeros(D, D, dtype=torch.float64, device=device)
    for l in range(D):
        # A term contributes only when (l % 2) == (m % 2)
        #   AND (l % 2) == (n % 2)
        #   AND l ≤ min(m, n)
        #   AND (m + n) is even (parity_match)
        valid = (
            parity_match
            & (l <= torch.minimum(m_grid, n_grid))
            & (((m_grid - l) % 2) == 0)
        )

        # (-1)^((m-l)/2) — only used for valid entries; clamp to keep arithmetic
        # well-defined when invalid.
        p_idx = ((m_grid - l).clamp(min=0)) // 2         # (D, D) int64, non-negative
        q_idx = ((n_grid - l).clamp(min=0)) // 2         # (D, D) int64, non-negative
        sign = (-1.0) ** p_idx.to(torch.float64)         # ±1.0

        # (tanh r / 2)^{(m+n)/2 - l}  — integer non-negative exponent for valid entries
        exp_t = (half_sum - l).clamp(min=0).to(torch.float64)
        # Sign-aware power: works for half_tanh < 0 (the exponent is a
        # non-negative integer for valid entries, so `(-1)^exp_t` is ±1).
        # Edge case at r = 0: `0.0 ** 0.0 = 1.0` in torch, consistent with the
        # formula's behaviour at the identity (only the diagonal l = m term
        # contributes and gives 1).
        ht_pow = half_tanh.sign() ** exp_t * half_tanh.abs() ** exp_t

        # (cosh r)^{-l} — scalar
        cosh_pow = cosh_r ** (-l)

        # Denominator 1 / (l! · p_idx! · q_idx!), in log-space
        log_denom = log_fact[l] + log_fact[p_idx] + log_fact[q_idx]
        inv_denom = torch.exp(-log_denom)

        term = sign * ht_pow * cosh_pow * inv_denom
        accum = accum + torch.where(valid, term, torch.zeros_like(term))

    # Apply the common (cosh r)^{-1/2} · √(m! n!) prefactor
    S0 = sqrt_mn_fact * accum / cosh_r.sqrt()                          # (D, D) real
    S0 = S0.to(torch.complex128)

    # --- Apply phase factor for general φ: S_{mn}(r,φ) = exp(iφ(m-n)/2) S_{mn}(r,0) ---
    phases_half = rotation_phases(phi / 2, D).to(device)   # exp(iφn/2), shape (D,)
    S = S0 * phases_half.unsqueeze(1)         # left-multiply: scale rows by exp(iφm/2)
    S = S * phases_half.conj().unsqueeze(0)   # right-multiply: scale cols by exp(-iφn/2)
    return S


def beamsplitter_matrix(
    theta: torch.Tensor, phi: torch.Tensor, cutoff_dim: int
) -> torch.Tensor:
    """Two-mode beamsplitter BS(θ,φ) in Fock basis using photon-number sectors.

    Conserves total photon number N_tot = n_a + n_b. The (D,D,D,D) tensor is
    assembled sector by sector, each of size ≤ D. No D²×D² matrix is built.

    Matrix elements within sector N_tot = N:
        BS[m, N-m, k, N-k] = sqrt(k!(N-k)!/m!(N-m)!) * Σ_j C(k,j)*C(N-k,t)
                              * cos^{N-j-t} * (-sin·e^{-iφ})^t * (sin·e^{iφ})^j
    where t = m-k+j. Convention matches SF: BS_CV(θ,φ) = BS_SF(θ,φ).

    All tensor ops are out-of-place for vmap compatibility. Powers of cos/sin
    are computed in float64 to avoid complex 0^0 → NaN.

    Args:
        theta:      Beamsplitter mixing angle (real scalar).
        phi:        Phase angle (real scalar).
        cutoff_dim: Fock space truncation D.

    Returns:
        Complex tensor of shape (D, D, D, D) indexed [out_a, out_b, in_a, in_b].
        Differentiable w.r.t. theta and phi.
    """
    D = cutoff_dim
    device = theta.device

    # Use real cos/sin to avoid complex 0^0 → NaN (real 0^0 = 1 in PyTorch)
    cos_r = theta.to(torch.float64).cos()
    sin_r = theta.to(torch.float64).sin()
    phi_c = phi.to(torch.complex128)

    # log(n!) = lgamma(n+1) for n = 0..2D-2
    log_fact = torch.lgamma(
        torch.arange(2 * D, dtype=torch.float64, device=device) + 1
    )

    # Accumulate into a flat (D^4,) tensor using out-of-place index_put
    BS_flat = torch.zeros(D * D * D * D, dtype=torch.complex128, device=device)

    for N in range(2 * D - 1):
        k_min = max(0, N - D + 1)
        k_max = min(N, D - 1)
        if k_min > k_max:
            continue

        k_arr = torch.arange(k_min, k_max + 1, device=device)   # (B_sz,)
        m_arr = torch.arange(k_min, k_max + 1, device=device)   # (B_sz,)
        B_sz = k_arr.shape[0]

        k_g, m_g = torch.meshgrid(k_arr, m_arr, indexing='ij')  # (B_sz, B_sz)
        l_g = N - k_g   # input b-mode count
        n_g = N - m_g   # output b-mode count (unused but kept for clarity)

        # log sqrt(m! n! / k! l!)  — output factorials in numerator (from Bogoliubov derivation)
        log_sqrt = 0.5 * (log_fact[m_g] + log_fact[n_g]
                          - log_fact[k_g] - log_fact[l_g])

        result = torch.zeros(B_sz, B_sz, dtype=torch.complex128, device=device)
        for j in range(N + 1):
            t_g = m_g - k_g + j   # (B_sz, B_sz) integer, may have invalid entries
            valid = (j <= k_g) & (t_g >= 0) & (t_g <= l_g)
            if not valid.any():
                continue

            t_safe = t_g.clamp(0)
            log_binom = (log_fact[k_g] - log_fact[j] - log_fact[(k_g - j).clamp(0)]
                         + log_fact[l_g] - log_fact[t_safe]
                         - log_fact[(l_g - t_safe).clamp(0)])

            # The exponents are photon counts — non-negative for every *valid*
            # entry; only the padding entries masked out below can come out
            # negative. Clamp them to ≥ 0 BEFORE the pow: at sin θ == 0
            # exactly, an unclamped negative exponent materialises
            # 0^negative = Inf, which the mask discards in the forward but the
            # product rule multiplies against the masked-zero gradient in
            # backward — 0·Inf = NaN, the dead-head trigger of the 2026-06
            # sweep post-mortem (ADR-0004). With the clamp the padding
            # computes 0^0, whose forward (1) and gradient (0) torch guards —
            # the same pattern the squeeze gate already uses for exp_t.
            # Pinned by tests/test_gate_gradients.py.
            p_cos = (N - j - t_g).clamp(min=0).to(torch.float64)
            p_sin = (j + t_g).clamp(min=0).to(torch.float64)

            # Real powers: avoids complex 0^0 → NaN when cos or sin is zero
            cos_pow = cos_r ** p_cos   # real
            sin_pow = sin_r ** p_sin   # real

            # Phase: (-e^{-iφ}·sin)^t * (e^{iφ}·sin)^j → e^{iφ(j-t)} * (-1)^t  [SF convention]
            phase_exp = torch.exp(1j * phi_c * (j - t_g).to(torch.complex128))
            sign_t = (-1.0) ** t_g.to(torch.float64)   # (B_sz, B_sz) — exponent is t, not j

            term = (torch.exp(log_binom.to(torch.complex128))
                    * cos_pow.to(torch.complex128)
                    * sin_pow.to(torch.complex128)
                    * phase_exp * sign_t.to(torch.complex128))
            result = result + torch.where(valid, term, torch.zeros_like(term))

        result = result * torch.exp(log_sqrt.to(torch.complex128))

        # Scatter result into BS_flat using out-of-place index_put
        # Linear index: BS[m, N-m, k, N-k] → m*D³ + (N-m)*D² + k*D + (N-k)
        lin_idx = (m_g * (D ** 3) + (N - m_g) * (D ** 2) + k_g * D + (N - k_g)).reshape(-1)
        BS_flat = BS_flat.index_put((lin_idx,), result.reshape(-1))

    return BS_flat.reshape(D, D, D, D)


def two_mode_squeezing_matrix(
    r: torch.Tensor, phi: torch.Tensor, cutoff_dim: int
) -> torch.Tensor:
    """Two-mode squeezing S2(r, φ) = exp(r(e^{-iφ} ab - e^{iφ} a†b†)).

    Generator in D²×D² space:
        G = r(e^{-iφ} kron(a,a) - e^{iφ} kron(a†,a†))

    S2 produces entangled two-mode squeezed (EPR) states. In the limit r→∞
    it creates a perfectly entangled state (ideal EPR pair).

    Args:
        r:          Squeezing parameter (real scalar, r ≥ 0).
        phi:        Phase angle (real scalar).
        cutoff_dim: Fock space truncation D.

    Returns:
        Complex tensor of shape (D, D, D, D) indexed [out_i, out_j, in_i, in_j].
        Differentiable w.r.t. r and phi.
    """
    D = cutoff_dim
    device = r.device

    ns = torch.arange(1, D, dtype=torch.float64, device=device)
    a_dag = torch.diag(torch.sqrt(ns), diagonal=-1).to(torch.complex128)
    a = a_dag.mH.contiguous()

    r_c = r.to(torch.complex128)
    e_iphi = torch.exp(1j * phi.to(torch.complex128))

    generator = r_c * (e_iphi.conj() * torch.kron(a, a) - e_iphi * torch.kron(a_dag, a_dag))
    S2_flat = torch.linalg.matrix_exp(generator)   # (D², D²)
    return S2_flat.reshape(D, D, D, D)             # [out_a, out_b, in_a, in_b]

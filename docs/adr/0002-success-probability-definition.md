# Success probability = ‖P(M)|ψ⟩‖² / λ², stored as raw norms

Status: accepted

The thesis figures report the heralding probability of the post-selected LCU/QSVT
step as the normalised ratio `‖P(M)|ψ⟩‖² / λ²`, with the subnormalisation
`λ = Σ_j |c_j| · (Σ_i |b_i|)ʲ` — the nested-LCU block-encoding scale of
`P(M) = Σ_j c_j Mʲ` built from `M = Σ_i b_i U_i`. The artefacts store only the raw
per-sample norms `‖P(M)|ψ⟩‖²` (in `predictions/epoch_NNNN[_train].npz` under
`success_probs`, shape `(N, num_heads)`); λ is **derived at figure time** from the
`lcu_coeffs` / `poly_coeffs` already saved per epoch in `diagnostics/epoch_NNNN.npz`,
never duplicated into storage.

Why the ratio and not the raw norm: the learned `b_i` and `c_j` are unconstrained, so
the code's internal `success_prob` variable (the raw norm, `cv_attention.py`
`_apply_polynomial_iterative`) can exceed 1 and is *not* a probability — it conflates
coefficient magnitude with state overlap. The ratio is the physical success
probability of a heralded implementation on hardware. Raw norms are kept (rather than
storing the ratio) because they remain useful for diagnosing the
gate-param-bound / NaN-head regime and λ is recoverable for free.

Caveats recorded deliberately:

- The ratio is the success probability **in the truncated simulation**: the `U_i` are
  sub-isometries at finite cutoff, so a small part of the norm deficit is Fock
  truncation leakage, not heralding failure.
- The CVQNN block `W` has **no** success probability — it is unitary in exact
  arithmetic; its norm deficit is pure truncation leakage and stays on the separate
  `w_trunc_loss` track. Do not fold it into this figure.

# Success probability = ‖P(M)|ψ⟩‖² / λ², stored as raw norms

Status: accepted

The thesis figures report the heralding probability of the post-selected LCU/QSVT
step as the normalised ratio `‖P(M)|ψ⟩‖² / λ²`, with the subnormalisation
`λ = Σ_j |c_j| · (Σ_i |b_i|)ʲ` — the nested-LCU block-encoding scale of
`P(M) = Σ_j c_j Mʲ` built from `M = Σ_i b_i U_i`. The artefacts store only the raw
per-sample norms `‖P(M)|ψ⟩‖²` (in `predictions/epoch_NNNN[_train].npz` under
`success_probs`, shape `(N, num_heads)` for the canonical models — see the stacked
model below); λ is **derived at figure time** from the `lcu_coeffs` / `poly_coeffs`
already saved per epoch in `diagnostics/epoch_NNNN.npz`, never duplicated into
storage.

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

## The stacked model: one stage, and an upper bound

The seq-to-seq stacked model (ADR-0003) post-selects **once per stage** — once per
seq-to-seq block, plus once more in the aggregator under `pooling="quixer"` — where
the canonical models post-select once in total. Three consequences, all of them
scoping decisions rather than definition changes; the ratio and its λ mean exactly
what they mean above.

- **Shape.** A stacked head emits one value per sequence *position* as well as per
  sample and head, so `success_probs` is `(N, num_heads, num_positions)` under
  `pooling="mean"`. (Under `pooling="quixer"` the recorded stage is the seq-to-one
  aggregator, which restores the canonical `(N, num_heads)`.) The figures fold the
  position axis into the sample population rather than averaging it away — the same
  treatment `_state_stats` gives the stacked model's per-position states — so each
  (sample, position) pair contributes one count per head.
- **Stage.** Only the **decoder-input stage**'s success probabilities reach the
  artefacts: `StackedCVQuixer.forward` keeps the last block's (or the aggregator's)
  and discards the earlier blocks'. This follows ADR-0003's existing definition of
  "the" diagnostic state, which already scopes state norms and photon numbers the
  same way. λ therefore comes from that stage's block-prefixed coefficients
  (`block{b}_lcu_coeffs` / `agg_lcu_coeffs`), never the flat canonical keys — which a
  stacked run does not write at all.
- **Upper bound.** Because `‖P(M)|ψ⟩‖ ≤ Σ_j |c_j| · ‖M‖ʲ ≤ λ` for the unit-norm input
  every stage receives (the `U_i` are sub-isometries, so `‖M‖ ≤ Σ_i |b_i|`), the ratio
  is in `[0, 1]` per stage. End-to-end heralding success is a product of such factors,
  so **the reported figure is an upper bound on the end-to-end value**, not an
  estimate of it. The bound is robust to the fact that a block's shared `M` mixes all
  positions: any product of factors in `[0, 1]` is at most the last factor, however
  they correlate. The thesis captions state this on multi-stage runs; a claim that
  post-selection is *affordable* cannot rest on this figure, whereas a claim that it
  is *costly and compounds with depth* can.

Measuring the omitted stages is deliberately out of scope here — it would need the
forward pass to keep every stage's values and every stacked run re-evaluated, whereas
the scoping above is derivable from artefacts already on disk.

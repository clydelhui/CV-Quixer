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

## The stacked model: one stage per figure, and costs that add

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
- **Stage.** During *training* only the **decoder-input stage**'s success
  probabilities reach the artefacts: `StackedCVQuixer.forward` keeps the last block's
  (or the aggregator's) and discards the earlier blocks'. This follows ADR-0003's
  existing definition of "the" diagnostic state, which already scopes state norms and
  photon numbers the same way. λ always comes from the plotted stage's block-prefixed
  coefficients (`block{b}_lcu_coeffs` / `agg_lcu_coeffs`), never the flat canonical
  keys — which a stacked run does not write at all. Earlier stages are recovered
  post-hoc by `experiments/eval_block_stages.py`, which re-evaluates a run block by
  block from its checkpoints and writes one sidecar per (epoch, stage)
  (`predictions/epoch_NNNN_block{b}.npz`); the figures then emit one file per stage.
- **Per stage the ratio is a probability.** `‖P(M)|ψ⟩‖ ≤ Σ_j |c_j| · ‖M‖ʲ ≤ λ` for the
  unit-norm input every stage receives (the `U_i` are sub-isometries, so
  `‖M‖ ≤ Σ_i |b_i|`), so the ratio is in `[0, 1]` for each stage independently.

### Costs add across stages, heads and positions — they do not multiply

An earlier revision of this ADR claimed end-to-end heralding was a *product* of the
per-stage ratios, making a single-stage figure an upper bound. **That is wrong**, and
the error mattered: it framed the cheapest stage as if it bounded the whole model.

Nothing in this architecture requires two heralded events to succeed on the *same
shot*:

- **Heads** are independent registers with no entanglement between them, and each
  head's readout is an *expectation value* estimated from many shots. Head `h`'s
  statistics can be collected on their own and the classical numbers concatenated.
- **Blocks** chain **classically**. `tokens` is a real tensor of readouts
  (`cv_seq2seq.py`, the block loop in `forward`) and every block re-prepares from
  `FockState.vacuum`. No quantum state crosses a block boundary — the model is
  inherently measure-and-refeed, so no coherent end-to-end version of it exists.
- **Positions** under `pooling="mean"` are separate circuit runs for the same reason.

So with `S` the successful shots needed per readout estimate, the cost of one
inference is a **sum**:

```
T = Σ_stages Σ_heads Σ_positions  S / p
```

Consequences for reporting:

- The correct per-head aggregate is `Σ_h ⟨1/p_h⟩` — total shot overhead — or
  equivalently the **harmonic mean** of the per-head probabilities, which is dominated
  by the *worst* head. Never the arithmetic mean (dominated by the best head) and
  never a product.
- `⟨1/p⟩ ≠ 1/⟨p⟩`. Cost is an expectation of a reciprocal; using `1/mean(p)`
  understated the measured runs by 5–7%.
- A figure covering one stage is **one term of a sum**, so it *understates* total
  cost. It is not a bound on the end-to-end probability. This is why every stage is
  now measured rather than one being reported as a bound.

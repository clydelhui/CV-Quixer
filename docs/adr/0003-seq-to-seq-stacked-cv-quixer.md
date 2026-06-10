---
status: accepted
---

# Seq-to-seq stacked CV-Quixer as a separate model, with query unitaries from a shared hypernet trunk

## Context

The canonical CV-Quixer head is seq-to-one: a whole patch sequence collapses to
a single readout vector per head, so attention blocks cannot be stacked. We want
a stackable seq-to-seq attention block — position i's output token is the
readout of `W · P(M) · U_{q,i}|0⟩`, where the query unitary `U_{q,i}` is
per-position and input-dependent, while `M`, `P`, and `W` are shared across
positions within a head. The new model (`model="quantum_stacked"`) exists
*alongside* `quantum`/`quantum_shared` as a comparison arm; it never replaces
them.

## Decision

One new model family, `quantum_stacked`: `num_seq2seq_blocks ≥ 1` uniform
seq-to-seq blocks, followed by either mean-pooling over positions or an
**aggregator block** (`pooling="quixer"` — a canonical seq-to-one head stack on
the final token sequence). Both endings produce the same `H×R` decoder input as
the canonical model, so the decoder is identical across all variants and the
comparison isolates the attention stack.

## Considered options / trade-offs

- **Query parameter source.** `U_{q,i}` params are a *second slice of the same
  hypernet output* (final linear width doubles), not a second hypernetwork and
  not owned per-position parameters. Mirrors classical attention (Q and K are
  projections of one token embedding), keeps queries input-dependent, and adds
  params only in the final linear. A separate CNN per head buys capacity in conv
  filters — the least relevant place — at ~30% more cost; owned-parameter
  queries would make position i ask the same question regardless of content.
  `U_{q,i}` reuses `U_i`'s exact gate plan (acts on vacuum ⇒ no leading
  interferometer).
- **Query-state renormalisation.** `U_{q,i}|0⟩` is renormalised *before* `P(M)`
  (same clamp-divisor + zero-on-failure guard as the existing renorms). On real
  hardware the query preparation is unitary; the truncation leakage is a
  simulation artifact. Renormalising keeps `success_prob` meaning exactly what
  it means in the canonical model (pure polynomial post-selection probability),
  preserving diagnostic comparability. The leakage is tracked as a **separate**
  stream `query_trunc_loss`, weighted by a new `query_trunc_lambda` (default
  0.01 — single-application leakage like W's, not the compounding per-patch
  0.1).
- **Per-patch trunc loss redefined.** In a seq-to-seq block the polynomial's
  input is the query state, not vacuum, so the per-patch penalty becomes the
  LCU terms' leakage **on the block's actual input states**
  (`1 − Σ_i ‖U_i|q_j⟩‖²/N`, mean over positions). This is what physically
  compounds through the polynomial, it stays free (fused into the first LCU
  pass per query), and it degenerates to the historical vacuum-referenced
  definition wherever the input *is* vacuum (canonical models, aggregator
  block). Keeping the vacuum definition would cost an extra full pass to
  preserve an accident of the old input.
- **Engine strategy: iterative, no materialisation.** `P(M)|q_i⟩` is computed
  per query via the existing iterative LCU/polynomial path, with a new vmap
  axis (head → batch → query → patch). Cost is `d·N²` gate-plan applications
  per head per block — ~N× the canonical model, × blocks. Materialising `M` as
  a `D^m × D^m` matrix once per head is actually *cheaper* at the current
  defaults (`D^m = 36 < d·N = 48`) but inverts badly as `m` grows, forks the
  engine into two maintained paths, and breaks the documented "no LCU matrix is
  ever materialised / no `D^{3m}`" invariant the thesis cost narrative relies
  on. We accept the bounded constant-factor loss.
- **Inter-block interface.** Block 1 keeps per-head CNN hypernetworks (input is
  genuinely image patches; matches the canonical comparison baseline); blocks
  ≥ 2 use per-head `Linear(H×R → 2×gate_params)` (the `LinearCVHead` pattern —
  deeper transformer blocks are linear projections too). Positional encoding is
  injected once, in block 1 (standard ViT practice); deeper blocks get position
  through tokens and their per-position `b_i`. Blocks are uniform (same heads,
  modes, observables ⇒ constant token width) and share **no** parameters.
  Identity residuals `x + block(x)` from block 2 onward (block 1's shapes
  differ), togglable via `block_residual` (default on) as the trainability/
  purity ablation; LayerNorm deliberately withheld unless residuals alone fail.
- **Aggregator accounting.** The aggregator is *not* counted by
  `num_seq2seq_blocks` (it is seq-to-one, hence the field name), and
  `num_seq2seq_blocks ≥ 1` is enforced: an aggregator-on-raw-patches model is
  exactly the existing CVQuixer, which remains the sole owner of that
  configuration.
- **No migration guard for the new config fields.** ADR-0001's loud guard
  exists because `cvqnn_num_layers`'s absence was *ambiguous* for the existing
  model. The new fields (`num_seq2seq_blocks`, `pooling`, `block_residual`,
  `query_trunc_lambda`) cannot affect any model an old `config.json` can
  describe, so dacite defaults are semantically safe. Validity is enforced at
  construction in `QuantumConfig.__post_init__` instead.

## Consequences

- Quantum compute per head scales as `d·N²` per block (×N over canonical);
  stacking multiplies further. Cutoff sweeps on stacked runs need smaller
  `--test-fraction`.
- Loss gains a third truncation stream:
  `CE + λ_trunc·patch + λ_query·query + λ_cvqnn·W`; all streams are flat means
  over every axis they have. `success_prob` gains a position axis under the
  same floor/warning machinery.
- Diagnostics npz keys gain `block{b}_`/`agg_` prefixes for stacked runs
  (canonical key names untouched); "the" diagnostic state is defined as the
  decoder-input stage (aggregator states, or last block's per-position states).
- `num_seq2seq_blocks` is a legal but coarse `scaling_knob`; `num_heads` stays
  the default. New `full_experiment.py` flags / sweep axes / run-dir markers
  (`__sb{n}`, `__pool-…`, `__nores`) follow existing conventions; stacked
  defaults are otherwise identical to the canonical frozen baseline so the
  stacking is the only varied factor.

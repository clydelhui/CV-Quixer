# The `qsvt` polynomial mode alternates the LCU adjoint, rather than raising the LCU to literal matrix powers

The [[polynomial]] `P(M) = Σⱼ cⱼ Mʲ` is described in CONTEXT.md as "modelling
post-selected QSVT", but the `standard` construction — apply the [[LCU]] `M`
`j` times to build `Mʲ` — is a faithful quantum singular-value transform (QSVT)
*only when `M` is Hermitian*. Our `M = Σᵢ bᵢ Uᵢ` is a general, non-Hermitian
operator, so `Mʲ` is not the singular-value monomial `σʲ`. We add a second
[[polynomial mode]], `qsvt`, that builds the genuine singular-value transform by
alternating `M` with its adjoint `M†`. The existing model is untouched: `qsvt`
is an opt-in knob (`poly_mode`), `standard` stays the default and byte-identical.

Three design choices in `qsvt` have plausible-looking wrong alternatives; this
ADR records why we picked each.

## Even terms are `M†M`, not `M M†`

Write the SVD `M = Σₖ σₖ |lₖ⟩⟨rₖ|` (`|rₖ⟩` right/input singular vectors, `|lₖ⟩`
left/output). The QSVT singular-value transform of a monomial has definite
parity:

```
even  xⁿ → V Σⁿ V† = (M†M)^{n/2}      (stays in the input singular-vector space)
odd   xⁿ → W Σⁿ V† = M (M†M)^{(n-1)/2} (maps input → output)
```

Built iteratively from `v₀ = ψ` by alternating applications — `vⱼ = M vⱼ₋₁` when
`j` is odd, `vⱼ = M† vⱼ₋₁` when `j` is even — this gives

```
v₀ = I ,  v₁ = M ,  v₂ = M†M ,  v₃ = M M†M ,  v₄ = M†M M†M ,  …
```

So `x² = M†M`, **not** `M M†`. This is forced, not cosmetic: `M†M = VΣ²V†`
lives in the *input* space, the same space as the identity term `x⁰ = I`, so the
even terms stack coherently with `I` and with each other. `M M† = WΣ²W†` lives in
the *output* space and would not. The user's headline example `x³ → M M† M` is
`v₃` and is unaffected by this choice; the even terms are what the choice pins
down.

## `M†` is the conjugate-transpose of the truncated matrix, not the gate with negated parameters

`M† = Σᵢ bᵢ* Uᵢ†`. Each `Uᵢ` is an ordered product of Fock-basis gate matrices,
so `Uᵢ†` reverses the op order *and the site order within each op* (the
beamsplitter mesh does not commute) and daggers each gate. There are two ways to
dagger a gate:

- **Conjugate-transpose the truncated matrix** the gate already builds
  (`Mᴳ.conj().T`; diagonal phase gates conjugate their phase vector).
- **Re-derive with negated parameters** (`S(r)† “=” S(−r)`, `D(α)† “=” D(−α)`, …).

These agree only for exact unitaries. At finite cutoff `D` the analytic Fock
matrices are deliberate sub-isometries (column norms ≤ 1), so the truncated
`S(−r)` is *not* the conjugate-transpose of the truncated `S(r)`. Parameter
negation would make `M†` a *different operator* than the adjoint of the `M` we
actually apply — and then `M†M = VΣ²V†` no longer holds, breaking the very
singular-value identities that motivate `qsvt`. We take the conjugate-transpose:
`M†` is by construction the true adjoint of the simulated `M`. Implementation is
additive — `GateOp` gains an `apply_dagger` callback the `standard` path never
calls, and `_CVHeadBase` gains `_apply_gate_plan_dagger` — so the `standard`
code path and its checkpoints are byte-identical.

## Mixed parities are realised by plain vector addition, not an extra herald

A textbook QSVT circuit produces a definite-parity polynomial; our `P(M)` sums
all `j = 0..d`, so the even part (input space) and odd part (output space) coexist.
In hardware these are two different flag states of a block-encoding ancilla and
superposing them needs an explicit parity-LCU. In the simulation they are just
two vectors, and the existing `result = Σⱼ cⱼ vⱼ` already adds them — vector
addition *is* the parity superposition. `success_prob = ‖result‖²` is the joint
post-selection over both branches, and the [[subnormalisation]]
`λ = Σⱼ|cⱼ|(Σᵢ|bᵢ|)ʲ` is unchanged (dagger conjugates `bᵢ`, so `Σ|bᵢ*| = Σ|bᵢ|`,
and term `j` still costs `j` block-encoding applications). No new loss,
diagnostic, or penalty is needed.

## Consequences

`poly_mode ∈ {standard, qsvt}` is a `QuantumConfig` string knob defaulting to
`standard`, so pre-existing runs and checkpoints reload byte-identically (absent
key → `standard` → the prior code path). It reuses the same `bᵢ`/`cⱼ`/`Uᵢ`
parameters — a `qsvt` head's `state_dict` is structurally identical to a
`standard` one; only the forward computation differs — so it is *not* a new
parameter structure and did **not** warrant a new `model` string. It lives in the
shared head base (`_CVHeadBase`), so one change point covers all model variants
(`quantum`, `quantum_shared`, `quantum_stacked`) and the aggregator block, and
the daggering acts on `M` regardless of whether the polynomial's input state is
the vacuum or a seq-to-seq query state.

At `poly_degree ≤ 1` the two modes coincide (no `M†` application occurs), so
`qsvt` diverges from `standard` only at `degree ≥ 2` — the regression test pins
both facts.

The knob is orthogonal to `coeff_ablation`, `poly_init_noise`, and
`positional_encoding`: `qsvt` composes with a frozen `lcu_poly` polynomial
(`P = Σⱼ` alternating terms) or a symmetry-broken init without special-casing.
Exposed as `full_experiment.py --poly-mode` and a manual sweep axis with the
`__qsvt` run-dir marker.

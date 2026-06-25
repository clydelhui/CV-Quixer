# Coefficient ablation freezes coefficients to uniform values, rather than tying them to a single trainable scalar

We want two ablations on the learned combination coefficients of a CV-Quixer
head (CONTEXT.md: **coefficient ablation**), to isolate whether the learned
weighting structure earns its keep:

- **`lcu`** — collapse the per-position [[LCU]] coefficients `bᵢ` (a complex
  scalar per patch, `M = Σᵢ bᵢ Uᵢ`) to a single value shared across positions.
- **`lcu_poly`** — additionally collapse the [[polynomial]] coefficients `cⱼ`
  (`P(M) = Σⱼ cⱼ Mʲ`) to a single value.

The natural first framing — keep each as *one trainable scalar* — is what the
ablations were originally described as. We rejected it: under this model's
algebra a single trainable scalar carries no expressivity, so "train it" and
"freeze it" describe the same function class, and freezing is the honest,
lower-variance, fewer-moving-parts choice. Two facts drive this.

## A single trainable LCU scalar is gauge-redundant with the polynomial

A shared LCU scalar `b` factors out of every polynomial power, because it is a
scalar and commutes with the (non-commuting) unitaries:
`(b · Σᵢ Uᵢ)ʲ = bʲ (Σᵢ Uᵢ)ʲ`. So

```
P(M) = Σⱼ cⱼ bʲ (Σᵢ Uᵢ)ʲ ,
```

and with the per-degree `cⱼ` still trainable (the `lcu` arm), any `bʲ` is
absorbable into `cⱼ` (set `cⱼ' = cⱼ bʲ`). The pair `(b, {cⱼ})` and `(1, {cⱼ
bʲ})` realise the identical operator. A trainable shared `b` is therefore a pure
gauge freedom — it adds nothing the polynomial cannot already express. The
*only* non-redundant change the `lcu` ablation makes is removing the
**per-position** weighting (`bᵢ → uniform`); the scalar's trainability is a
no-op. We freeze it at `bᵢ = 1/N` (the existing uniform init), giving the fixed
average `M = (1/N) Σᵢ Uᵢ`.

## A single polynomial scalar is washed out by the post-selection renormalisation

`P(M)|ψ⟩` is renormalised before readout (CONTEXT.md: [[success probability]]).
A global scalar `c` multiplying the whole polynomial cancels:

```
c·P(M)|ψ⟩ / ‖c·P(M)|ψ⟩‖ = (c/|c|) · P(M)|ψ⟩ / ‖P(M)|ψ⟩‖ ,
```

and `c/|c|` is a global sign/phase that leaves every real-observable readout
invariant. A single trainable polynomial scalar is thus **readout-inert** — it
moves only the success-probability / truncation auxiliary loss, never the
classification output. Training it would be measuring noise. So the `lcu_poly`
arm freezes `cⱼ` at all-ones, `P(M) = Σⱼ Mʲ`, the genuinely-different
*equal-weight* polynomial (the real content of the ablation is the lost
per-degree weighting, not the magnitude).

## Freeze at all-ones, not at the `c = [1, 0, …]` init

The obvious "uniform" value — the existing coefficient init `c = [1, 0, …]` —
is a trap: it makes `P(M) = I`, which discards the LCU result entirely and
turns the head into a bare state preparation. The frozen polynomial must be
equal *and nonzero across degrees*; `cⱼ = 1 ∀j` (`P = I + M + ⋯ + Mᵈ`) is the
faithful uniform analogue of the full polynomial, keeping the identity
passthrough term. A side effect: because `P(M) ≠ I` at init, the `lcu_poly` arm
structurally precludes the `P(M)=I` initialisation that drives
[[uniform-predictor collapse]].

## Consequences

The knob is a single ordered enum `coeff_ablation ∈ {none, lcu, lcu_poly}`
(cumulative), defaulting to `none` so pre-existing runs and checkpoints reload
byte-identically (absent key → `none` → trainable `nn.Parameter`s, exactly the
prior code path). The frozen coefficients are non-trainable (excluded from the
optimiser and the parameter count), so the ablation runs are *smaller* than
their baselines by `2N + (d+1)` parameters per head — acceptable because the
sweep configs are manual-mode (no `target_params` autoscaling), so the
architecture is otherwise pinned to the baseline. Coefficients live in the
shared head base (`_CVHeadBase`), so one change point covers all three model
variants (`quantum`, `quantum_shared`, `quantum_stacked`) and the aggregator
block.

The cost: the ablation arms are no longer "the same model with one trainable
parameter" — they are strictly constrained sub-models. That is the correct
framing for an ablation, but it means the comparison is "learned weighting vs
*no* weighting", not "rich weighting vs *one* weight". A future reader who
expected the latter (from the informal "single trainable `b`" description)
should read this ADR.

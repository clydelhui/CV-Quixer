---
status: accepted
---

# Apply a fixed Killoran CVQNN block after the polynomial, before readout

## Context

The CV-Quixer head was measuring observables directly on the post-selected
polynomial output `P(M)|ψ⟩`. This is incorrect: the model is missing a
variational quantum-neural-network stage between the polynomial and the
readout. Every other circuit in the model is *hypernetwork-driven* and
input-dependent, so a future reader would reasonably assume any added circuit
should be too — hence this record.

## Decision

Insert a **CVQNN block `W`** into each head: a fixed, per-image, trainable
Killoran-style variational circuit applied to the renormalised post-selected
state, with observables measured on `W·P(M)|ψ⟩` instead. `W`'s gate parameters
are owned `nn.Parameter`s (input-independent), distinct from the
hypernetwork-emitted per-patch unitaries `U_i`. One `W` per head; lives in the
shared `_CVHeadBase` so both `quantum` and `quantum_shared` variants inherit it.

Pipeline per head: `vacuum → LCU+Polynomial → post-select renorm → W → renorm → readout`.

## Considered options / trade-offs

- **Parameter source.** Owned `nn.Parameter`s, *not* hypernet-emitted — the
  per-patch tokens are already consumed by the LCU, leaving nothing to
  condition a hypernet on. This is the literal Killoran CVQNN.
- **Canonical form.** Each `W` layer is the *full two-interferometer* Killoran
  form `U₁ → S → U₂ → D → Φ`, unlike `U_i` which drops the leading `U₁`
  (trivial on the vacuum `U_i` first acts on). `W` acts on a non-vacuum state,
  so `U₁` is restored. Built as `_INTERFEROMETER_SEQUENCE + _GATE_SEQUENCE`,
  reusing existing `GateOp`s.
- **Interferometer granularity.** `W`'s interferometers reuse the single
  beamsplitter-column `_INTERFEROMETER_SEQUENCE` (option b), *not* the universal
  Clements mesh. Exactly universal at the current `num_modes=2` default; revisit
  if scaling modes up, since a single column is not universal for `m>2`.
- **Depth.** New `cvqnn_num_layers` knob, default `1`. `0` disables `W`
  entirely (clean ablation against the old model). Legal-but-not-default
  `scaling_knob` (too coarse for budget targeting).
- **Init.** Small Gaussian noise (`std 1e-2`) ⇒ `W ≈ I` at start (not exactly).
  Exact zero-init was the first choice (clean "starts as the old model"
  narrative), but it lands precisely on a **NaN-gradient singularity** in the
  analytic displacement (`α=0`, off-diagonal complex power `exp(s·log α)`) and
  beamsplitter (`θ=0`, `sin(θ)**0`) Fock formulas. The hypernet-emitted `U_i`
  params are continuous linear outputs that are never *exactly* zero, so this
  latent singularity never surfaced before; only `W`'s deliberate zero-init hit
  it. Small noise sits off the singularity (as the hypernet params already do)
  and additionally breaks `W`-symmetry across heads. Consequence: there is no
  exact "`cvqnn_num_layers=1` equals `0` at init" point — the parity guarantee
  is only the structural one below (`L_W=0` ⇒ identical `state_dict`).
- **Normalisation.** `W` applies *after* post-selection; the state is
  renormalised again after `W` before readout (truncated squeeze/displace are
  sub-isometries, and readout is raw `Tr(ρÔ)` with no internal normalisation).
- **Truncation penalty.** `W`'s leakage `1 − ‖W|ψ⟩‖²` is tracked and penalised
  *separately* from the per-patch penalty (which compounds through the
  polynomial powers and warrants a heavier weight), via a new
  `cvqnn_trunc_lambda` (default `0.01`, vs `0.1` for the per-patch term). Free
  to compute — it is the same norm used for the post-`W` renormalisation.

## Consequences

- **Canonical default flips to `cvqnn_num_layers=1`** (`QuantumConfig` +
  `full_experiment.py`). The corrected model is the default everywhere;
  `cvqnn_num_layers=0` survives only as the ablation switch.
- **Checkpoint-incompatible** with all prior runs when `W` is on. The frozen
  13,530-param baseline is retired and headline numbers must be re-run.
- **Migration, not silent compatibility.** Old runs load via
  `experiments/migrate_add_cvqnn_field.py`, which bakes `cvqnn_num_layers: 0` /
  `cvqnn_trunc_lambda: 0.0` into existing `config.json` files. A **loud guard**
  at the dacite reload sites raises a hinted error on the missing key rather
  than fabricating a default — deliberately *no* silent shim, to avoid a
  permanent "absence means 0" footgun that would mask future config bugs.

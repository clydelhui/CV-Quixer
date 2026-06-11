# Non-finite training failures fail loudly; no silent NaN containment

Status: accepted

Context: the 2026-06 sweep NaN post-mortem. Three runs trained for epochs with
two permanently-dead heads while CE/accuracy kept improving — the failure was
invisible outside the trunc-loss stream. Root cause: the beamsplitter and
displacement Fock-matrix builders had NaN-singular *gradients* at exactly-zero
gate parameters (finite forward), killing a head's hypernetwork in one Adam
step. Two containment paths then shaped what was observable: the
post-selection guard `torch.where(norm_sq > floor, …)` zeroed the dead head's
state (keeping CE finite — the *silent* part), while `clamp(NaN)` in the same
renorm leaked NaN gradients into the head's LCU/poly coefficients (the
*spreading* part).

Decision, three parts:

1. **Fix non-finite sources at the source, never by sanitising downstream.**
   The gate builders are patched so no Inf/NaN is ever materialised
   (`beamsplitter_matrix`: padding exponents clamped to ≥ 0;
   `displacement_matrix`: pow exponents floored to 2 with closed-form s ∈ {0,1}
   entries — complex `z**s` has a NaN backward exactly at s = 1, z = 0).
   Pinned by `tests/test_gate_gradients.py` (exhaustive signed-zero × exponent
   scan, finite-difference gradient checks at the singular points, golden-file
   value preservation, end-to-end forced-zero regression).

2. **The renorm's NaN propagation in backward is deliberate — do not
   "harden" it.** `_postselect_renorm` intentionally does NOT `nan_to_num` its
   `norm_sq` before the clamp. A future unknown NaN source must corrupt
   visibly (per-head trunc → NaN, grad groups → NaN) so the always-on debug
   instrumentation (`cv_quixer/utils/debug_nan.py`, `debug/` artefacts) fires:
   forensic dump + anomaly replay at the event, abort at the end of the first
   event-free epoch. Sanitising would convert "forensic dump + abort" into
   "quietly absorbed, results subtly wrong" — strictly worse. A comment at
   `_postselect_renorm` points here.

3. **No skip-`optimizer.step()`-on-non-finite-gradients guard**, for the same
   reason: skipping the step masks the failure (the head survives, the run
   silently continues minus one batch) and perturbs the trajectory in a way
   that is invisible in the results.

Rejected alternatives, recorded because each looks like an improvement out of
context:

- `nan_to_num` before the clamp in `_postselect_renorm` (one line, makes the
  containment guard "complete") — rejected: with sources fixed, it only ever
  absorbs *unknown future bugs*, exactly the ones that must stay loud.
- Skip-step guard — rejected as above.
- Repelling hypernet outputs away from zero (`|θ| ≥ ε` clips) — rejected:
  θ = 0 / α = 0 are physically healthy points (identity gates) that trained
  hypernets demonstrably occupy and cross; excluding them changes the model to
  dodge what was purely a computational artifact.
- Iterated multiplication for all displacement powers — rejected in favour of
  the surgical exponent floor: keeps PyTorch's pow for every entry it computes
  correctly (bit-identical values for s ≥ 2), avoids reimplementing powers.

Consequence for future changes: any edit that sanitises NaN/Inf between the
quantum engine and the loss (renorm, readout, loss assembly) must revisit this
ADR and say what now makes the failure visible.

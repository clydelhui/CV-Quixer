# CV-Quixer

A hypernetwork-driven Continuous-Variable Quantum Vision Transformer. This glossary
pins the language of the quantum head so terms stay consistent across the model code,
experiments, and the thesis write-up.

## Language

**Per-patch unitary** (`U_i`):
The input-dependent Killoran-style gate sequence (`S → BS → R → D → K`) applied to one
patch's token, with gate parameters emitted by the CNN hypernetwork. One `U_i` per patch.
_Avoid_: patch circuit, token unitary.

**LCU**:
The linear combination of per-patch unitaries `M = Σ_i b_i U_i`, with trainable complex
coefficients `b_i`. Not materialised as a matrix.
_Avoid_: mixing, attention matrix.

**Polynomial** (`P(M)`):
The real-coefficient matrix polynomial `P(M) = Σ_j c_j Mʲ` applied to the input state,
modelling post-selected QSVT. Followed by post-selection renormalisation.
_Avoid_: QSVT (that is the technique `P(M)` models, not the object).

**CVQNN block** (`W`):
A fixed, per-image, trainable Killoran-style variational circuit applied to the
post-polynomial state before observable readout. Its gate parameters are owned
`nn.Parameter`s (input-independent), distinct from the hypernetwork-emitted `U_i`.
One `W` per head. Each `W` layer is the *canonical* (full two-interferometer)
Killoran form `U₁ → S → U₂ → D → Φ`, unlike `U_i` which drops the leading
interferometer `U₁` (trivial on the vacuum that `U_i` first acts on).
_Avoid_: variational layer, ansatz, output circuit.

**Success probability**:
The probability that the heralded LCU/QSVT implementation of `P(M)` post-selects
successfully: `‖P(M)|ψ⟩‖² / λ²`, measured in the truncated simulation (so a small part
of the norm deficit is Fock truncation, not heralding failure). Applies only to the
Polynomial step — the CVQNN block `W` is unitary and has no success probability; its
norm deficit is truncation leakage.
_Avoid_: state norm (that is the unnormalised numerator), post-selection rate.

**Subnormalisation** (`λ`):
The block-encoding scale factor of the heralded polynomial,
`λ = Σ_j |c_j| · (Σ_i |b_i|)ʲ` — the nested-LCU normalisation of `P(M)` built from the
LCU of unitaries `M`. Derivable from the learned coefficients; one scalar per head.
_Avoid_: normalisation constant (ambiguous with state renormalisation).

**Head**:
One `(LCU + Polynomial + CVQNN block + readout)` pipeline. The model runs `num_heads`
independent heads in parallel and concatenates their readouts.
_Avoid_: channel, attention head (it is CV, not dot-product attention).

**Readout**:
The vector of observable expectation values (`⟨x̂⟩`, `⟨p̂⟩`, `⟨n̂⟩`, …) measured on the
final per-head state and fed to the classical decoder.
_Avoid_: measurement, output (too generic).

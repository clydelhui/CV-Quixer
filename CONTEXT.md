# CV-Quixer

A hypernetwork-driven Continuous-Variable Quantum Vision Transformer. This glossary
pins the language of the quantum head so terms stay consistent across the model code,
experiments, and the thesis write-up.

## Language

**Per-patch unitary** (`U_i`):
The input-dependent Killoran-style gate sequence (`S → BS → R → D → K`) applied to one
patch's token, with gate parameters emitted by the CNN hypernetwork. One `U_i` per patch.
_Avoid_: patch circuit, token unitary.

**Query unitary** (`U_{q,i}`):
The input-dependent gate sequence that prepares position i's query state
`U_{q,i}|0⟩` in the seq-to-seq block. Emitted by the same hypernetwork embedding
as the per-patch unitary `U_i` (a second projection of the shared patch
embedding, mirroring classical attention's Q/K split). One `U_{q,i}` per patch.
_Avoid_: query circuit, query state (that is `U_{q,i}|0⟩`, not the unitary).

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

**Head**:
One `(LCU + Polynomial + CVQNN block + readout)` pipeline. Each block (or the
whole model, for the non-stacked variants) runs `num_heads` independent heads in
parallel and concatenates their readouts. In a seq-to-seq block a head emits one
readout per position (one query state each); elsewhere it emits a single readout.
_Avoid_: channel, attention head (it is CV, not dot-product attention).

**Seq-to-seq block**:
One CV-Quixer attention stage that maps an N-token sequence to an N-token
sequence. All positions share each head's LCU, polynomial, and CVQNN block;
position i differs only in its query state — its output token is the readout of
`W · P(M) · U_{q,i}|0⟩`. Stackable; the first block consumes raw patches, deeper
blocks consume the previous block's tokens.
_Avoid_: attention layer, mixer block, token block.

**Aggregator block**:
The optional final seq-to-one stage of a stacked model: a canonical CV-Quixer
head (vacuum input, no query unitaries) consuming the last seq-to-seq block's
token sequence and emitting a single readout vector. The alternative to
mean-pooling over positions.
_Avoid_: pooling block, readout block.

**Readout**:
The vector of observable expectation values (`⟨x̂⟩`, `⟨p̂⟩`, `⟨n̂⟩`, …) measured on the
final per-head state and fed to the classical decoder.
_Avoid_: measurement, output (too generic).

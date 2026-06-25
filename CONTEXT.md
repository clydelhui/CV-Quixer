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

**Positional encoding** (`none` / `1d` / `2d`):
The fixed signal added to a patch's hypernetwork features so the [[per-patch
unitary]] it emits can depend on *where* the patch sits, not only on its pixels.
Three variants: `2d` (the default) encodes the patch's (row, column) grid
position; `1d` encodes its position in the flattened patch sequence (no grid
awareness); `none` adds no positional signal. It enters only at the first stage
that consumes raw patches — so in a stacked model, only the first
[[seq-to-seq block]] carries it.
_Avoid_: positional embedding (it is a fixed sinusoid, not a learned
parameter), patch index (that is only the `1d` ordinate, not the encoding).

**Coefficient ablation** (`none` / `lcu` / `lcu_poly`):
An ablation that *freezes* a head's learned combination coefficients to fixed
uniform values, removing learned weighting structure while leaving the
hypernetwork-emitted [[per-patch unitary]]s and the [[CVQNN block]] `W`
trainable. Three cumulative levels: `none` (the default — `bᵢ` and `cⱼ` trained
as normal); `lcu` (freeze the [[LCU]] coefficients to `bᵢ = 1/N`, a fixed
uniform average `M = (1/N) Σᵢ Uᵢ` with no per-position weighting); `lcu_poly`
(additionally freeze the [[polynomial]] coefficients to all-ones,
`P(M) = Σⱼ Mʲ`). The frozen scalars are deliberately *not* trainable: a single
trainable scalar would be gauge-redundant — a global LCU scalar folds into the
trainable `cⱼ` (`(b·ΣUᵢ)ʲ = bʲ(ΣUᵢ)ʲ`), and a global polynomial scalar cancels
in the post-selection renormalisation (so it is [[success probability]]-/trunc
only, readout-inert). The freeze therefore targets the *weighting pattern*, not
a magnitude. Applies uniformly across all heads of every model variant (the
coefficients live in the shared head base), including any [[aggregator block]].
_Avoid_: coefficient sharing (the values are frozen, not merely tied, and
"shared" collides with the `quantum_shared` model), uniform-predictor collapse
(a training failure, not a knob — though `lcu_poly` structurally precludes the
`P(M)=I` init that drives it).

**Uniform-predictor collapse**:
A training failure in which a run stays pinned at the trivial constant predictor —
`train_loss = ln(num_classes)` (≈2.3026 for 10 classes) and accuracy at chance from
epoch 1, never escaping. Mechanistically a *plateau-escape race*: the polynomial
coefficient init `c = [1, 0, …]` makes `P(M) = I`, so the readout is input-independent
at initialisation and the loss is structurally pinned at `ln(num_classes)`; whether a
run escapes depends on whether the quantum-circuit gradient self-amplifies before it
decays below the noise floor. Driven by readout/decoder-side knobs (`poly_degree`,
`decoder_num_layers`, `num_heads`), *not* by the Hilbert-space knobs (`num_modes`,
`cutoff_dim`, `cvqnn_num_layers`).
_Avoid_: barren plateau (the quantum Hilbert-space phenomenon, which this is
explicitly NOT — `num_modes`/`cutoff` are not drivers), dead run, NaN/OOM collapse
(those are separate, loud failures).

**Re-roll**:
A *fresh-from-scratch* re-run of a selected subset of a sweep's runs into the same
sweep directory, with one knob changed — built to retry runs that hit
[[uniform-predictor collapse]] under a symmetry-breaking init (`--poly-init-noise`).
Unlike a [[top-up]], a re-roll never resumes a checkpoint (the checkpoint *is* the
collapsed state); it restarts from the seeded init so the changed knob takes effect.
Each re-roll dir carries an explicit `reroll__` name prefix (human-visible
identification, distinct from the `poly_init_noise` coordinate a future sweep might
vary as an ordinary axis) and records a reference to the original run it derives
from (`history["meta"]["reroll_of"]`); reporting tools pair the two by that
reference — not by name-stripping, since the changed knob need not be the one in
the dir name. A re-roll is compared against its original at the max-common epoch
count.
_Avoid_: re-run (ambiguous — the whole sweep, or a top-up), top-up (that resumes;
a re-roll restarts), retry (suggests an identical relaunch, not a changed knob).

**Top-up**:
Raising selected runs of a sweep to a shared *target total* epoch count — resuming
each run from its latest checkpoint where one exists, restarting from scratch where
none does, and leaving runs already at the target untouched. A topped-up run is
statistically equivalent to, but not bit-identical with, an uninterrupted run of the
same length (data-order RNG is not checkpointed).
_Avoid_: re-run (suggests from scratch), extend / additional epochs (suggests an
additive count — the target is a total).

**Invocation**:
One launch of an experiment entry-point script (sweep, run, eval, or whole-sweep
eval): the exact command together with its launch context — when it ran, on which
host, at which code revision, and (for SLURM launches) what it submitted to the
queue. An artefact can accumulate several invocations — the launch that created it
plus each resume or top-up — and records all of them in order.
_Avoid_: command (only the argv string — an invocation includes its context),
launch (ambiguous with the SLURM submission it may contain).

**Run manifest**:
The JSON plan an orchestrator (`sweep.py`, `resume_sweep.py`,
`eval_cutoff_sweep_all.py`) writes to launch a batch of runs: a `runs[]` list —
each entry a dense `index` (the SLURM array task id), a `run_name`, and the
`args` argv replayed verbatim by the entry-point script — plus `n_runs` and the
launch invocations. It is the seam between the Python orchestrators and the
shell job-array scripts (`run_sweep.sh`, `run_eval_cutoff_sweep_array.sh`),
which select their task by `index` and print each `args` element on its own
line. All three orchestrators share this schema, so one array script consumes
any of their manifests; the shared launch helpers live in
`experiments/_orchestration.py`.
_Avoid_: config (that is the per-run ExperimentConfig), grid (only the sweep's
axes, not the launch plan it expands into).

**Configuration identity**:
The full set of sweep coordinates that makes two runs "the same experiment
repeated" — model variant, observable preset, budget fields, and every
architecture knob — everything except the training seed. Cross-run reports
seed-average only within one configuration identity; runs differing in it are
never averaged together.
_Avoid_: grid point (the manifest's view of it), run group.

**Coordinate filter**:
A selection of runs by matching a subset of their configuration coordinates
against allowed value sets — within one coordinate the allowed values are OR'd,
across coordinates they are AND'd (e.g. `num_modes ∈ {2,3}` *and*
`num_heads ∈ {5,10}`). Shared by the top-up and reporting tools so the same
filter selects the same runs in both; a run whose value for a filtered
coordinate is absent or unresolvable is excluded with a warning, never matched.
_Avoid_: subset filter, run mask, query.

**All-else-equal trend line**:
In an accuracy-versus-one-coordinate figure, the set of configurations that agree
on every *independent* sweep coordinate except the one on the x-axis — the runs
that isolate the effect of varying that single coordinate. A sweep that varies
several coordinates produces one such trend line per combination of the *other*
varying independent coordinates; each is plotted as its own connected series.
Dependent coordinates are deliberately not held fixed (see below), so they are
free to follow the x-axis coordinate along the line.
_Avoid_: series (the generic colour/legend grouping, which may be coarser),
sweep axis (the coordinate itself, not the runs along it).

**Dependent coordinate**:
A configuration coordinate whose value is a deterministic function of the
independent sweep axes rather than a freely chosen knob — e.g. the decoder hidden
width when it is sized as a multiple of the model's readout width (itself set by
the head count and mode count). A dependent coordinate co-varies with the axes
that drive it, so it is excluded from the "all else equal" comparison: holding it
fixed would make it impossible to vary any of its driving axes alone, collapsing
every trend line to a single point.
_Avoid_: derived field (implementation phrasing), hyperparameter (suggests an
independently chosen knob, which a dependent coordinate is not).

**Artefact tier**:
A named, nested subset of a run's output artefacts, ordered by how much local
disk it costs to pull from the cluster. The ladder is
`figures` ⊂ `light` ⊂ `excl_train_ckpt` ⊂ `full`: `figures` is the derived
outputs only (tables + every `.png`), `light` (the default pull) adds the small
text/raw artefacts that drive local re-reporting (`config.json`, `history.json`,
`subset_indices.npz`, `debug/`), `excl_train_ckpt` adds the evaluation-payload
npz needed to re-derive test-side figures (test `predictions`, `diagnostics`,
`test_images`) but not the *training payload*, and `full` is a complete mirror.
The training payload — `checkpoints/` and the per-epoch `_train.npz` — is the
heaviest artefact and is only needed to resume training or re-derive train-side
figures, so it lands only in `full`.
_Avoid_: artefact level (ambiguous with log levels), bundle (suggests a single
archive — a tier is a filter over the live tree, not a tarball).

**Epoch artefacts**:
The per-epoch raw output payload an experiment writes for one evaluation pass:
the test (and optionally train) prediction arrays — `y_true`, `y_pred`,
`y_probs`, `readouts`, and `success_probs` where the model post-selects —
together with the quantum-diagnostics arrays (gate-param samples, state norms,
mean photon number) and the coefficient snapshots (`lcu_coeffs`/`poly_coeffs`,
plus `cvqnn_params` and the block-prefixed stacked keys where present). One
bundle per epoch, persisted as `predictions/epoch_NNNN[_train].npz` +
`diagnostics/epoch_NNNN.npz`. The live writers (`full_experiment`,
`eval_cutoff_sweep`) produce it through one shared module, and
`report_diagnostics` consumes it; the bundle, its model-variant key set, and its
on-disk layout are the single contract between them. (`backfill_artefacts` is a
frozen, archived writer that predates the shared module and emits the older
canonical-only key set — deliberately not migrated.)
_Avoid_: artefact (too generic — qualify as epoch artefacts), diagnostics (only
the quantum-diagnostics arrays, not the predictions), Artefact tier (a pull-time
subset of a whole run, not one epoch's payload).

# Re-rolls restart collapsed runs in-place, paired to their original by reference

A subset of sweep runs hit **uniform-predictor collapse** (CONTEXT.md): the
polynomial coefficient init `c = [1, 0, …]` makes `P(M) = I`, so the readout is
input-independent at initialisation and the loss is pinned at `ln(num_classes)`;
whether a run escapes is a race between the quantum-circuit gradient
self-amplifying and decaying below the noise floor. The mitigation is an optional
symmetry-breaking init (`--poly-init-noise`, seeding `c_{j≥1} ≠ 0`). We need to
retry exactly the collapsed runs under that init and measure the change in escape
rate. This is the **re-roll** operation (CONTEXT.md). Four decisions shaped it.

## A re-roll restarts from scratch; it never resumes

The existing top-up tool (`resume_sweep.py`) resumes each run from `latest.pt`.
For a collapsed run that is exactly wrong: `latest.pt` *is* the collapsed state,
and the changed init never takes effect on a resumed model. A re-roll therefore
**always restarts fresh** (seeded init, no `--resume`), which is the opposite
checkpoint semantics from a top-up. That opposition is why a re-roll is a
**separate orchestrator** (`rerun_sweep.py`) rather than a flag on
`resume_sweep.py` — bolting two opposite checkpoint contracts onto one tool would
muddy the well-defined top-up semantics. The two still share the manifest schema,
launch helpers (`_orchestration.py`), and run-selection machinery
(`_run_selection.py`).

## Re-rolls live in the original sweep directory, compared in-place by a flag

A re-roll writes into the **same sweep dir** as the runs it retries, not a fresh
sweep. The alternative — a self-contained re-roll sweep that re-runs each config
twice (a noise-off control arm + a noise-on arm) for one apples-to-apples figure
— was rejected on **compute cost**: the original collapsed runs already exist and
a fresh noise-off restart at the same seed reproduces the collapse deterministically,
so re-running the control arm only burns GPU reproducing a known result. Instead
`report_sweep.py` gains a `--rerolls {ignore,replace,compare}` flag: `ignore`
(default) drops re-roll dirs so the existing full-sweep report is byte-unchanged;
`replace` substitutes each re-roll for its original; `compare` plots the paired
configs (original vs re-roll) at the **global max-common epoch count** so the
comparison is fair despite the two arms being measured at different times. The
cost of option A is that the baseline is the *original* run, measured under its
own epoch budget — hence the max-common-epoch cap rather than an assumption of
equal horizons.

## The re-roll marker is kept orthogonal to the `poly_init_noise` coordinate

`poly_init_noise` is registered as an ordinary configuration coordinate
(`FILTERABLE_FIELDS` / `CONFIG_IDENTITY_FIELDS`), so a *future* sweep that varies
it as a first-class axis groups and plots like any other knob. That means "has
`poly_init_noise` set" cannot identify a re-roll — such a future sweep member
would carry the same `__pin{eps}` marker without being a re-roll of anything. So
re-roll-ness is a matter of **provenance**, marked explicitly and separately: each
re-roll dir carries a `reroll__` name prefix (human-visible in any `ls`, and the
signal `report_sweep` keys on). Conflating the two would misclassify every future
poly-init sweep member as a re-roll.

## Pairing is by recorded reference, not by name-stripping

A re-roll pairs to the original it derives from. The tempting approach — strip the
`reroll__` prefix and the `__pin{eps}` marker from the dir name to recover the
original name — silently assumes `poly_init_noise` is the *only* changed knob. But
a re-roll changes one knob in general, and the better mitigations may be
`poly_degree≥3` or `decoder_num_layers=2` (handoff §5), whose markers (`__pd3`,
`__dnl2`) differ from the original's — name-stripping a fixed `__pin` suffix would
fail to pair them. Instead each re-roll records an explicit reference to its
original in `history["meta"]["reroll_of"]` (set by a new `--reroll-of` flag on
`full_experiment.py`, injected by `rerun_sweep.py`, also mirrored in the rerun
manifest). `report_sweep` pairs by that reference, robust to whichever knob moved;
the changed coordinate is then discoverable by diffing the pair's configs, so the
`compare` figure can plot along whatever actually changed rather than hard-wiring
the `poly_init_noise` axis.

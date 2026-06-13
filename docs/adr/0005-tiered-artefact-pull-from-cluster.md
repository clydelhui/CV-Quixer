# Tiered, additive artefact pull from the cluster

Status: accepted

Context: a full sweep is ~23 GB locally and dominated almost entirely by raw
`.npz`. Per run (~300 MB), the breakdown is ~95% `predictions/` +
`diagnostics/` npz, of which the per-epoch *train-side* `_train.npz` alone is
~94 MB/epoch — an order of magnitude larger than everything else combined.
Rendered figures are ~2.3 MB/run and the text artefacts (`config.json`,
`history.json`, …) are sub-MB. A plain `rsync -avz` of `results/` therefore
spends nearly all its time and local disk on artefacts that are only inputs to
figures, not the figures themselves.

The premise that unlocks the design: `report_sweep.py` (and the per-run
`report_diagnostics.py` suite it drives) now runs **on the cluster**
(`scripts/submit_report.sh` et al.), writing `summary.csv/md` + `figures/`
in-place into the sweep dir. So every derived analysis output already exists
remotely; the only question is how much *raw* data has to follow the figures
down for the analysis the user still does locally (re-plotting, `--max-epoch`
re-reports, ad-hoc metric recompute).

Decision:

1. **A four-rung artefact tier ladder, `figures` ⊂ `light` ⊂ `excl_train_ckpt`
   ⊂ `full`** (see CONTEXT.md "Artefact tier"), pulled by a new local-side
   `scripts/pull_results.sh <repo-relative-path>… [--tier T]`. `light` is the
   default: figures + the small text/raw artefacts that let `report_sweep` /
   `report_diagnostics` re-run locally. The split point that names the top two
   rungs is the **training payload** (`checkpoints/` + `_train.npz`): the
   ~94 MB/epoch train npz is singled out because it is the entire reason a pull
   is slow, and it is needed *only* to resume training or re-derive train-side
   figures — neither of which the cluster-report workflow does locally.

2. **The tool is the repo's first local-side script.** Every other
   `scripts/*.sh` runs on the cluster; this one runs on the laptop and rsyncs
   *from* the cluster. The remote is read from a gitignored `scripts/.pull_config`
   (`REMOTE=user@host`, `REMOTE_ROOT=~/CV-Quixer`); a tracked
   `.pull_config.example` documents the format. The target is given as the
   repo-relative path (`results/sweeps/<sweep>/`), identical on both machines
   because the cluster checkout mirrors the local layout — the same path that
   `submit_report.sh` already takes.

3. **Pull is additive; never `rsync --delete`.** A re-pull can only add files,
   so pulling `light` after `excl_train_ckpt` keeps the heavy npz already on
   disk. Space reclaim is a separate, intent-named `--prune` flag that deletes
   only the artefacts in tiers *above* the chosen one, by an explicit whitelist
   of heavy patterns — it is structurally incapable of touching figures,
   tables, config, or any unrecognised local file.

4. **rsync flags fixed to `-aP --info=progress2`, no `-z`.** `-a` preserves
   mtimes so re-pulls are incremental; `-P` resumes interrupted big-file
   transfers; one overall progress bar. Compression is omitted because `.png`
   and `.npz` are already dense and the `light` default is tiny anyway.
   `--dry-run` is a passthrough.

Rejected alternatives, each recorded because it looks reasonable out of context:

- **Plain `rsync -avz results/`** — the status quo. Rejected: pulls the
  ~94 MB/epoch train npz that local analysis never reads, and `-z` burns CPU on
  already-compressed data.
- **Pull everything, then prune locally** — rejected: defeats the point; the
  cost is the *transfer*, not the resident bytes, and the train npz is exactly
  what you do not want to transfer.
- **Generate figures locally instead of on the cluster** — rejected by the
  premise: that requires the raw npz locally (the slow figures need test/train
  predictions), which is the transfer we are avoiding. Cluster CPU report jobs
  are free against the GPU budget.
- **`--mirror` (rsync `--delete` scoped to the current tier) for space
  reclaim** — rejected in favour of `--prune`: `--delete` works from a
  blacklist (remove everything not in the current tier's file list), so a
  careless `--tier light --mirror` silently deletes previously-pulled heavy npz
  *and* any hand-made local file in the run dir. `--prune` works from a
  whitelist of the heavy artefacts above the chosen tier and cannot.
- **Env-var or full-remote-path interface** instead of a config file —
  rejected: the user pulls often and wanted the host to persist across shells
  without re-exporting or re-typing the `user@host:~/CV-Quixer/...` prefix.

Consequence for future changes: any new heavy artefact written by
`full_experiment.py` must be slotted into a tier in `pull_results.sh` (and, if
it belongs to the training payload, added to the `--prune` whitelist),
otherwise it silently rides along in `full` only and is invisible to the
`light`/`excl_train_ckpt` analysis workflow. If figure generation ever moves
back to the laptop, rung (3)'s premise — that derived outputs already exist
remotely — no longer holds and this ADR must be revisited.

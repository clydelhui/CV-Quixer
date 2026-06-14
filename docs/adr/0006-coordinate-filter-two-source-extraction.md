# Coordinate-filter matching is shared, but its coordinate extraction is two-source

A **coordinate filter** (CONTEXT.md) selects a subset of runs by matching a
subset of their configuration coordinates (`num_modes`, `num_heads`, …) against
allowed value sets. Both the top-up tool (`resume_sweep.py`) and the reporting
tool (`report_sweep.py`) need it, and they must select the *same* runs for the
same filter so a "top up subset X → plot subset X" workflow is coherent. We
therefore share the field registry + matching predicate + argparse wiring in
`experiments/_run_selection.py` (and `report_sweep.py` derives its
`CONFIG_IDENTITY_FIELDS`/`ARCH_META_FIELDS` from that registry rather than
keeping a parallel copy — the drift its `_check_identity_drift` guard exists to
prevent).

We deliberately did **not** share *coordinate extraction*, because the two tools
see different data:

- `resume_sweep.py` reads `sweep_manifest.json`, whose entries carry only the
  raw replayed `args` (explicit flags — *not* resolved values), and it must
  still match runs that died before epoch 1 (the "fresh" retry path), which have
  **no `history.json` and no `config.json`**. So it extracts from `config.json`
  when present (resolved, catches auto-scaled knobs) and falls back to parsing
  manifest `args` otherwise.
- `report_sweep.py` already loads each run's resolved coordinates from
  `history["meta"]` and only ever considers runs with ≥1 completed epoch.

Consequence: a budget-mode run whose `num_heads` was auto-scaled has the
resolved value in `config.json`/`meta` but **not** in its manifest `args`. A
coordinate filter on such a never-started run can only see explicit args, so an
auto-scaled coordinate is unresolvable there — those runs are **excluded with a
warning**, never silently matched. This is the price of keeping the never-started
retry path filterable, and is why the two tools read coordinates from different
sources for the same filter.

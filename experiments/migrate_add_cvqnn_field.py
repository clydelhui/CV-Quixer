#!/usr/bin/env python
"""Migrate pre-CVQNN run configs so they still load as W-free (pre-W) models.

The CVQNN block W (``QuantumConfig.cvqnn_num_layers``, default 1) post-dates the
original frozen model. The canonical default now builds *with* W, which is
checkpoint-incompatible with every run trained before it. A run with
``cvqnn_num_layers=0`` is, by construction, byte-identical to the pre-W model
(no W params registered), so old checkpoints load cleanly against it — the only
gap is the saved ``config.json``, which has no ``cvqnn_num_layers`` key and would
otherwise be rebuilt with W on (dacite fills the *new* default of 1).

This one-shot script walks every ``config.json`` under the given roots and bakes
``cvqnn_num_layers: 0`` and ``cvqnn_trunc_lambda: 0.0`` into each run's
``quantum`` block, so the run reloads as a pre-W model. Idempotent: a config that
already has ``cvqnn_num_layers`` is left untouched.

The companion loud guard in ``cv_quixer/config/utils.experiment_config_from_dict``
refuses to load an un-migrated config (with a hint to run this script) rather than
silently defaulting — so forgetting to migrate fails loudly, not silently.

Usage:
    uv run python experiments/migrate_add_cvqnn_field.py --runs-root results/
    uv run python experiments/migrate_add_cvqnn_field.py \
        --runs-root results/runs results/sweeps --dry-run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# The values that reproduce a pre-W (W-free) model.
_CVQNN_NUM_LAYERS_PREW = 0
_CVQNN_TRUNC_LAMBDA_PREW = 0.0


def _iter_config_jsons(roots: list[Path]):
    """Yield every config.json found anywhere under each root (recursive)."""
    seen: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("config.json"):
            rp = path.resolve()
            if rp not in seen:
                seen.add(rp)
                yield path


def migrate_config(path: Path, *, dry_run: bool) -> str:
    """Migrate one config.json in place. Returns a status: 'migrated' |
    'skipped' (already has the key) | 'no-quantum' (unexpected shape)."""
    with open(path) as f:
        raw = json.load(f)

    quantum = raw.get("quantum")
    if not isinstance(quantum, dict):
        return "no-quantum"
    if "cvqnn_num_layers" in quantum:
        return "skipped"

    quantum["cvqnn_num_layers"] = _CVQNN_NUM_LAYERS_PREW
    quantum.setdefault("cvqnn_trunc_lambda", _CVQNN_TRUNC_LAMBDA_PREW)

    if not dry_run:
        with open(path, "w") as f:
            json.dump(raw, f, indent=2)
    return "migrated"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root",
        type=Path,
        nargs="+",
        default=[Path("results")],
        help="One or more roots to scan recursively for config.json "
        "(default: results/). e.g. results/runs results/sweeps",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would change without writing any files.",
    )
    args = parser.parse_args()

    counts = {"migrated": 0, "skipped": 0, "no-quantum": 0}
    for path in _iter_config_jsons(args.runs_root):
        status = migrate_config(path, dry_run=args.dry_run)
        counts[status] += 1
        if status == "migrated":
            verb = "would migrate" if args.dry_run else "migrated"
            print(f"  {verb}: {path}")
        elif status == "no-quantum":
            print(f"  WARNING no 'quantum' block, skipped: {path}")

    prefix = "[dry-run] " if args.dry_run else ""
    print(
        f"\n{prefix}Done. migrated={counts['migrated']} "
        f"skipped(already had key)={counts['skipped']} "
        f"no-quantum={counts['no-quantum']}"
    )


if __name__ == "__main__":
    main()

"""Re-roll selected sweep runs: fresh restart with a symmetry-breaking init.

See CONTEXT.md "Re-roll" and ADR-0007. A subset of runs hit *uniform-predictor
collapse* (CONTEXT.md): the polynomial-coefficient init c = [1, 0, …] makes
P(M) = I, the readout is input-independent at init, and the loss is pinned at
ln(num_classes). This tool retries exactly those runs under ``--poly-init-noise``
(which seeds c_{j>=1} != 0) so the symmetry is broken from step 0.

A re-roll is NOT a top-up (``resume_sweep.py``): it must **restart from scratch**,
because the collapsed ``latest.pt`` *is* the failure — resuming it would re-load
the collapsed weights and the new init would never take effect. So each re-roll
replays the original ``full_experiment.py`` argv from ``sweep_manifest.json`` with:

  * ``--poly-init-noise <eps>`` injected (the changed knob),
  * ``--run-name`` rewritten to ``reroll__<original>__pin<eps>`` (explicit
    provenance marker, lands in the same sweep dir, distinct from an ordinary
    poly-init sweep member),
  * ``--reroll-of <original>`` injected (recorded into ``history["meta"]`` so
    ``report_sweep --rerolls`` pairs the re-roll with its original by reference,
    robust to whichever knob changed), and
  * NO ``--resume``. ``--epochs`` is left at the original's value unless
    ``--epochs`` overrides it.

Selection (all AND'd): ``--runs-file`` (a ``low_accuracy_runs.txt``-format include
list — the verified collapsed runs), ``--runs`` fnmatch patterns, and the shared
coordinate filter. At least one selector is required — there is no "re-roll
everything" default. A re-roll whose target dir already exists is skipped
(idempotent re-launch).

The plan is written to ``<sweep-dir>/rerun_manifest_<ts>.json`` (the runs[] schema
``scripts/run_sweep.sh`` consumes unchanged); the original ``sweep_manifest.json``
is never modified.

Examples
--------
Re-roll the verified collapsed runs at eps=0.05 (inspect only)::

    uv run python experiments/rerun_sweep.py \\
        --sweep-dir results/sweeps/grid_quantum_<ts>/ \\
        --runs-file results/sweeps/grid_quantum_<ts>/low_accuracy_runs.txt \\
        --poly-init-noise 0.05 --dry-run

Re-roll just the poly_degree=1 collapses as a SLURM array::

    uv run python experiments/rerun_sweep.py \\
        --sweep-dir results/sweeps/grid_quantum_<ts>/ \\
        --runs-file .../low_accuracy_runs.txt --poly-degree 1 \\
        --poly-init-noise 0.05 0.1 --launch slurm
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from fnmatch import fnmatch
from pathlib import Path

from _orchestration import launch_local, submit_slurm_array
from _run_selection import (
    REROLL_PREFIX,
    add_filter_args,
    coords_from_args,
    coords_from_config_json,
    parse_filter_args,
    read_run_names_file,
    run_matches,
)

from cv_quixer.provenance import invocation_record

FULL_EXPERIMENT = "experiments/full_experiment.py"
RUN_SWEEP_SH = "scripts/run_sweep.sh"


def reroll_run_name(original_run_name: str, eps: float) -> str:
    """The re-roll dir name: ``reroll__<original>__pin<eps>`` (CONTEXT.md)."""
    return f"{REROLL_PREFIX}{original_run_name}__pin{eps}"


def rewrite_run_args(
    args: list[str], original_run_name: str, eps: float,
    target_epochs: int | None,
) -> list[str]:
    """Original argv rewritten for a fresh re-roll (never resumed).

    Injects --poly-init-noise / --reroll-of, rewrites --run-name to the re-roll
    name, optionally rewrites --epochs, and deliberately appends no --resume.
    """
    out = list(args)

    def _set(flag: str, value: str) -> None:
        if flag in out:
            out[out.index(flag) + 1] = value
        else:
            out.extend([flag, value])

    _set("--run-name", reroll_run_name(original_run_name, eps))
    _set("--poly-init-noise", str(eps))
    _set("--reroll-of", original_run_name)
    if target_epochs is not None:
        _set("--epochs", str(target_epochs))
    return out


def _run_coords(run_dir: Path, args: list[str]) -> dict:
    """Resolve a run's filterable coordinates (config.json wins over argv; ADR-0006)."""
    coords = coords_from_args(args)
    config_path = run_dir / "config.json"
    if config_path.is_file():
        try:
            with open(config_path) as f:
                coords.update(coords_from_config_json(json.load(f)))
        except (json.JSONDecodeError, OSError):
            pass
    return coords


def build_manifest(
    sweep_dir: Path,
    poly_init_noises: list[float],
    *,
    runs_file: Path | None = None,
    patterns: list[str] | None = None,
    filters: dict[str, set] | None = None,
    target_epochs: int | None = None,
) -> dict:
    """Plan the re-rolls for the selected runs × each poly_init_noise value.

    Selection narrows by the include file (``runs_file``), fnmatch ``patterns``,
    and the coordinate ``filters`` — all AND'd. At least one selector is required
    (raises ValueError otherwise — there is no "re-roll everything" default). A
    re-roll whose target dir already exists is recorded in ``skipped``.
    """
    filters = filters or {}
    if runs_file is None and patterns is None and not filters:
        raise ValueError(
            "a re-roll needs an explicit selection — pass --runs-file, --runs, "
            "or a coordinate filter (there is no 're-roll everything' default)."
        )
    include = read_run_names_file(runs_file) if runs_file is not None else None

    with open(sweep_dir / "sweep_manifest.json") as f:
        source = json.load(f)

    entries: list[dict] = []
    skipped: list[dict] = []
    for run in source["runs"]:
        name = run["run_name"]
        if include is not None and name not in include:
            continue
        if patterns is not None and not any(fnmatch(name, p) for p in patterns):
            continue
        if filters and not run_matches(
            _run_coords(sweep_dir / name, run["args"]), filters, run_name=name,
        ):
            continue
        for eps in poly_init_noises:
            new_name = reroll_run_name(name, eps)
            if (sweep_dir / new_name).exists():
                skipped.append({"run_name": new_name, "reason": "dir exists"})
                continue
            entries.append({
                "index": len(entries),
                "run_name": new_name,
                "reroll_of": name,
                "poly_init_noise": eps,
                "args": rewrite_run_args(run["args"], name, eps, target_epochs),
            })

    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        # Launch provenance (CONTEXT.md: Invocation).
        "invocations": [invocation_record()],
        "sweep_dir": str(sweep_dir),
        "source_manifest": str(sweep_dir / "sweep_manifest.json"),
        "poly_init_noises": list(poly_init_noises),
        "target_epochs": target_epochs,
        "runs_file": str(runs_file) if runs_file is not None else None,
        "patterns": list(patterns) if patterns is not None else None,
        "filters": {k: sorted(v) for k, v in filters.items()},
        "n_runs": len(entries),
        "runs": entries,
        "skipped": skipped,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-roll selected sweep runs: fresh restart with "
        "--poly-init-noise to escape uniform-predictor collapse (ADR-0007).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--sweep-dir", type=Path, required=True,
                        help="sweep directory written by experiments/sweep.py")
    parser.add_argument("--poly-init-noise", type=float, nargs="+", required=True,
                        metavar="EPS",
                        help="one or more poly-init-noise std values (one re-roll "
                             "per selected run per eps)")
    parser.add_argument("--runs-file", type=Path, default=None,
                        help="include list of run names to re-roll "
                             "(low_accuracy_runs.txt format)")
    parser.add_argument("--runs", type=str, nargs="+", default=None,
                        metavar="PATTERN",
                        help="fnmatch pattern(s) on run_name to narrow selection")
    parser.add_argument("--epochs", type=int, default=None,
                        help="override the re-roll epoch count (default: the "
                             "original run's --epochs, replayed verbatim)")
    # poly_init_noise is this tool's own flag (the value to set), not a coordinate
    # to filter on — exclude it from the auto-generated filter flags.
    add_filter_args(parser, exclude={"poly_init_noise"})
    parser.add_argument("--launch", choices=["local", "slurm", "none"],
                        default="none",
                        help="local: run sequentially here; slurm: submit a job "
                             "array; none: just write the manifest")
    parser.add_argument("--dry-run", action="store_true",
                        help="alias for --launch none (write manifest + plan only)")
    args = parser.parse_args()

    if not args.sweep_dir.is_dir():
        parser.error(f"--sweep-dir does not exist: {args.sweep_dir}")
    if not (args.sweep_dir / "sweep_manifest.json").is_file():
        parser.error(f"no sweep_manifest.json under {args.sweep_dir}")
    if args.runs_file is not None and not args.runs_file.is_file():
        parser.error(f"--runs-file does not exist: {args.runs_file}")

    filters = parse_filter_args(args, exclude={"poly_init_noise"})
    if args.runs_file is None and args.runs is None and not filters:
        parser.error(
            "specify at least one selector: --runs-file, --runs, or a "
            "coordinate filter (no 're-roll everything' default).")

    manifest = build_manifest(
        args.sweep_dir, args.poly_init_noise,
        runs_file=args.runs_file, patterns=args.runs,
        filters=filters, target_epochs=args.epochs,
    )

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    manifest_path = args.sweep_dir / f"rerun_manifest_{timestamp}.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Re-roll of {args.sweep_dir} at eps={args.poly_init_noise}")
    print(f"  manifest: {manifest_path}")
    for run in manifest["runs"]:
        print(f"    [{run['index']}] {run['run_name']}")
    for run in manifest["skipped"]:
        print(f"    [-] {run['run_name']}: skip ({run['reason']})")
    if not manifest["runs"]:
        print("Nothing to do — no run matched the selection, or every target "
              "re-roll dir already exists.")
        return

    launch = "none" if args.dry_run else args.launch
    if launch == "local":
        failures = launch_local(manifest, FULL_EXPERIMENT)
        print(f"\nRe-roll finished: "
              f"{manifest['n_runs'] - failures}/{manifest['n_runs']} succeeded.")
        if failures:
            sys.exit(1)
    elif launch == "slurm":
        submit_slurm_array(manifest, manifest_path, RUN_SWEEP_SH)
    else:
        print("\n(manifest written; no runs launched — use --launch local|slurm)")


if __name__ == "__main__":
    main()

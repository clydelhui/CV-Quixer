"""Top up a sweep: raise selected runs to a target total epoch count.

See CONTEXT.md "Top-up". Reads the sweep's `sweep_manifest.json` and, for every
selected run, replays its original `full_experiment.py` argv (the args must be
replayed verbatim — full_experiment rebuilds its config from CLI flags even on
resume) with `--epochs` rewritten to the new *total* target (not additive: all
runs converge to the same epoch count for fair cross-run comparison) and
`--resume <run>/checkpoints/latest.pt` appended.

Per-run dispositions:

- **resume** — a checkpoint exists and the run is below the target.
- **fresh**  — no checkpoint (the run died before completing epoch 1, or never
  started): relaunched from scratch into the same run dir WITHOUT `--resume`.
  Safe — `latest.pt` is written every epoch, so its absence proves no per-epoch
  artefacts exist. This makes the tool double as a retry-failed-runs tool.
- **skip**   — already at/above the target (a resume would be a pure no-op
  burning an array slot); recorded in the manifest's `skipped` list.

The plan is written to `<sweep-dir>/resume_manifest_<ts>.json` — same `runs[]`
schema as `sweep_manifest.json` (re-indexed 0..K-1), so `scripts/run_sweep.sh`
consumes it unchanged. The original `sweep_manifest.json` is never modified.

Reproducibility caveat: `--resume` restores model + optimizer state but NOT RNG
state (only `torch.manual_seed(seed)` at startup), so a topped-up run sees
different batch shuffling in its new epochs than an uninterrupted run of the
same length — statistically equivalent, not bit-identical.

Examples
--------
Inspect the plan only::

    uv run python experiments/resume_sweep.py \\
        --sweep-dir results/sweeps/<sweep>_<ts>/ --epochs 6 --dry-run

Top up a subset to 6 total epochs as a SLURM array::

    uv run python experiments/resume_sweep.py \\
        --sweep-dir results/sweeps/<sweep>_<ts>/ --epochs 6 \\
        --runs 'p8000__*' --launch slurm
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from fnmatch import fnmatch
from pathlib import Path

from _orchestration import launch_local, submit_slurm_array

from cv_quixer.provenance import invocation_record

# Repo-relative entry / array scripts the launchers drive (see _orchestration).
# The resume manifest keeps sweep_manifest.json's runs[] schema (index,
# run_name, args), so the existing sweep array script consumes it unchanged.
FULL_EXPERIMENT = "experiments/full_experiment.py"
RUN_SWEEP_SH = "scripts/run_sweep.sh"


def current_epochs(run_dir: Path) -> int:
    """Completed epochs for a run — len(history["epoch"]["test_acc"]), 0 if none."""
    history_path = run_dir / "history.json"
    if not history_path.is_file():
        return 0
    with open(history_path) as f:
        history = json.load(f)
    return len(history.get("epoch", {}).get("test_acc") or [])


def rewrite_run_args(
    args: list[str], target_epochs: int, resume_ckpt: Path | None
) -> list[str]:
    """Original argv with --epochs set to the target and --resume appended."""
    out = list(args)
    if "--epochs" in out:
        out[out.index("--epochs") + 1] = str(target_epochs)
    else:
        out += ["--epochs", str(target_epochs)]
    if resume_ckpt is not None:
        out += ["--resume", str(resume_ckpt)]
    return out


def build_manifest(
    sweep_dir: Path, target_epochs: int, patterns: list[str] | None = None
) -> dict:
    """Plan the top-up for the selected runs in the sweep's manifest.

    ``patterns`` (fnmatch on run_name, any-match) narrows selection; None = all.
    """
    with open(sweep_dir / "sweep_manifest.json") as f:
        source = json.load(f)

    entries: list[dict] = []
    skipped: list[dict] = []
    for run in source["runs"]:
        if patterns is not None and not any(
            fnmatch(run["run_name"], p) for p in patterns
        ):
            continue
        run_dir = sweep_dir / run["run_name"]
        epochs = current_epochs(run_dir)
        if epochs >= target_epochs:
            skipped.append({"run_name": run["run_name"], "current_epochs": epochs})
            continue
        ckpt = run_dir / "checkpoints" / "latest.pt"
        if not ckpt.is_file():
            ckpt = None  # fresh start: no per-epoch artefacts exist to clobber
        entries.append({
            "index": len(entries),
            "run_name": run["run_name"],
            "action": "resume" if ckpt is not None else "fresh",
            "current_epochs": epochs,
            "args": rewrite_run_args(run["args"], target_epochs, ckpt),
        })

    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        # Launch provenance (CONTEXT.md: Invocation) — the exact
        # resume_sweep.py command that planned this top-up.
        "invocations": [invocation_record()],
        "sweep_dir": str(sweep_dir),
        "source_manifest": str(sweep_dir / "sweep_manifest.json"),
        "target_epochs": target_epochs,
        "patterns": list(patterns) if patterns is not None else None,
        "n_runs": len(entries),
        "runs": entries,
        "skipped": skipped,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Top up a sweep: raise selected runs to a target total "
        "epoch count (resume from latest.pt, restart checkpoint-less runs, "
        "skip runs already at the target).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--sweep-dir", type=Path, required=True,
                        help="sweep directory written by experiments/sweep.py")
    parser.add_argument("--epochs", type=int, required=True,
                        help="target TOTAL epoch count per run (not additive)")
    parser.add_argument("--runs", type=str, nargs="+", default=None,
                        metavar="PATTERN",
                        help="fnmatch pattern(s) on run_name to select a "
                             "subset (default: all runs in the manifest)")
    parser.add_argument("--launch", choices=["local", "slurm", "none"],
                        default="none",
                        help="local: run sequentially here; slurm: submit a "
                             "job array; none: just write the manifest")
    parser.add_argument("--dry-run", action="store_true",
                        help="alias for --launch none (write manifest + plan only)")
    args = parser.parse_args()

    if not args.sweep_dir.is_dir():
        parser.error(f"--sweep-dir does not exist: {args.sweep_dir}")
    if not (args.sweep_dir / "sweep_manifest.json").is_file():
        parser.error(f"no sweep_manifest.json under {args.sweep_dir}")

    manifest = build_manifest(args.sweep_dir, args.epochs, args.runs)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    manifest_path = args.sweep_dir / f"resume_manifest_{timestamp}.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Top-up of {args.sweep_dir} to {args.epochs} total epochs")
    print(f"  manifest: {manifest_path}")
    for run in manifest["runs"]:
        print(f"    [{run['index']}] {run['run_name']}: "
              f"{run['action']} {run['current_epochs']} -> {args.epochs}")
    for run in manifest["skipped"]:
        print(f"    [-] {run['run_name']}: "
              f"skip (already at {run['current_epochs']} >= {args.epochs})")
    if not manifest["runs"]:
        print("Nothing to do — every selected run is already at the target.")
        return

    launch = "none" if args.dry_run else args.launch
    if launch == "local":
        failures = launch_local(manifest, FULL_EXPERIMENT)
        print(f"\nTop-up finished: "
              f"{manifest['n_runs'] - failures}/{manifest['n_runs']} succeeded.")
        if failures:
            sys.exit(1)
    elif launch == "slurm":
        submit_slurm_array(manifest, manifest_path, RUN_SWEEP_SH)
    else:
        print("\n(manifest written; no runs launched — use --launch local|slurm)")


if __name__ == "__main__":
    main()

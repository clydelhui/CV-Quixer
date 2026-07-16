#!/usr/bin/env python3
"""Clone a curated run-list into a fresh qsvt-mode sweep (ADR-0009).

The extended-run configs (``results/extended_runs_25ep.txt``) are a heterogeneous
mix of ``quantum`` / ``quantum_shared`` / ``quantum_stacked`` runs with different
knob combinations, so they cannot be expressed as one Cartesian ``sweep.py`` grid.
This builder instead **replays each run's original argv** from its source
``sweep_manifest.json``, appends ``--poly-mode qsvt``, and starts the run **fresh
from scratch** at a chosen epoch count — *not* a resume: a qsvt run resumed from a
standard checkpoint would measure "switch to qsvt mid-training", not a clean qsvt
run (their state_dicts are byte-compatible, which would make that mistake silent).

Because the ``qsvt`` polynomial mode only differs from ``standard`` at
``poly_degree >= 2``, verify the run-list configs satisfy that (the extended runs
all do).

The run-list format is one run per non-comment line, whitespace-separated, with
at least three columns: ``run_name  source_sweep_dir  gpu`` (further columns are
ignored). Runs are grouped by their ``gpu`` column and written to one manifest per
GPU type inside a single new sweep dir, so each group can be submitted to
``scripts/run_sweep.sh`` with the matching ``--gres`` override. The new run names
are prefixed with the source model tag (``quantum`` / ``shared`` / ``stacked``)
because the original names do not encode the model — two different-model runs can
share a name and would otherwise collide in one sweep dir.

Example (run from the repo root on the cluster login node):

    uv run python experiments/build_qsvt_extended.py
    # → writes results/sweeps/qsvt_extended_<ts>/manifest_<gpu>.json and prints
    #   the exact `sbatch --gres=... --array=... scripts/run_sweep.sh <manifest>`
    #   lines to submit.

This writes manifests only; it never launches anything (submit the printed
`sbatch` lines yourself). Reporting afterwards is the usual
`scripts/submit_report.sh <new_sweep_dir>` plus a `report_sweep_compare.py`
overlay against the source (standard-mode) sweeps.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Make ``cv_quixer`` importable when run as a script from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cv_quixer.provenance import invocation_record

# Source-sweep-dir basename prefixes → the short model tag used in the new run
# name. The tag disambiguates otherwise-identical run names across model variants.
_MODEL_TAGS = ("quantum", "shared", "stacked")

# argv flags that take a value and must be dropped from the replayed argv (the
# builder re-adds --epochs; --resume must never survive — a qsvt run starts fresh).
_DROP_WITH_VALUE = ("--epochs", "--resume")


def _model_tag(sweep_dir: str) -> str:
    """Short model tag inferred from the source sweep-dir name (``high_epoch_<tag>_…``)."""
    base = Path(sweep_dir).name
    for tag in _MODEL_TAGS:
        if base.startswith(f"high_epoch_{tag}"):
            return tag
    # Fall back to any tag appearing as a token, else a generic marker.
    for tag in _MODEL_TAGS:
        if f"_{tag}_" in base:
            return tag
    return "run"


def _parse_runlist(path: Path) -> list[tuple[str, str, str]]:
    """Parse ``(run_name, source_sweep_dir, gpu)`` triples from the run-list file."""
    entries: list[tuple[str, str, str]] = []
    for lineno, raw in enumerate(path.read_text().splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        cols = line.split()
        if len(cols) < 3:
            raise ValueError(
                f"{path}:{lineno}: expected at least 3 whitespace columns "
                f"(run_name source_sweep_dir gpu), got {len(cols)}: {line!r}"
            )
        entries.append((cols[0], cols[1], cols[2]))
    if not entries:
        raise ValueError(f"{path}: no run entries found (all lines blank/comments).")
    return entries


def _transform_args(
    args: list[str], new_run_name: str, runs_root: str, poly_mode: str, epochs: int
) -> list[str]:
    """Replay one run's argv into a fresh qsvt run: drop epochs/resume, rewrite
    the run-name + runs-root, and append the poly-mode + epoch overrides."""
    out: list[str] = []
    i = 0
    while i < len(args):
        a = args[i]
        if a in _DROP_WITH_VALUE:
            i += 2  # skip the flag and its value
            continue
        if a == "--run-name":
            out += ["--run-name", new_run_name]
            i += 2
            continue
        if a == "--runs-root":
            out += ["--runs-root", runs_root]
            i += 2
            continue
        out.append(a)
        i += 1
    out += ["--poly-mode", poly_mode, "--epochs", str(epochs)]
    return out


def build(args: argparse.Namespace) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    new_dir = Path(args.sweeps_root) / f"{args.out_name}_{timestamp}"
    new_dir.mkdir(parents=True, exist_ok=True)

    entries = _parse_runlist(Path(args.runlist))

    # gpu → list[(new_run_name, new_args)]
    groups: dict[str, list[tuple[str, list[str]]]] = {}
    for orig_name, sweep_dir, gpu in entries:
        manifest_path = Path(sweep_dir) / "sweep_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"source manifest not found: {manifest_path} (needed to replay "
                f"argv for run {orig_name!r})"
            )
        src_manifest = json.loads(manifest_path.read_text())
        try:
            src = next(r for r in src_manifest["runs"] if r["run_name"] == orig_name)
        except StopIteration:
            raise KeyError(
                f"run {orig_name!r} not found in {manifest_path} — check the "
                "run-list against the source sweep."
            )
        tag = _model_tag(sweep_dir)
        new_name = f"{tag}__{orig_name}__{args.poly_mode}"
        new_args = _transform_args(
            src["args"], new_name, str(new_dir), args.poly_mode, args.epochs
        )
        groups.setdefault(gpu, []).append((new_name, new_args))

    invocation = invocation_record()
    print(f"qsvt sweep dir: {new_dir}  ({len(entries)} runs)")
    for gpu, runs in sorted(groups.items()):
        manifest = {
            "sweep_name": new_dir.name,
            "sweep_dir": str(new_dir),
            "gpu": gpu,
            "n_runs": len(runs),
            "runs": [
                {"index": i, "run_name": name, "args": run_args}
                for i, (name, run_args) in enumerate(runs)
            ],
            "invocations": [invocation],
        }
        manifest_path = new_dir / f"manifest_{gpu.replace('-', '')}.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))
        gres = "gpu:a100-40:1" if gpu.startswith("a100") else f"gpu:{gpu}:1"
        print(f"\n# {gpu}: {len(runs)} run(s) → {manifest_path}")
        print(
            f"sbatch --gres={gres} --array=0-{len(runs) - 1} "
            f"scripts/run_sweep.sh {manifest_path}"
        )
    print("\n(manifests written; no runs launched — submit the sbatch lines above.)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clone a curated run-list into a fresh qsvt-mode sweep (ADR-0009)."
    )
    parser.add_argument(
        "--runlist", type=str, default="results/extended_runs_25ep.txt",
        help="run-list file: 'run_name source_sweep_dir gpu' per non-comment line "
        "(default: results/extended_runs_25ep.txt).",
    )
    parser.add_argument(
        "--epochs", type=int, default=25,
        help="total epochs for each fresh qsvt run (default: 25).",
    )
    parser.add_argument(
        "--poly-mode", type=str, default="qsvt", choices=["standard", "qsvt"],
        help="polynomial construction mode to run these configs in (default: qsvt).",
    )
    parser.add_argument(
        "--sweeps-root", type=str, default="results/sweeps",
        help="parent dir for the new sweep dir (default: results/sweeps).",
    )
    parser.add_argument(
        "--out-name", type=str, default="qsvt_extended",
        help="new sweep-dir name stem, timestamp appended (default: qsvt_extended).",
    )
    build(parser.parse_args())


if __name__ == "__main__":
    main()

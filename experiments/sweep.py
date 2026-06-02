"""Hyperparameter sweep orchestrator for CV-Quixer.

Fans a grid over two axes — parameter budget (`--target-params`) and observable
readout preset (`--observables`), optionally repeated over `--seeds` — into
independent `full_experiment.py` runs. Each grid point is ONE training run in
its own process, so one config crashing never takes down the sweep.

The grid is written to `results/sweeps/<sweep_name>_<ts>/sweep_manifest.json`;
each run lands in a sibling sub-directory named like `p13760__obs-xpxsps__seed42`
(a full `full_experiment.py` run dir). Aggregate the results afterwards with
`experiments/report_sweep.py --sweep-dir <that dir>`.

Examples
--------
Write the manifest only (inspect before launching)::

    uv run python experiments/sweep.py \\
        --target-params 8000 13760 20000 --observables x xpxsps pnr --dry-run

Run the grid locally, sequentially (small/smoke configs)::

    uv run python experiments/sweep.py \\
        --target-params 8000 13760 --observables xp xpxsps \\
        --epochs 1 --train-fraction 0.02 --test-fraction 0.05 --launch local

Submit the grid as a SLURM array (one array task per grid point)::

    uv run python experiments/sweep.py \\
        --target-params 8000 13760 20000 --observables x xp xpxsps pnr \\
        --epochs 3 --train-fraction 0.1 --test-fraction 0.1 --launch slurm

All runs share `--subset-seed` and the same train/test fractions so every
config sees the identical data subset (apples-to-apples comparison).
"""

from __future__ import annotations

import argparse
import itertools
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from cv_quixer.config.observable_presets import PRESET_NAMES

# full_experiment.py, relative to the repo root (this file's grandparent).
REPO_ROOT = Path(__file__).resolve().parent.parent
FULL_EXPERIMENT = "experiments/full_experiment.py"
RUN_SWEEP_SH = "scripts/run_sweep.sh"


def _run_name(
    target_params: int,
    observables: str,
    seed: int,
    num_layers: int,
    scaling_knob: str,
    trunc_lambda: float | None,
) -> str:
    """Encode a grid point as a filesystem-safe run-directory name.

    The ``__knob-{knob}`` marker is omitted for the historic default
    ``cnn_channels_2`` and the ``__L{n}`` suffix is omitted at ``num_layers == 1``,
    so single-knob / single-layer sweeps keep their historic run-directory names
    while multi-knob grids stay collision-free. A ``__tl{λ}`` marker is appended
    only when ``trunc_lambda`` is explicitly swept (``None`` ⇒ inherit
    full_experiment.py's default, no marker).
    """
    base = f"p{target_params}__obs-{observables}"
    if scaling_knob != "cnn_channels_2":
        base += f"__knob-{scaling_knob}"
    base += f"__seed{seed}"
    if num_layers != 1:
        base += f"__L{num_layers}"
    if trunc_lambda is not None:
        base += f"__tl{trunc_lambda}"
    return base


def build_manifest(args: argparse.Namespace) -> dict:
    """Expand the Cartesian grid into a manifest dict (no side effects)."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    sweep_dir = Path(args.sweeps_root) / f"{args.sweep_name}_{timestamp}"

    # Args every run shares (data subset + epochs). Kept identical across the
    # grid so comparisons are apples-to-apples.
    common: list[str] = ["--runs-root", str(sweep_dir), "--subset-seed", str(args.subset_seed)]
    if args.model != "quantum":
        common += ["--model", args.model]
    if args.epochs is not None:
        common += ["--epochs", str(args.epochs)]
    if args.train_fraction is not None:
        common += ["--train-fraction", str(args.train_fraction)]
    if args.test_fraction is not None:
        common += ["--test-fraction", str(args.test_fraction)]
    if args.wandb:
        common += ["--wandb", "--wandb-group", f"{args.sweep_name}_{timestamp}"]

    # trunc_lambda is an optional axis: when not given, iterate a single None so
    # the flag is omitted and behaviour is byte-identical to a no-axis sweep.
    trunc_lambdas = args.trunc_lambda if args.trunc_lambda else [None]

    runs: list[dict] = []
    for idx, (tp, obs, seed, n_layers, knob, tl) in enumerate(
        itertools.product(
            args.target_params, args.observables, args.seeds,
            args.num_layers, args.scaling_knob, trunc_lambdas,
        )
    ):
        run_name = _run_name(tp, obs, seed, n_layers, knob, tl)
        run_args = [
            "--target-params", str(tp),
            "--observables", obs,
            "--seed", str(seed),
            "--num-layers", str(n_layers),
            "--scaling-knob", knob,
            "--run-name", run_name,
            *common,
        ]
        if tl is not None:
            run_args += ["--trunc-lambda", str(tl)]
        runs.append(
            {
                "index": idx,
                "run_name": run_name,
                "target_params": tp,
                "observables": obs,
                "seed": seed,
                "num_layers": n_layers,
                "scaling_knob": knob,
                "trunc_lambda": tl,
                "args": run_args,
            }
        )

    return {
        "sweep_name": args.sweep_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "sweeps_root": str(args.sweeps_root),
        "sweep_dir": str(sweep_dir),
        "axes": {
            "target_params": list(args.target_params),
            "observables": list(args.observables),
            "seeds": list(args.seeds),
            "num_layers": list(args.num_layers),
            "scaling_knob": list(args.scaling_knob),
            "trunc_lambda": list(args.trunc_lambda or []),
        },
        "common_args": common,
        "n_runs": len(runs),
        "runs": runs,
    }


def launch_local(manifest: dict) -> int:
    """Run each grid point sequentially via subprocess. Returns failure count."""
    failures = 0
    for run in manifest["runs"]:
        cmd = [sys.executable, FULL_EXPERIMENT, *run["args"]]
        print(f"\n=== [{run['index'] + 1}/{manifest['n_runs']}] {run['run_name']} ===")
        print("  " + " ".join(cmd))
        result = subprocess.run(cmd, cwd=REPO_ROOT)
        if result.returncode != 0:
            failures += 1
            print(f"  ✗ run {run['run_name']} exited with code {result.returncode}")
    return failures


def launch_slurm(manifest: dict, manifest_path: Path) -> None:
    """Submit the grid as a SLURM array (or print the command if sbatch is absent)."""
    n = manifest["n_runs"]
    cmd = ["sbatch", f"--array=0-{n - 1}", RUN_SWEEP_SH, str(manifest_path)]
    print("\nSLURM array submission:")
    print("  " + " ".join(cmd))
    if shutil.which("sbatch") is None:
        print(
            "\nsbatch not found on PATH — run the command above on the cluster "
            "login node (from the repo root)."
        )
        return
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fan a (param-count × observables × seed) grid into "
        "independent full_experiment.py runs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # --- grid axes --------------------------------------------------------
    parser.add_argument(
        "--target-params", type=int, nargs="+", required=True,
        help="one or more parameter budgets (auto-scales cnn_channels_2 per run)",
    )
    parser.add_argument(
        "--observables", type=str, nargs="+", default=["xpxsps"],
        choices=PRESET_NAMES,
        help="one or more observable readout presets",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[42],
        help="one or more training seeds (for error bars / repeats)",
    )
    parser.add_argument(
        "--num-layers", type=int, nargs="+", default=[1],
        help="one or more per-patch circuit depths L (default [1]); each L adds "
        "a stacked gate sequence + a BS→Rot interferometer",
    )
    parser.add_argument(
        "--scaling-knob", type=str, nargs="+", default=["cnn_channels_2"],
        help="one or more QuantumConfig fields to auto-scale toward each "
        "--target-params (grid axis; e.g. cnn_channels_2 num_heads). Forwarded "
        "to full_experiment.py per run.",
    )
    parser.add_argument(
        "--trunc-lambda", type=float, nargs="+", default=None,
        help="one or more Fock truncation penalty weights (grid axis). Omit to "
        "inherit full_experiment.py's default (no extra runs, no name marker).",
    )
    # --- shared run settings (identical across the grid) ------------------
    parser.add_argument(
        "--model", type=str, default="quantum",
        choices=["quantum", "quantum_shared", "classical"],
        help="model variant for every run (forwarded to full_experiment.py; "
        "'quantum_shared' auto-scales on num_heads)",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--train-fraction", type=float, default=None)
    parser.add_argument("--test-fraction", type=float, default=None)
    parser.add_argument(
        "--subset-seed", type=int, default=42,
        help="data-subset seed shared by every run (apples-to-apples)",
    )
    # --- orchestration ----------------------------------------------------
    parser.add_argument("--sweep-name", type=str, default="sweep")
    parser.add_argument("--sweeps-root", type=str, default="results/sweeps")
    parser.add_argument(
        "--launch", choices=["local", "slurm", "none"], default="none",
        help="local: run sequentially here; slurm: submit a job array; "
        "none: just write the manifest",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="alias for --launch none (write manifest + print plan only)",
    )
    parser.add_argument(
        "--wandb", action="store_true",
        help="enable W&B for every run, grouped under the sweep name",
    )
    args = parser.parse_args()

    launch = "none" if args.dry_run else args.launch

    manifest = build_manifest(args)
    sweep_dir = Path(manifest["sweep_dir"])
    sweep_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = sweep_dir / "sweep_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Sweep: {manifest['sweep_name']}  ({manifest['n_runs']} runs)")
    print(f"  target_params: {manifest['axes']['target_params']}")
    print(f"  observables:   {manifest['axes']['observables']}")
    print(f"  seeds:         {manifest['axes']['seeds']}")
    print(f"  num_layers:    {manifest['axes']['num_layers']}")
    print(f"  scaling_knob:  {manifest['axes']['scaling_knob']}")
    print(f"  trunc_lambda:  {manifest['axes']['trunc_lambda']}")
    print(f"  manifest:      {manifest_path}")
    for run in manifest["runs"]:
        print(f"    [{run['index']}] {run['run_name']}")

    if launch == "local":
        failures = launch_local(manifest)
        print(
            f"\nLocal sweep finished: {manifest['n_runs'] - failures}/"
            f"{manifest['n_runs']} runs succeeded."
        )
        print(f"Aggregate: uv run python experiments/report_sweep.py --sweep-dir {sweep_dir}")
        if failures:
            sys.exit(1)
    elif launch == "slurm":
        launch_slurm(manifest, manifest_path)
        print(f"\nAfter the array finishes, aggregate with:")
        print(f"  uv run python experiments/report_sweep.py --sweep-dir {sweep_dir}")
    else:
        print("\n(manifest written; no runs launched — use --launch local|slurm)")


if __name__ == "__main__":
    main()

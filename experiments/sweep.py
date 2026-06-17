"""Hyperparameter sweep orchestrator for CV-Quixer.

Fans a grid over two axes — parameter budget (`--target-params`) and observable
readout preset (`--observables`), optionally repeated over `--seeds` — into
independent `full_experiment.py` runs. Each grid point is ONE training run in
its own process, so one config crashing never takes down the sweep.

The grid is written to `results/sweeps/<sweep_name>_<ts>/sweep_manifest.json`;
each run lands in a sibling sub-directory named like `p13760__obs-xpxsps__seed42`
(a full `full_experiment.py` run dir). Aggregate the results afterwards with
`experiments/report_sweep.py --sweep-dir <that dir>`.

`--scaling-knob` is required — every grid point must name the knob to auto-scale
toward each `--target-params` (no implicit default), so the budget knob is always
explicit in the manifest and the per-run argv.

Examples
--------
Write the manifest only (inspect before launching)::

    uv run python experiments/sweep.py \\
        --target-params 8000 13760 20000 --observables x xpxsps pnr \\
        --scaling-knob num_heads --dry-run

Run the grid locally, sequentially (small/smoke configs)::

    uv run python experiments/sweep.py \\
        --target-params 8000 13760 --observables xp xpxsps \\
        --scaling-knob num_heads \\
        --epochs 1 --train-fraction 0.02 --test-fraction 0.05 --launch local

Submit the grid as a SLURM array (one array task per grid point)::

    uv run python experiments/sweep.py \\
        --target-params 8000 13760 20000 --observables x xp xpxsps pnr \\
        --scaling-knob num_heads \\
        --epochs 3 --train-fraction 0.1 --test-fraction 0.1 --launch slurm

All runs share `--subset-seed` and the same train/test fractions so every
config sees the identical data subset (apples-to-apples comparison).
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from datetime import datetime
from pathlib import Path

from _orchestration import launch_local, submit_slurm_array

from cv_quixer.config.observable_presets import PRESET_NAMES
from cv_quixer.provenance import invocation_record

# Repo-relative entry / array scripts the launchers drive (see _orchestration).
FULL_EXPERIMENT = "experiments/full_experiment.py"
RUN_SWEEP_SH = "scripts/run_sweep.sh"

# Manual architecture axes: directly-set QuantumConfig knobs (no auto-scaling).
# Each tuple is (dest, full_experiment_flag, run_name_marker, help). `dest` is the
# argparse destination (matches full_experiment.py's flag), `marker` keeps run
# directory names short + collision-free. These coexist with the budget axes
# (--target-params/--scaling-knob); the chosen scaling_knob is still auto-scaled.
ARCH_AXES: tuple[tuple[str, str, str, str], ...] = (
    ("num_heads", "--num-heads", "nh", "parallel CV attention heads"),
    ("num_modes", "--num-modes", "nm", "bosonic modes"),
    ("cutoff_dim", "--cutoff-dim", "cd", "Fock cutoff D"),
    ("poly_degree", "--poly-degree", "pd", "matrix polynomial degree d"),
    ("cnn_channels_1", "--cnn-channels-1", "c1", "first conv output channels"),
    ("cnn_channels_2", "--cnn-channels-2", "c2", "second conv output channels"),
    ("cnn_kernel_size", "--cnn-kernel-size", "ck", "conv kernel size"),
    ("decoder_hidden_dim", "--decoder-hidden-dim", "dh", "decoder MLP hidden width"),
    ("cnn_num_conv_layers", "--cnn-num-conv-layers", "ncl", "total CNN conv layers"),
    ("hypernet_num_linear_layers", "--hypernet-num-linear-layers", "hll",
     "total hypernet DNN linear layers"),
    ("decoder_num_layers", "--decoder-num-layers", "dnl", "total decoder linear layers"),
    ("cvqnn_num_layers", "--cvqnn-num-layers", "cw", "CVQNN block W depth L_W (0 = no W)"),
    ("num_seq2seq_blocks", "--num-seq2seq-blocks", "sb",
     "seq-to-seq blocks (--model quantum_stacked, ADR-0003)"),
)


def _run_name(point: dict) -> str:
    """Encode a grid point (a dict of axis→value) as a filesystem-safe run name.

    Budget mode (``target_params`` set) keeps its historic base ``p{tp}__obs-…``
    plus the ``__knob-{knob}`` marker (omitted for the historic ``cnn_channels_2``);
    manual mode (``target_params`` is ``None``) uses a ``manual__obs-…`` base. The
    ``__L{n}`` / ``__tl{λ}`` markers and one short marker per active manual arch
    axis (e.g. ``__nh6``) are appended only when that axis is set, so single-axis
    sweeps keep short names and multi-axis grids stay collision-free.
    """
    tp = point["target_params"]
    obs = point["observables"]
    if tp is not None:
        base = f"p{tp}__obs-{obs}"
        if point["scaling_knob"] != "cnn_channels_2":
            base += f"__knob-{point['scaling_knob']}"
    else:
        base = f"manual__obs-{obs}"
    base += f"__seed{point['seed']}"
    if point["num_layers"] != 1:
        base += f"__L{point['num_layers']}"
    if point["trunc_lambda"] is not None:
        base += f"__tl{point['trunc_lambda']}"
    if point.get("decoder_hidden_mult") is not None:
        base += f"__dhm{point['decoder_hidden_mult']}"
    if point.get("query_trunc_lambda") is not None:
        base += f"__qtl{point['query_trunc_lambda']}"
    if point.get("poly_init_noise") is not None:
        base += f"__pin{point['poly_init_noise']}"
    if point.get("pooling") is not None:
        base += f"__pool-{point['pooling']}"
    if point.get("block_residual") == "off":
        base += "__nores"
    for dest, _flag, marker, _help in ARCH_AXES:
        if point.get(dest) is not None:
            base += f"__{marker}{point[dest]}"
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
    if args.gate_param_bound is not None:
        common += ["--gate-param-bound", str(args.gate_param_bound)]
    if args.wandb:
        common += ["--wandb", "--wandb-group", f"{args.sweep_name}_{timestamp}"]

    # Optional axes iterate a single None when not given, so the corresponding
    # flag is omitted and behaviour is byte-identical to a no-axis sweep.
    #   - budget mode: target_params + scaling_knob are set (auto-scaling).
    #   - manual mode: target_params is None ⇒ no --target-params/--scaling-knob.
    target_params = args.target_params if args.target_params else [None]
    scaling_knobs = args.scaling_knob if args.scaling_knob else [None]
    trunc_lambdas = args.trunc_lambda if args.trunc_lambda else [None]
    # decoder_hidden_mult is a float axis (like trunc_lambda), not an int ARCH_AXES.
    decoder_hidden_mults = args.decoder_hidden_mult if args.decoder_hidden_mult else [None]
    # Stacked-model axes (non-int, so not ARCH_AXES): query_trunc_lambda is a
    # float axis; pooling / block_residual are choice axes.
    query_trunc_lambdas = args.query_trunc_lambda if args.query_trunc_lambda else [None]
    poly_init_noises = args.poly_init_noise if args.poly_init_noise else [None]
    poolings = args.pooling if args.pooling else [None]
    block_residuals = args.block_residual if args.block_residual else [None]
    arch_values = {dest: (getattr(args, dest) or [None]) for dest, *_ in ARCH_AXES}

    # Ordered axis names + their value-lists for one flat Cartesian product.
    axis_names = (
        ["target_params", "observables", "seed", "num_layers", "scaling_knob",
         "trunc_lambda", "decoder_hidden_mult", "query_trunc_lambda",
         "poly_init_noise", "pooling", "block_residual"]
        + [dest for dest, *_ in ARCH_AXES]
    )
    axis_value_lists = (
        [target_params, args.observables, args.seeds, args.num_layers,
         scaling_knobs, trunc_lambdas, decoder_hidden_mults,
         query_trunc_lambdas, poly_init_noises, poolings, block_residuals]
        + [arch_values[dest] for dest, *_ in ARCH_AXES]
    )

    runs: list[dict] = []
    for idx, combo in enumerate(itertools.product(*axis_value_lists)):
        point = dict(zip(axis_names, combo))
        run_name = _run_name(point)
        run_args = [
            "--observables", point["observables"],
            "--seed", str(point["seed"]),
            "--num-layers", str(point["num_layers"]),
            "--run-name", run_name,
            *common,
        ]
        # Budget mode: emit --target-params + --scaling-knob (auto-scaling).
        if point["target_params"] is not None:
            run_args = (
                ["--target-params", str(point["target_params"]),
                 "--scaling-knob", point["scaling_knob"]]
                + run_args
            )
        if point["trunc_lambda"] is not None:
            run_args += ["--trunc-lambda", str(point["trunc_lambda"])]
        if point["decoder_hidden_mult"] is not None:
            run_args += ["--decoder-hidden-mult", str(point["decoder_hidden_mult"])]
        if point["query_trunc_lambda"] is not None:
            run_args += ["--query-trunc-lambda", str(point["query_trunc_lambda"])]
        if point["poly_init_noise"] is not None:
            run_args += ["--poly-init-noise", str(point["poly_init_noise"])]
        if point["pooling"] is not None:
            run_args += ["--pooling", point["pooling"]]
        if point["block_residual"] is not None:
            run_args += ["--block-residual", point["block_residual"]]
        # Manual arch axes: emit each set field directly.
        for dest, flag, _marker, _help in ARCH_AXES:
            if point[dest] is not None:
                run_args += [flag, str(point[dest])]
        runs.append({"index": idx, "run_name": run_name, **point, "args": run_args})

    axes = {
        "target_params": list(args.target_params or []),
        "observables": list(args.observables),
        "seeds": list(args.seeds),
        "num_layers": list(args.num_layers),
        "scaling_knob": list(args.scaling_knob or []),
        "trunc_lambda": list(args.trunc_lambda or []),
        "decoder_hidden_mult": list(args.decoder_hidden_mult or []),
        "query_trunc_lambda": list(args.query_trunc_lambda or []),
        "poly_init_noise": list(args.poly_init_noise or []),
        "pooling": list(args.pooling or []),
        "block_residual": list(args.block_residual or []),
    }
    for dest, *_ in ARCH_AXES:
        axes[dest] = list(getattr(args, dest) or [])

    return {
        "sweep_name": args.sweep_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        # Launch provenance (CONTEXT.md: Invocation) — the exact sweep.py
        # command that defined this grid, so a re-run never has to be
        # reverse-engineered from axes/common_args.
        "invocations": [invocation_record()],
        "sweeps_root": str(args.sweeps_root),
        "sweep_dir": str(sweep_dir),
        "axes": axes,
        "common_args": common,
        "n_runs": len(runs),
        "runs": runs,
    }


def build_parser() -> argparse.ArgumentParser:
    """Construct the sweep CLI parser (extracted from main for testability)."""
    parser = argparse.ArgumentParser(
        description="Fan a (param-count × observables × seed) grid into "
        "independent full_experiment.py runs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # --- grid axes --------------------------------------------------------
    # Two modes (combinable per-run for fixed-field budget sweeps):
    #   budget — --target-params [+ --scaling-knob] auto-scales a knob to a budget.
    #   manual — set architecture knobs directly (the ARCH_AXES below); no scaling.
    # At least one of {--target-params, an ARCH_AXES flag} is required.
    parser.add_argument(
        "--target-params", type=int, nargs="+", default=None,
        help="budget mode: one or more parameter budgets (auto-scales --scaling-knob "
        "per run). Omit for a manual sweep that sets architecture knobs directly.",
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
        "--scaling-knob", type=str, nargs="+", default=None,
        help="budget mode only: one or more QuantumConfig fields to auto-scale "
        "toward each --target-params (grid axis). Required when --target-params is "
        "given (no silent default). Use num_heads for quantum / quantum_shared "
        "width, cnn_channels_2 for the CNN-width knob.",
    )
    parser.add_argument(
        "--trunc-lambda", type=float, nargs="+", default=None,
        help="one or more Fock truncation penalty weights (grid axis). Omit to "
        "inherit full_experiment.py's default (no extra runs, no name marker).",
    )
    parser.add_argument(
        "--decoder-hidden-mult", type=float, nargs="+", default=None,
        help="manual grid axis: one or more decoder-hidden-dim multipliers c "
        "(decoder_hidden_dim = round(c * decoder_in_dim)). Mutually exclusive with "
        "the --decoder-hidden-dim axis.",
    )
    parser.add_argument(
        "--query-trunc-lambda", type=float, nargs="+", default=None,
        help="grid axis (--model quantum_stacked): one or more query-unitary "
        "truncation penalty weights. Omit to inherit full_experiment.py's default.",
    )
    parser.add_argument(
        "--poly-init-noise", type=float, nargs="+", default=None,
        help="manual grid axis: one or more poly-init-noise std values (seeds the "
        "polynomial coeffs c_{j>=1} to break uniform-predictor collapse; __pin "
        "marker). Omit to inherit full_experiment.py's default (off, no marker).",
    )
    parser.add_argument(
        "--pooling", type=str, nargs="+", default=None,
        choices=["mean", "quixer"],
        help="grid axis (--model quantum_stacked): one or more pooling modes "
        "('mean' | 'quixer'). Omit to inherit full_experiment.py's default.",
    )
    parser.add_argument(
        "--block-residual", type=str, nargs="+", default=None,
        choices=["on", "off"],
        help="grid axis (--model quantum_stacked): residual wiring on/off "
        "('off' marks the run dir __nores). Omit to inherit the default.",
    )
    # Manual architecture axes (no auto-scaling). Each is nargs='+', default None
    # ⇒ inherit full_experiment.py's value (single 'inherit' point, no name marker).
    for dest, flag, _marker, helptxt in ARCH_AXES:
        parser.add_argument(
            flag, type=int, nargs="+", default=None,
            help=f"manual grid axis: {helptxt} (forwarded to full_experiment.py)",
        )
    # --- shared run settings (identical across the grid) ------------------
    parser.add_argument(
        "--model", type=str, default="quantum",
        choices=["quantum", "quantum_shared", "quantum_stacked", "classical"],
        help="model variant for every run (forwarded to full_experiment.py; "
        "'quantum_shared' auto-scales on num_heads; 'quantum_stacked' is the "
        "seq-to-seq stacked model, ADR-0003)",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--train-fraction", type=float, default=None)
    parser.add_argument("--test-fraction", type=float, default=None)
    parser.add_argument(
        "--subset-seed", type=int, default=42,
        help="data-subset seed shared by every run (apples-to-apples)",
    )
    parser.add_argument(
        "--gate-param-bound", type=str, default=None,
        help="forwarded to every run's full_experiment.py: soft-clip the magnitude "
        "gate params ('auto' = cutoff-aware photon budget, or a float). Off by "
        "default. A fixed setting, not a grid axis.",
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
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Require at least one budget axis or one manual arch axis, so a sweep can
    # never be silently empty / under-specified. decoder_hidden_mult is a manual
    # axis too.
    any_arch_axis = (
        any(getattr(args, dest) for dest, *_ in ARCH_AXES)
        or bool(args.decoder_hidden_mult)
        or bool(args.poly_init_noise)
    )
    if not args.target_params and not any_arch_axis:
        parser.error(
            "specify either --target-params (budget mode) or at least one manual "
            "architecture axis (e.g. --num-heads / --num-modes / --poly-degree)."
        )
    if args.decoder_hidden_dim and args.decoder_hidden_mult:
        parser.error(
            "--decoder-hidden-dim and --decoder-hidden-mult are mutually exclusive."
        )
    # The budget knob must be explicit whenever a budget is given.
    if args.target_params and not args.scaling_knob:
        parser.error("--scaling-knob is required when --target-params is given.")
    if args.scaling_knob and not args.target_params:
        parser.error("--scaling-knob is only valid together with --target-params.")

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
    if manifest["axes"].get("decoder_hidden_mult"):
        print(f"  decoder_hidden_mult: {manifest['axes']['decoder_hidden_mult']}")
    for _ax in ("query_trunc_lambda", "pooling", "block_residual"):
        if manifest["axes"].get(_ax):
            print(f"  {_ax}: {manifest['axes'][_ax]}")
    for dest, *_ in ARCH_AXES:
        if manifest["axes"].get(dest):
            print(f"  {dest}:{' ' * max(1, 13 - len(dest))}{manifest['axes'][dest]}")
    print(f"  manifest:      {manifest_path}")
    for run in manifest["runs"]:
        print(f"    [{run['index']}] {run['run_name']}")

    if launch == "local":
        failures = launch_local(manifest, FULL_EXPERIMENT)
        print(
            f"\nLocal sweep finished: {manifest['n_runs'] - failures}/"
            f"{manifest['n_runs']} runs succeeded."
        )
        print(f"Aggregate: uv run python experiments/report_sweep.py --sweep-dir {sweep_dir}")
        if failures:
            sys.exit(1)
    elif launch == "slurm":
        submit_slurm_array(manifest, manifest_path, RUN_SWEEP_SH)
        print(f"\nAfter the array finishes, aggregate with:")
        print(f"  uv run python experiments/report_sweep.py --sweep-dir {sweep_dir}")
    else:
        print("\n(manifest written; no runs launched — use --launch local|slurm)")


if __name__ == "__main__":
    main()

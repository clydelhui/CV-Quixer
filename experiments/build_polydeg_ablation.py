"""Build (and launch) the polynomial-degree ablation sweep.

A CV-Quixer head applies a matrix polynomial P(M) = Σ_{j=0}^{d} c_j M^j to the
post-LCU state, where ``d = poly_degree``. This ablation asks whether collapsing
the polynomial to **degree 1** (P(M) = c_0 I + c_1 M — a bare linear combination,
no higher matrix powers) significantly degrades accuracy relative to the curated
best configs, which sit at degree 2 or 3.

This builder fans the **16 curated epoch-extension configs** (the ones in
``results/extended_runs_25ep.txt``, spanning all three quantum models) over the
``{poly_degree=1}`` arm into one fresh-from-scratch **16-run** sweep.

The original-degree **baseline is deliberately NOT generated** — it is reused
from the existing ``high_epoch_*`` runs (the same configs at their native
poly_degree 2/3) and compared via ``report_sweep_compare.py``, so this builder
only emits the degree-1 arm.

For each of the 16 source argv (read verbatim from the source sweep's
``sweep_manifest.json`` ``runs[].args``) × ``{degree}`` it:

  * overwrites ``--poly-degree`` with the ablation degree (sources carry 2 / 3),
  * rewrites ``--run-name`` to append ``__pd<degree>`` as the override marker
    (the source name already embeds its native ``__pd2``/``__pd3``; two quantum
    sources differ *only* in that marker, so the degree cannot be rewritten in
    place without colliding — it is appended as the trailing override instead),
  * repoints ``--runs-root`` at the new ``results/sweeps/polydeg_ablation_<ts>``
    dir,
  * normalises ``--epochs`` to 10 (quantum/shared sources are already 10; the
    stacked sources carry ``--epochs 3`` and must be rewritten),
  * drops any ``--resume`` (every run starts from scratch),
  * keeps everything else verbatim (``--model``, ``--gate-param-bound auto``,
    ``--subset-seed 42``, fractions, all arch flags).

Lowering the polynomial degree only *reduces* the per-head matrix-power work, so
the Fock-sim memory/wall (set by num_heads × cutoff^num_modes × d) is ≤ each
source's. The GPU/wall map from ``results/extended_runs_25ep.txt`` therefore
carries over unchanged (conservatively): a100-40 for quantum/shared/stacked-nm2,
h100-96 for the heavy stacked-nm3 runs. Runs are ordered so each target-GPU group
is a contiguous index range, so the two SLURM array slices can be submitted
separately over the one manifest (the schema ``scripts/run_sweep.sh`` consumes
unchanged).

Collapse caveat: at degree 1 the default polynomial init c = [1, 0] makes
P(M) = I — input-independent at init, the uniform-predictor start of ADR-0007.
This builder therefore seeds the polynomial coeffs with ``--poly-init-noise``
(default 0.1) on every generated run to break that symmetry, recorded in the
run name as the ``__pin<eps>`` marker. Set ``--poly-init-noise 0`` to reproduce
the bare default init instead.

Examples
--------
Inspect the 16-run plan only::

    uv run python experiments/build_polydeg_ablation.py --dry-run

Submit both GPU groups as SLURM array slices::

    uv run python experiments/build_polydeg_ablation.py --launch slurm
"""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import sys
from datetime import datetime
from pathlib import Path

from _orchestration import launch_local, stamp_slurm_provenance, submit_one

# The curated-config selection, GPU map, and the verbatim-argv loaders are shared
# with the sibling ablation builders (same 16 configs, same source-manifest read).
from build_coeff_ablation import (
    GPU_CONFIG,
    GPU_ORDER,
    _model_from_args,
    parse_source_runs_file,
    resolve_source_runs,
)

from cv_quixer.provenance import invocation_record

FULL_EXPERIMENT = "experiments/full_experiment.py"
RUN_SWEEP_SH = "scripts/run_sweep.sh"

# The polynomial degree(s) this sweep trains from scratch. The native-degree
# baseline (2/3) is reused, not regenerated.
DEFAULT_DEGREES = (1,)

# Symmetry-breaking noise seeded into the polynomial coeffs c_{j>=1} on every
# generated run (the degree-1 runs are input-independent at the bare default
# init — see the module docstring / ADR-0007).
DEFAULT_POLY_INIT_NOISE = 0.1

# Every run is normalised to this many epochs (stacked sources carry 3).
DEFAULT_EPOCHS = 10

# The curated-config selection list (run_name / source sweep / target GPU).
DEFAULT_SOURCE_RUNS_FILE = "results/extended_runs_25ep.txt"


def arm_run_name(
    original_run_name: str, degree: int, model: str = "quantum",
    poly_init_noise: float = 0.0,
) -> str:
    """The ablation dir name: ``<model>__<original>__pd<degree>[__pin<eps>]``.

    The ``<model>__`` prefix is load-bearing (manual-mode names do not encode the
    model, so same-arch ``quantum`` / ``quantum_shared`` runs share a name — see
    ``build_coeff_ablation.arm_run_name``). ``__pd<degree>`` is the trailing
    override marker; the source name still carries its native ``__pd2``/``__pd3``
    so the ablation run remains traceable to its source config. ``__pin<eps>``
    (sweep.py's marker spelling) is appended when poly-init noise is seeded.
    """
    name = f"{model}__{original_run_name}__pd{degree}"
    if poly_init_noise > 0:
        name += f"__pin{poly_init_noise}"
    return name


def rewrite_run_args(
    args: list[str], degree: int, *, model: str, runs_root: str, target_epochs: int,
    poly_init_noise: float = 0.0,
) -> list[str]:
    """Original argv rewritten for one fresh degree-ablation run (never resumed).

    Overwrites ``--poly-degree``, injects ``--poly-init-noise`` (when > 0),
    rewrites ``--run-name`` to the ``<model>__…__pd<degree>[__pin<eps>]`` form
    (``arm_run_name``), repoints ``--runs-root``, normalises ``--epochs``, and
    strips any ``--resume`` (with its value). ``--model`` itself and everything
    else are replayed verbatim.
    """
    out = list(args)

    def _set(flag: str, value: str) -> None:
        if flag in out:
            out[out.index(flag) + 1] = value
        else:
            out.extend([flag, value])

    if "--run-name" in out:
        i = out.index("--run-name") + 1
        out[i] = arm_run_name(out[i], degree, model, poly_init_noise)
    else:  # pragma: no cover - source argv always carries --run-name
        out.extend(["--run-name", arm_run_name("run", degree, model, poly_init_noise)])
    _set("--poly-degree", str(degree))
    if poly_init_noise > 0:
        _set("--poly-init-noise", str(poly_init_noise))
    _set("--runs-root", runs_root)
    _set("--epochs", str(target_epochs))

    if "--resume" in out:
        i = out.index("--resume")
        del out[i:i + 2]
    return out


def build_manifest(
    source_runs: list[dict],
    *,
    sweep_dir: Path,
    degrees: tuple[int, ...] = DEFAULT_DEGREES,
    target_epochs: int = DEFAULT_EPOCHS,
    poly_init_noise: float = DEFAULT_POLY_INIT_NOISE,
) -> dict:
    """Fan the resolved source runs over ``degrees`` into a sweep manifest.

    ``source_runs`` are ``{run_name, gpu, model, args}`` (from
    ``resolve_source_runs``). Runs are grouped by target GPU in ``GPU_ORDER`` so
    each group is a contiguous ``index`` range; within a group, source order is
    preserved and the degrees are emitted in ``degrees`` order. The returned
    ``slurm_groups`` maps each present GPU to its ``[lo, hi]`` inclusive range.
    """
    runs_root = str(sweep_dir)
    entries: list[dict] = []
    slurm_groups: dict[str, list[int]] = {}

    by_gpu: dict[str, list[dict]] = {}
    for src in source_runs:
        by_gpu.setdefault(src["gpu"], []).append(src)
    unknown = set(by_gpu) - set(GPU_CONFIG)
    if unknown:
        raise ValueError(f"unknown / unusable GPU(s) in selection: {sorted(unknown)}")

    for gpu in GPU_ORDER:
        group = by_gpu.get(gpu)
        if not group:
            continue
        lo = len(entries)
        for src in group:
            model = src.get("model") or _model_from_args(src["args"])
            for degree in degrees:
                entries.append({
                    "index": len(entries),
                    "run_name": arm_run_name(
                        src["run_name"], degree, model, poly_init_noise),
                    "source_run_name": src["run_name"],
                    "model": model,
                    "poly_degree": degree,
                    "poly_init_noise": poly_init_noise,
                    "gpu": gpu,
                    "args": rewrite_run_args(
                        src["args"], degree, model=model, runs_root=runs_root,
                        target_epochs=target_epochs, poly_init_noise=poly_init_noise,
                    ),
                })
        slurm_groups[gpu] = [lo, len(entries) - 1]

    # Uniqueness guard: two entries with the same run_name would write into one
    # run dir and silently clobber each other. The model prefix separates
    # same-arch quantum vs shared configs; this catches a selection list that
    # names the same (model, config) twice, or a degrees list with a repeat.
    names = [e["run_name"] for e in entries]
    dups = sorted({n for n in names if names.count(n) > 1})
    if dups:
        raise ValueError(
            f"duplicate run name(s) in the polydeg-ablation plan: {dups} — the "
            "selection list names the same (model, config) more than once (or "
            "--poly-degrees repeats a value); de-duplicate before building."
        )

    return {
        "sweep_name": sweep_dir.name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        # Launch provenance (CONTEXT.md: Invocation).
        "invocations": [invocation_record()],
        "sweep_dir": runs_root,
        "poly_degrees": list(degrees),
        "poly_init_noise": poly_init_noise,
        "target_epochs": target_epochs,
        "slurm_groups": slurm_groups,
        "n_runs": len(entries),
        "runs": entries,
    }


def sbatch_commands(manifest: dict, manifest_path: Path) -> list[str]:
    """One ``sbatch`` array-slice command per GPU group (pure; no side effects)."""
    cmds: list[str] = []
    for gpu in GPU_ORDER:
        if gpu not in manifest["slurm_groups"]:
            continue
        lo, hi = manifest["slurm_groups"][gpu]
        cfg = GPU_CONFIG[gpu]
        cmds.append(shlex.join([
            "sbatch",
            f"--time={cfg['time']}",
            f"--gres={cfg['gres']}",
            f"--array={lo}-{hi}",
            RUN_SWEEP_SH,
            str(manifest_path),
        ]))
    return cmds


def _submit_slurm_groups(manifest: dict, manifest_path: Path) -> None:
    """Submit each GPU group as its own array slice; stamp provenance."""
    cmds = sbatch_commands(manifest, manifest_path)
    print("\nSLURM array submissions (one per GPU group):")
    for cmd in cmds:
        print("  " + cmd)
    if shutil.which("sbatch") is None:
        print(
            "\nsbatch not found on PATH — run the commands above on the cluster "
            "login node (from the repo root)."
        )
        return

    submissions = [submit_one(shlex.split(cmd)) for cmd in cmds]
    stamp_slurm_provenance(manifest, manifest_path, submissions)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the polynomial-degree ablation sweep: 16 curated "
        "configs × {poly_degree=1}, fresh from scratch (the native-degree 2/3 "
        "baseline is reused, not regenerated).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source-runs-file", type=Path, default=Path(DEFAULT_SOURCE_RUNS_FILE),
        help="curated-config selection list (run_name / source sweep dir / GPU)",
    )
    parser.add_argument(
        "--poly-degrees", type=int, nargs="+", default=list(DEFAULT_DEGREES),
        help="polynomial degree(s) to train from scratch (default: 1)",
    )
    parser.add_argument(
        "--poly-init-noise", type=float, default=DEFAULT_POLY_INIT_NOISE,
        help="symmetry-breaking std seeded into the polynomial coeffs of every "
             "generated (input-independent) run; 0 disables it (default: 0.1)",
    )
    parser.add_argument(
        "--sweeps-root", type=Path, default=Path("results/sweeps"),
        help="parent dir for the new polydeg_ablation_<ts> sweep dir",
    )
    parser.add_argument(
        "--sweep-name", type=str, default="polydeg_ablation",
        help="sweep dir is named <sweep-name>_<ts>",
    )
    parser.add_argument(
        "--epochs", type=int, default=DEFAULT_EPOCHS,
        help="target TOTAL epoch count per run (stacked sources are rewritten)",
    )
    parser.add_argument(
        "--launch", choices=["local", "slurm", "none"], default="none",
        help="local: run sequentially here; slurm: submit one array slice per "
             "GPU group; none: just write the manifest",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="alias for --launch none (write manifest + plan only)",
    )
    args = parser.parse_args()

    if not args.source_runs_file.is_file():
        parser.error(f"--source-runs-file does not exist: {args.source_runs_file}")

    rows = parse_source_runs_file(args.source_runs_file)
    source_runs = resolve_source_runs(rows)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    sweep_dir = args.sweeps_root / f"{args.sweep_name}_{timestamp}"
    manifest = build_manifest(
        source_runs, sweep_dir=sweep_dir,
        degrees=tuple(args.poly_degrees), target_epochs=args.epochs,
        poly_init_noise=args.poly_init_noise,
    )

    sweep_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = sweep_dir / "sweep_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Include list for the restricted comparison: exactly the degree-1 ablation
    # runs + their 16 source configs, so report_sweep_compare --include-file keeps
    # only these (and drops the other archs/seeds that live in the high_epoch_*
    # baseline sweeps).
    include_names = [e["run_name"] for e in manifest["runs"]]
    include_names += [src["run_name"] for src in source_runs]
    include_path = sweep_dir / "compare_include.txt"
    with open(include_path, "w") as f:
        f.write(
            "# Restricted-comparison include list: degree-1 ablation runs + their\n"
            "# 16 source configs. Pass to report_sweep_compare --include-file.\n"
        )
        f.write("\n".join(include_names) + "\n")

    print(f"Polydeg-ablation sweep: {manifest['sweep_name']}  ({manifest['n_runs']} runs)")
    print(f"  source runs:   {args.source_runs_file}  ({len(source_runs)} configs)")
    print(f"  degrees:       {manifest['poly_degrees']}")
    print(f"  poly-init-noise: {manifest['poly_init_noise']}")
    print(f"  epochs:        {manifest['target_epochs']}")
    print(f"  manifest:      {manifest_path}")
    print(f"  include list:  {include_path}  ({len(include_names)} names)")
    for gpu, (lo, hi) in manifest["slurm_groups"].items():
        print(f"  GPU {gpu}: indices {lo}-{hi}")
    for run in manifest["runs"]:
        print(f"    [{run['index']}] ({run['gpu']}) {run['run_name']}")

    print("\nSLURM array slices (one per GPU group):")
    for cmd in sbatch_commands(manifest, manifest_path):
        print("  " + cmd)

    launch = "none" if args.dry_run else args.launch
    if launch == "local":
        failures = launch_local(manifest, FULL_EXPERIMENT)
        print(f"\nSweep finished: "
              f"{manifest['n_runs'] - failures}/{manifest['n_runs']} succeeded.")
        if failures:
            sys.exit(1)
    elif launch == "slurm":
        _submit_slurm_groups(manifest, manifest_path)
        print(f"\nAfter the arrays finish, aggregate with:")
        print(f"  bash scripts/submit_report.sh {sweep_dir}")
    else:
        print("\n(manifest written; no runs launched — use --launch local|slurm)")


if __name__ == "__main__":
    main()

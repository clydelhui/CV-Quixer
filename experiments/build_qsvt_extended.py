"""Build (and launch) the qsvt polynomial-mode sweep.

A CV-Quixer head applies a matrix polynomial ``P(M)`` to the post-LCU state. The
``poly_mode`` knob (CONTEXT.md "Polynomial mode", ADR-0009) chooses how each
degree-``j`` term is built from the LCU ``M``: ``standard`` uses the literal power
``M^j``; ``qsvt`` alternates ``M`` with its adjoint ``M†`` to form the faithful
singular-value transform. The two coincide at ``poly_degree <= 1`` and differ at
degree >= 2 — every curated config here is degree 2 or 3, so ``qsvt`` is
meaningful throughout.

This builder fans the **16 curated epoch-extension configs** (the ones in
``results/extended_runs_25ep.txt``, spanning all three quantum models) over the
``{qsvt}`` arm into one fresh-from-scratch **16-run** sweep.

The ``standard`` baseline is deliberately NOT generated — it is reused from the
existing ``high_epoch_*`` runs (the same configs in the default ``standard`` mode)
and compared via ``report_sweep_compare.py``, so this builder only emits the
``qsvt`` arm.

For each of the 16 source argv (read verbatim from the source sweep's
``sweep_manifest.json`` ``runs[].args``) × ``{qsvt}`` it:

  * injects ``--poly-mode <mode>``,
  * rewrites ``--run-name`` to append ``__<mode>`` (sweep.py's ``__qsvt`` marker),
  * repoints ``--runs-root`` at the new ``results/sweeps/qsvt_extended_<ts>`` dir,
  * normalises ``--epochs`` to 10 (quantum/shared sources are already 10; the
    stacked sources carry ``--epochs 3`` and must be rewritten),
  * drops any ``--resume`` (every run starts from scratch),
  * keeps everything else verbatim (``--model``, ``--gate-param-bound auto``,
    ``--subset-seed 42``, fractions, all arch flags).

``qsvt`` applies the same number of block-encoding steps as ``standard`` (some
are ``M†`` rather than ``M``), so the Fock-sim memory/wall is unchanged from each
source. The GPU/wall map from ``results/extended_runs_25ep.txt`` therefore carries
over, **except** the heavy stacked-nm3 runs are remapped off ``h100-96`` onto
``h200-141``: the 96 GB H100s are sometimes MIG-split into 2×46 GB slices too
small for those ~86-89 GB configs (``DEFAULT_GPU_REMAP``; override with
``--gpu-remap``). At 10 epochs every run fits its wall in one window (a100 <= 12h,
h200 <= 3h), so no top-up is needed. Runs are ordered so each target-GPU group is
a contiguous index range, so the SLURM array slices can be submitted separately
over the one manifest (the schema ``scripts/run_sweep.sh`` consumes unchanged).

Examples
--------
Inspect the 16-run plan only::

    uv run python experiments/build_qsvt_extended.py --dry-run

Submit both GPU groups as SLURM array slices::

    uv run python experiments/build_qsvt_extended.py --launch slurm
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from _orchestration import launch_local

# The curated-config selection, GPU map, verbatim-argv loaders, and the per-GPU
# SLURM-slice submitter are shared with the sibling ablation builders (same 16
# configs, same source-manifest read, same one-manifest / contiguous-group layout).
from build_coeff_ablation import (
    GPU_CONFIG,
    GPU_ORDER,
    _model_from_args,
    _submit_slurm_groups,
    parse_source_runs_file,
    resolve_source_runs,
    sbatch_commands,
)

from cv_quixer.provenance import invocation_record

FULL_EXPERIMENT = "experiments/full_experiment.py"

# The polynomial mode(s) this sweep trains from scratch. The 'standard' baseline
# is reused (not regenerated; ADR-0009).
DEFAULT_POLY_MODES = ("qsvt",)

# Every run is normalised to this many epochs (stacked sources carry 3). At 10
# epochs every run fits its GPU wall in a single window — no top-ups.
DEFAULT_EPOCHS = 10

# The curated-config selection list (run_name / source sweep / target GPU).
DEFAULT_SOURCE_RUNS_FILE = "results/extended_runs_25ep.txt"

# Source-GPU relabelling applied before grouping: the heavy stacked-nm3 configs
# are listed as h100-96, but the 96 GB H100s are sometimes MIG-split into 2×46 GB
# slices too small for them, so redirect to the un-split 141 GB H200s. Override or
# extend with --gpu-remap OLD=NEW; the run-list file itself is never edited.
DEFAULT_GPU_REMAP = {"h100-96": "h200-141"}


def arm_run_name(original_run_name: str, mode: str, model: str = "quantum") -> str:
    """The variant dir name: ``<model>__<original>__<mode>``.

    The ``<model>__`` prefix is **load-bearing**, not cosmetic: manual-mode run
    names do not encode the model (it lives in ``--model``, not a name marker), so
    a ``quantum`` and a ``quantum_shared`` run with identical architecture knobs
    share the same source run-name string. Merging all three source sweeps into
    one sweep dir would collide (and silently clobber) those two runs' directories
    without the prefix. ``__<mode>`` is sweep.py's marker spelling (``__qsvt``).
    """
    return f"{model}__{original_run_name}__{mode}"


def rewrite_run_args(
    args: list[str], mode: str, *, model: str, runs_root: str, target_epochs: int,
) -> list[str]:
    """Original argv rewritten for one fresh qsvt-mode run (never resumed).

    Injects ``--poly-mode``, rewrites ``--run-name`` to the ``<model>__…__<mode>``
    form (``arm_run_name``), repoints ``--runs-root``, normalises ``--epochs``, and
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
        out[i] = arm_run_name(out[i], mode, model)
    else:  # pragma: no cover - source argv always carries --run-name
        out.extend(["--run-name", arm_run_name("run", mode, model)])
    _set("--poly-mode", mode)
    _set("--runs-root", runs_root)
    _set("--epochs", str(target_epochs))

    if "--resume" in out:
        i = out.index("--resume")
        del out[i:i + 2]
    return out


def apply_gpu_remap(source_runs: list[dict], remap: dict[str, str]) -> list[dict]:
    """Return ``source_runs`` with each ``gpu`` field relabelled via ``remap``.

    A shallow copy per run (the run-list file / source manifests are untouched);
    a GPU not in ``remap`` is left as-is.
    """
    return [{**src, "gpu": remap.get(src["gpu"], src["gpu"])} for src in source_runs]


def build_manifest(
    source_runs: list[dict],
    *,
    sweep_dir: Path,
    modes: tuple[str, ...] = DEFAULT_POLY_MODES,
    target_epochs: int = DEFAULT_EPOCHS,
) -> dict:
    """Fan the resolved source runs over ``modes`` into a sweep manifest.

    ``source_runs`` are ``{run_name, gpu, model, args}`` (from
    ``resolve_source_runs``, after ``apply_gpu_remap``). Runs are grouped by target
    GPU in ``GPU_ORDER`` so each group is a contiguous ``index`` range; within a
    group, source order is preserved and the modes are emitted in ``modes`` order.
    The returned ``slurm_groups`` maps each present GPU to its ``[lo, hi]``
    inclusive index range.
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
            for mode in modes:
                entries.append({
                    "index": len(entries),
                    "run_name": arm_run_name(src["run_name"], mode, model),
                    "source_run_name": src["run_name"],
                    "model": model,
                    "poly_mode": mode,
                    "gpu": gpu,
                    "args": rewrite_run_args(
                        src["args"], mode, model=model, runs_root=runs_root,
                        target_epochs=target_epochs,
                    ),
                })
        slurm_groups[gpu] = [lo, len(entries) - 1]

    # Uniqueness guard: two entries with the same run_name would write into one
    # run dir and silently clobber each other's checkpoints/history. The model
    # prefix already separates same-arch quantum vs shared configs; this catches
    # what it can't — a selection list that names the same (model, config) twice,
    # or a modes list with a repeat.
    names = [e["run_name"] for e in entries]
    dups = sorted({n for n in names if names.count(n) > 1})
    if dups:
        raise ValueError(
            f"duplicate run name(s) in the qsvt plan: {dups} — the selection list "
            "names the same (model, config) more than once (or --poly-modes repeats "
            "a value); de-duplicate before building."
        )

    return {
        "sweep_name": sweep_dir.name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        # Launch provenance (CONTEXT.md: Invocation).
        "invocations": [invocation_record()],
        "sweep_dir": runs_root,
        "poly_modes": list(modes),
        "target_epochs": target_epochs,
        "slurm_groups": slurm_groups,
        "n_runs": len(entries),
        "runs": entries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the qsvt polynomial-mode sweep: 16 curated configs × "
        "{qsvt}, fresh from scratch (the 'standard' baseline is reused, not "
        "regenerated; ADR-0009).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source-runs-file", type=Path, default=Path(DEFAULT_SOURCE_RUNS_FILE),
        help="curated-config selection list (run_name / source sweep dir / GPU)",
    )
    parser.add_argument(
        "--poly-modes", type=str, nargs="+", default=list(DEFAULT_POLY_MODES),
        choices=["standard", "qsvt"],
        help="polynomial mode(s) to train from scratch (default: qsvt)",
    )
    parser.add_argument(
        "--gpu-remap", type=str, action="append", metavar="OLD=NEW", default=None,
        help="relabel a source GPU before grouping (repeatable). Defaults to "
        f"{DEFAULT_GPU_REMAP} (H100-96 MIG-split too small → H200-141); a spec "
        "here overrides the default for that GPU.",
    )
    parser.add_argument(
        "--sweeps-root", type=Path, default=Path("results/sweeps"),
        help="parent dir for the new qsvt_extended_<ts> sweep dir",
    )
    parser.add_argument(
        "--sweep-name", type=str, default="qsvt_extended",
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

    # GPU remap: the baked-in default, overridden/extended by any --gpu-remap spec.
    remap = dict(DEFAULT_GPU_REMAP)
    for spec in args.gpu_remap or []:
        if "=" not in spec:
            parser.error(f"--gpu-remap expects OLD=NEW, got {spec!r}")
        old, new = spec.split("=", 1)
        remap[old] = new

    rows = parse_source_runs_file(args.source_runs_file)
    source_runs = apply_gpu_remap(resolve_source_runs(rows), remap)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    sweep_dir = args.sweeps_root / f"{args.sweep_name}_{timestamp}"
    manifest = build_manifest(
        source_runs, sweep_dir=sweep_dir,
        modes=tuple(args.poly_modes), target_epochs=args.epochs,
    )

    sweep_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = sweep_dir / "sweep_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Include list for the restricted comparison: exactly the qsvt runs + their
    # 16 source (standard-mode) configs, so report_sweep_compare --include-file
    # keeps only these (and drops the other archs/seeds that live in the
    # high_epoch_* baseline sweeps).
    include_names = [e["run_name"] for e in manifest["runs"]]
    include_names += [src["run_name"] for src in source_runs]
    include_path = sweep_dir / "compare_include.txt"
    with open(include_path, "w") as f:
        f.write(
            "# Restricted-comparison include list: qsvt runs + their 16 source\n"
            "# (standard-mode) configs. Pass to report_sweep_compare --include-file.\n"
        )
        f.write("\n".join(include_names) + "\n")

    print(f"qsvt sweep: {manifest['sweep_name']}  ({manifest['n_runs']} runs)")
    print(f"  source runs:   {args.source_runs_file}  ({len(source_runs)} configs)")
    print(f"  poly modes:    {manifest['poly_modes']}")
    print(f"  gpu remap:     {remap}")
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

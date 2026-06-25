"""Build (and launch) the coefficient-ablation sweep.

A CV-Quixer head learns two sets of combination coefficients: the per-position
LCU coefficients b_i (M = Σ_i b_i U_i) and the per-degree polynomial coefficients
c_j (P(M) = Σ_j c_j M^j). The ``coeff_ablation`` knob (CONTEXT.md "Coefficient
ablation", ADR-0008) freezes them to fixed uniform values to ablate the learned
weighting structure: ``lcu`` freezes b_i = 1/N; ``lcu_poly`` additionally freezes
c_j = 1.

This builder fans the **16 curated epoch-extension configs** (the ones in
``results/extended_runs_25ep.txt``, spanning all three quantum models) over the
``{lcu, lcu_poly}`` arms into one fresh-from-scratch **32-run** sweep.

The full-coefficient **``none`` baseline is deliberately NOT generated** — it is
reused from the existing ``high_epoch_*`` runs and compared via
``report_sweep_compare.py`` / a notebook (ADR-0008), so this builder only emits
``lcu`` / ``lcu_poly``.

For each of the 16 source argv (read verbatim from the source sweep's
``sweep_manifest.json`` ``runs[].args``) × ``{lcu, lcu_poly}`` it:

  * injects ``--coeff-ablation <arm>``,
  * rewrites ``--run-name`` to append ``__ca<arm>`` (sweep.py's marker spelling),
  * repoints ``--runs-root`` at the new ``results/sweeps/coeff_ablation_<ts>`` dir,
  * normalises ``--epochs`` to 10 (quantum/shared sources are already 10; the
    stacked sources carry ``--epochs 3`` and must be rewritten),
  * drops any ``--resume`` (every run starts from scratch),
  * keeps everything else verbatim (``--model``, ``--gate-param-bound auto``,
    ``--subset-seed 42``, fractions, all arch flags).

Freezing coefficients does not change Fock-sim memory (set by num_heads ×
cutoff^num_modes), so the GPU/wall map from ``results/extended_runs_25ep.txt``
carries over unchanged: a100-40 for quantum/shared/stacked-nm2, h100-96 for the
heavy stacked-nm3 runs. Runs are ordered so each target-GPU group is a contiguous
index range, so the two SLURM array slices can be submitted separately over the
one manifest (the schema ``scripts/run_sweep.sh`` consumes unchanged).

Examples
--------
Inspect the 32-run plan only::

    uv run python experiments/build_coeff_ablation.py --dry-run

Submit both GPU groups as SLURM array slices::

    uv run python experiments/build_coeff_ablation.py --launch slurm
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

from cv_quixer.provenance import invocation_record

FULL_EXPERIMENT = "experiments/full_experiment.py"
RUN_SWEEP_SH = "scripts/run_sweep.sh"

# The two coefficient-ablation arms this sweep trains from scratch. The 'none'
# baseline is reused (not regenerated; ADR-0008).
ARMS = ("lcu", "lcu_poly")

# Every run is normalised to this many epochs (stacked sources carry 3).
DEFAULT_EPOCHS = 10

# The curated-config selection list (run_name / source sweep / target GPU).
DEFAULT_SOURCE_RUNS_FILE = "results/extended_runs_25ep.txt"

# Only these GPU types are usable on the cluster (pytorch/CUDA compat). Each
# group is submitted as its own array slice; ``time`` overrides run_sweep.sh's
# 08:00:00 #SBATCH default when the from-scratch 10-epoch wall needs more
# headroom (quantum/shared ran 4.6-8.0 h). GPU_ORDER fixes the contiguous group
# layout in the manifest (a100 indices first, then h100, then h200).
GPU_CONFIG: dict[str, dict[str, str]] = {
    "a100-40": {"gres": "gpu:a100-40:1", "time": "12:00:00"},
    "h100-96": {"gres": "gpu:h100-96:1", "time": "08:00:00"},
    "h200-141": {"gres": "gpu:h200-141:1", "time": "08:00:00"},
}
GPU_ORDER = ("a100-40", "h100-96", "h200-141")


def arm_run_name(original_run_name: str, arm: str, model: str = "quantum") -> str:
    """The ablation dir name: ``<model>__<original>__ca<arm>``.

    The ``<model>__`` prefix is **load-bearing**, not cosmetic: manual-mode run
    names do not encode the model (it lives in ``--model``, not a name marker), so
    a ``quantum`` and a ``quantum_shared`` run with identical architecture knobs
    share the same source run-name string. Merging all three source sweeps into
    one ``coeff_ablation`` dir would collide (and silently clobber) those two
    runs' directories without the prefix. ``__ca<arm>`` keeps sweep.py's marker
    spelling.
    """
    return f"{model}__{original_run_name}__ca{arm}"


def _model_from_args(args: list[str]) -> str:
    """The ``--model`` value in a replayed argv, or ``"quantum"`` (the default)."""
    return args[args.index("--model") + 1] if "--model" in args else "quantum"


def parse_source_runs_file(path: Path) -> list[dict]:
    """Parse the curated-config selection list into ``{run_name, sweep_dir, gpu}``.

    The ``results/extended_runs_25ep.txt`` format: blank lines and ``#`` comments
    (including the ``# === section ===`` headers) are ignored; for every other
    line the first three whitespace-delimited tokens are run_name / source sweep
    dir / target GPU (trailing columns — peak mem, est wall, orig acc — ignored).
    """
    rows: list[dict] = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 3:
            raise ValueError(
                f"malformed source-runs line (need run_name sweep_dir gpu): {line!r}"
            )
        rows.append({"run_name": fields[0], "sweep_dir": fields[1], "gpu": fields[2]})
    return rows


def resolve_source_runs(rows: list[dict]) -> list[dict]:
    """Attach each selected run's verbatim argv from its source sweep manifest.

    ``rows`` are ``{run_name, sweep_dir, gpu}`` (from ``parse_source_runs_file``).
    Returns ``{run_name, gpu, model, args}`` with ``args`` read verbatim from the
    source ``sweep_manifest.json``. Raises ``KeyError`` if a selected run is not
    in its source manifest (a stale selection list — fail loud, never silently
    drop).
    """
    manifest_cache: dict[str, dict] = {}
    resolved: list[dict] = []
    for row in rows:
        sweep_dir = row["sweep_dir"]
        if sweep_dir not in manifest_cache:
            with open(Path(sweep_dir) / "sweep_manifest.json") as f:
                manifest_cache[sweep_dir] = json.load(f)
        by_name = {r["run_name"]: r for r in manifest_cache[sweep_dir]["runs"]}
        if row["run_name"] not in by_name:
            raise KeyError(
                f"run {row['run_name']!r} not found in {sweep_dir}/sweep_manifest.json"
            )
        args = list(by_name[row["run_name"]]["args"])
        resolved.append({
            "run_name": row["run_name"],
            "gpu": row["gpu"],
            "model": _model_from_args(args),
            "args": args,
        })
    return resolved


def rewrite_run_args(
    args: list[str], arm: str, *, model: str, runs_root: str, target_epochs: int,
) -> list[str]:
    """Original argv rewritten for one fresh coefficient-ablation run (never resumed).

    Injects ``--coeff-ablation``, rewrites ``--run-name`` to the
    ``<model>__…__ca<arm>`` form (``arm_run_name``), repoints ``--runs-root``,
    normalises ``--epochs``, and strips any ``--resume`` (with its value).
    ``--model`` itself and everything else are replayed verbatim.
    """
    out = list(args)

    def _set(flag: str, value: str) -> None:
        if flag in out:
            out[out.index(flag) + 1] = value
        else:
            out.extend([flag, value])

    if "--run-name" in out:
        i = out.index("--run-name") + 1
        out[i] = arm_run_name(out[i], arm, model)
    else:  # pragma: no cover - source argv always carries --run-name
        out.extend(["--run-name", arm_run_name("run", arm, model)])
    _set("--coeff-ablation", arm)
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
    arms: tuple[str, ...] = ARMS,
    target_epochs: int = DEFAULT_EPOCHS,
) -> dict:
    """Fan the resolved source runs over ``arms`` into a sweep manifest.

    ``source_runs`` are ``{run_name, gpu, args}`` (from ``resolve_source_runs``).
    Runs are grouped by target GPU in ``GPU_ORDER`` so each group is a contiguous
    ``index`` range; within a group, source order is preserved and the arms are
    emitted in ``arms`` order. The returned ``slurm_groups`` maps each present GPU
    to its ``[lo, hi]`` inclusive index range.
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
            for arm in arms:
                entries.append({
                    "index": len(entries),
                    "run_name": arm_run_name(src["run_name"], arm, model),
                    "source_run_name": src["run_name"],
                    "model": model,
                    "coeff_ablation": arm,
                    "gpu": gpu,
                    "args": rewrite_run_args(
                        src["args"], arm, model=model, runs_root=runs_root,
                        target_epochs=target_epochs,
                    ),
                })
        slurm_groups[gpu] = [lo, len(entries) - 1]

    # Uniqueness guard: two entries with the same run_name would write into one
    # run dir and silently clobber each other's checkpoints/history. The model
    # prefix already separates same-arch quantum vs shared configs; this catches
    # what it can't — a selection list that names the same (model, config) twice.
    names = [e["run_name"] for e in entries]
    dups = sorted({n for n in names if names.count(n) > 1})
    if dups:
        raise ValueError(
            f"duplicate run name(s) in the coeff-ablation plan: {dups} — the "
            "selection list names the same (model, config) more than once; "
            "de-duplicate it before building."
        )

    return {
        "sweep_name": sweep_dir.name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        # Launch provenance (CONTEXT.md: Invocation).
        "invocations": [invocation_record()],
        "sweep_dir": runs_root,
        "coeff_ablation_arms": list(arms),
        "target_epochs": target_epochs,
        "slurm_groups": slurm_groups,
        "n_runs": len(entries),
        "runs": entries,
    }


def sbatch_commands(manifest: dict, manifest_path: Path) -> list[str]:
    """One ``sbatch`` array-slice command per GPU group (pure; no side effects).

    Each group's ``[lo, hi]`` range becomes ``--array=lo-hi`` (run_sweep.sh
    selects by ``index``), with the group's ``--gres`` and a ``--time`` override.
    """
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
    """Submit each GPU group as its own array slice; stamp provenance.

    Like ``_orchestration.submit_slurm_array`` but emits one submission per GPU
    group (each needing a different ``--gres`` / ``--time``); the submit + job-id
    parse (``submit_one``) and the atomic invocation write-back
    (``stamp_slurm_provenance``) are the shared `_orchestration` helpers, so the
    list of submissions lands under ``invocations[-1]["slurm"]``.
    """
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
        description="Build the coefficient-ablation sweep: 16 curated configs × "
        "{lcu, lcu_poly}, fresh from scratch (the 'none' baseline is reused, not "
        "regenerated; ADR-0008).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source-runs-file", type=Path, default=Path(DEFAULT_SOURCE_RUNS_FILE),
        help="curated-config selection list (run_name / source sweep dir / GPU)",
    )
    parser.add_argument(
        "--sweeps-root", type=Path, default=Path("results/sweeps"),
        help="parent dir for the new coeff_ablation_<ts> sweep dir",
    )
    parser.add_argument(
        "--sweep-name", type=str, default="coeff_ablation",
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
    manifest = build_manifest(source_runs, sweep_dir=sweep_dir, target_epochs=args.epochs)

    sweep_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = sweep_dir / "sweep_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Coeff-ablation sweep: {manifest['sweep_name']}  ({manifest['n_runs']} runs)")
    print(f"  source runs:   {args.source_runs_file}  ({len(source_runs)} configs)")
    print(f"  arms:          {manifest['coeff_ablation_arms']}")
    print(f"  epochs:        {manifest['target_epochs']}")
    print(f"  manifest:      {manifest_path}")
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

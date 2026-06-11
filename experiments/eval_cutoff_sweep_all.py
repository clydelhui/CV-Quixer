"""Run the cutoff-dim sweep across *every* run in a sweep experiment.

`experiments/eval_cutoff_sweep.py` re-evaluates ONE trained checkpoint at a list
of Fock cutoffs. This orchestrator fans that over every run directory under a
sweep produced by `experiments/sweep.py` — one `eval_cutoff_sweep.py` process per
run — then `experiments/report_cutoff_sweep.py` aggregates the per-run results
into cross-run figures.

It mirrors `experiments/sweep.py`: a manifest is written to
`<sweep-dir>/cutoff_sweep_manifest.json` and `--launch` chooses how to run it —
`local` (sequential subprocesses here), `slurm` (a job array via
`scripts/run_eval_cutoff_sweep_array.sh`, one task per run), or `none`/`--dry-run`
(manifest only).

Every per-run eval is given the SAME `--output-name` (`cutoff_sweep_<ts>`) so all
outputs land deterministically at `<run>/eval/<eval_name>/` and the aggregator can
find them by one name (recorded in the manifest).

Examples
--------
Inspect the plan only::

    uv run python experiments/eval_cutoff_sweep_all.py \\
        --sweep-dir results/sweeps/<sweep>_<ts>/ --dry-run

Local smoke (tiny test override, fast)::

    uv run python experiments/eval_cutoff_sweep_all.py \\
        --sweep-dir results/sweeps/<sweep>_<ts>/ \\
        --launch local --cutoffs 6 8 --test-limit 64

Submit one SLURM array task per run::

    uv run python experiments/eval_cutoff_sweep_all.py \\
        --sweep-dir results/sweeps/<sweep>_<ts>/ --launch slurm

Cost note: cutoff-sweep cost scales with `num_heads` (the quantum width), not the
parameter budget. The default cutoffs are `6 8 10`; adding `12` ~doubles per-run
cost and can bust the array wall time on high-head-count runs — pair it with
`--test-fraction 0.2` there (the recovery trend is unaffected by the smaller set).
"""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
import warnings
from datetime import datetime
from pathlib import Path

from cv_quixer.provenance import invocation_record

# Repo root = this file's grandparent (experiments/ -> repo root).
REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_SCRIPT = "experiments/eval_cutoff_sweep.py"
RUN_ARRAY_SH = "scripts/run_eval_cutoff_sweep_array.sh"


def discover_runs(sweep_dir: Path, checkpoint_name: str) -> list[dict]:
    """Find sweep sub-dirs that `eval_cutoff_sweep.py` can evaluate.

    A run is usable iff it has `checkpoints/<checkpoint_name>`, `config.json`,
    and `subset_indices.npz` (the three things the eval script requires). Others
    are skipped with a warning rather than aborting the whole sweep.
    """
    runs: list[dict] = []
    for run_dir in sorted(p for p in sweep_dir.iterdir() if p.is_dir()):
        ckpt = run_dir / "checkpoints" / checkpoint_name
        missing = [
            name for name, path in (
                (f"checkpoints/{checkpoint_name}", ckpt),
                ("config.json", run_dir / "config.json"),
                ("subset_indices.npz", run_dir / "subset_indices.npz"),
            ) if not path.is_file()
        ]
        if missing:
            warnings.warn(
                f"skipping {run_dir.name}: missing {', '.join(missing)}",
                RuntimeWarning, stacklevel=2,
            )
            continue
        runs.append({"run_name": run_dir.name, "checkpoint": str(ckpt)})
    return runs


def _forwarded_eval_args(args: argparse.Namespace, eval_name: str) -> list[str]:
    """Eval flags shared by every run (everything except --checkpoint)."""
    out = ["--cutoffs", *[str(c) for c in args.cutoffs],
           "--output-name", eval_name,
           "--batch-size", str(args.batch_size),
           "--subset-seed", str(args.subset_seed)]
    if args.eval_splits:
        out += ["--eval-splits", *args.eval_splits]
    if args.dtype is not None:
        out += ["--dtype", args.dtype]
    if args.test_fraction is not None:
        out += ["--test-fraction", str(args.test_fraction)]
    if args.test_limit is not None:
        out += ["--test-limit", str(args.test_limit)]
    if args.train_fraction is not None:
        out += ["--train-fraction", str(args.train_fraction)]
    if args.train_limit is not None:
        out += ["--train-limit", str(args.train_limit)]
    return out


def build_manifest(args: argparse.Namespace, runs: list[dict]) -> dict:
    """Expand the per-run eval invocations into a manifest dict (no side effects)."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    eval_name = f"cutoff_sweep_{timestamp}"
    shared = _forwarded_eval_args(args, eval_name)

    entries: list[dict] = []
    for idx, run in enumerate(runs):
        run_args = ["--checkpoint", run["checkpoint"], *shared]
        entries.append({
            "index": idx,
            "run_name": run["run_name"],
            "checkpoint": run["checkpoint"],
            "args": run_args,
        })

    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        # Launch provenance (CONTEXT.md: Invocation) — the exact
        # eval_cutoff_sweep_all.py command that defined this eval grid.
        "invocations": [invocation_record()],
        "sweep_dir": str(args.sweep_dir),
        "eval_name": eval_name,
        "checkpoint_name": args.checkpoint_name,
        "cutoffs": list(args.cutoffs),
        "eval_splits": list(args.eval_splits),
        "n_runs": len(entries),
        "runs": entries,
    }


def launch_local(manifest: dict) -> int:
    """Run each per-run eval sequentially via subprocess. Returns failure count."""
    failures = 0
    for run in manifest["runs"]:
        cmd = [sys.executable, EVAL_SCRIPT, *run["args"]]
        print(f"\n=== [{run['index'] + 1}/{manifest['n_runs']}] {run['run_name']} ===")
        print("  " + " ".join(cmd))
        if subprocess.run(cmd, cwd=REPO_ROOT).returncode != 0:
            failures += 1
            print(f"  ✗ {run['run_name']} failed")
    return failures


def launch_slurm(manifest: dict, manifest_path: Path) -> dict | None:
    """Submit the per-run evals as a SLURM array (or print the command).

    Returns ``{"sbatch_command", "job_id"}`` for the manifest's invocation
    record, or None when sbatch is unavailable and the command was only printed.
    """
    n = manifest["n_runs"]
    cmd = ["sbatch", f"--array=0-{n - 1}", RUN_ARRAY_SH, str(manifest_path)]
    print("\nSLURM array submission:")
    print("  " + " ".join(cmd))
    if shutil.which("sbatch") is None:
        print(
            "\nsbatch not found on PATH — run the command above on the cluster "
            "login node (from the repo root)."
        )
        return None
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    out = result.stdout.strip()
    if out:
        print("  " + out)
    if result.returncode != 0:
        if result.stderr.strip():
            print("  " + result.stderr.strip())
        raise subprocess.CalledProcessError(result.returncode, cmd)
    # sbatch prints "Submitted batch job <id>"; tolerate other formats.
    job_id = out.split()[-1] if out.startswith("Submitted batch job") else None
    return {"sbatch_command": shlex.join(cmd), "job_id": job_id}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run eval_cutoff_sweep.py over every run in a sweep directory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--sweep-dir", type=Path, required=True,
                        help="sweep directory written by experiments/sweep.py")
    parser.add_argument("--checkpoint-name", type=str, default="final_model.pt",
                        help="checkpoint file under each run's checkpoints/ to eval")
    # --- forwarded eval flags --------------------------------------------
    parser.add_argument("--cutoffs", type=int, nargs="+", default=[6, 8, 10],
                        help="cutoff_dim values (default 6 8 10; D=12 ~doubles "
                             "cost and can bust the array wall on high-head runs)")
    parser.add_argument("--eval-splits", type=str, nargs="+", default=["test"],
                        choices=["train", "test"],
                        help="dataset splits to evaluate at each cutoff")
    parser.add_argument("--dtype", type=str, default=None,
                        choices=["complex64", "complex128"],
                        help="override the quantum dtype (default: match training)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--test-fraction", type=float, default=None,
                        help="OVERRIDE forwarded to every run (re-subset test set)")
    parser.add_argument("--test-limit", type=int, default=None,
                        help="OVERRIDE forwarded to every run (first N test samples)")
    parser.add_argument("--train-fraction", type=float, default=None,
                        help="OVERRIDE forwarded to every run (re-subset train set)")
    parser.add_argument("--train-limit", type=int, default=None,
                        help="OVERRIDE forwarded to every run (first N train samples)")
    parser.add_argument("--subset-seed", type=int, default=42,
                        help="subset-seed forwarded to every run")
    # --- orchestration ---------------------------------------------------
    parser.add_argument("--launch", choices=["local", "slurm", "none"],
                        default="none",
                        help="local: run sequentially here; slurm: submit a job "
                             "array; none: just write the manifest")
    parser.add_argument("--dry-run", action="store_true",
                        help="alias for --launch none (write manifest + plan only)")
    args = parser.parse_args()

    if args.test_fraction is not None and args.test_limit is not None:
        parser.error("--test-fraction and --test-limit are mutually exclusive")
    if args.train_fraction is not None and args.train_limit is not None:
        parser.error("--train-fraction and --train-limit are mutually exclusive")
    if not args.sweep_dir.is_dir():
        parser.error(f"--sweep-dir does not exist: {args.sweep_dir}")

    launch = "none" if args.dry_run else args.launch

    runs = discover_runs(args.sweep_dir, args.checkpoint_name)
    if not runs:
        print(f"No usable runs under {args.sweep_dir} "
              f"(need checkpoints/{args.checkpoint_name} + config.json + "
              "subset_indices.npz per run).")
        sys.exit(1)

    manifest = build_manifest(args, runs)
    manifest_path = args.sweep_dir / "cutoff_sweep_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Cutoff sweep over {manifest['n_runs']} run(s) in {args.sweep_dir}")
    print(f"  checkpoint:  {args.checkpoint_name}")
    print(f"  cutoffs:     {manifest['cutoffs']}")
    print(f"  eval_name:   {manifest['eval_name']}  (-> <run>/eval/<eval_name>/)")
    print(f"  manifest:    {manifest_path}")
    for run in manifest["runs"]:
        print(f"    [{run['index']}] {run['run_name']}")

    if launch == "local":
        failures = launch_local(manifest)
        print(f"\nLocal cutoff sweep finished: "
              f"{manifest['n_runs'] - failures}/{manifest['n_runs']} succeeded.")
        print(f"Aggregate: uv run python experiments/report_cutoff_sweep.py "
              f"--sweep-dir {args.sweep_dir}")
        if failures:
            sys.exit(1)
    elif launch == "slurm":
        submission = launch_slurm(manifest, manifest_path)
        if submission is not None:
            # Close the loop in the invocation record: how the eval grid
            # entered the queue. Atomic replace — the just-submitted array
            # tasks read this file.
            tmp_path = manifest_path.with_suffix(".json.tmp")
            manifest["invocations"][-1]["slurm"] = submission
            with open(tmp_path, "w") as f:
                json.dump(manifest, f, indent=2)
            tmp_path.replace(manifest_path)
        print(f"\nAfter the array finishes, aggregate with:")
        print(f"  uv run python experiments/report_cutoff_sweep.py "
              f"--sweep-dir {args.sweep_dir}")
    else:
        print("\n(manifest written; no runs launched — use --launch local|slurm)")


if __name__ == "__main__":
    main()

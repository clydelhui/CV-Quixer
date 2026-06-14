"""Shared launch plumbing for the run-manifest orchestrators.

`sweep.py`, `resume_sweep.py`, and `eval_cutoff_sweep_all.py` each expand their
own grid into a **run manifest** (see CONTEXT.md: Run manifest) and then launch
it one of three ways. The *expansion* is orchestrator-specific (different argv
per grid point); the *launch* is not. This module owns the launch seam so the
job-id parse and the provenance write-back live in exactly one place.

The run-manifest contract (the seam between these Python orchestrators and the
shell job-array scripts `run_sweep.sh` / `run_eval_cutoff_sweep_array.sh`) —
keep byte-identical, the shell side selects by ``index`` and splits ``args`` one
element per line:

    {
      "n_runs": int,                 # == len(runs)
      "invocations": [ {...}, ... ],  # provenance; slurm stamps invocations[-1]
      "runs": [
        {"index": int,               # SLURM array task id (dense, 0..n_runs-1)
         "run_name": str,             # printed to the job log
         "args": list[str],          # CLI flags replayed verbatim by the entry point
         ...},                        # orchestrators add their own extra keys
        ...
      ],
    }

Torch-free on purpose (json/subprocess/pathlib only) so importing it never
breaks the orchestrators' deferred-torch fast path.
"""

from __future__ import annotations

import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

# Repo root = this file's grandparent (experiments/ -> repo root). The entry and
# array scripts are repo-relative, so every subprocess runs with this as cwd.
REPO_ROOT = Path(__file__).resolve().parent.parent


def launch_local(manifest: dict, entry_script: str) -> int:
    """Run each manifest entry sequentially as ``python <entry_script> <args>``.

    ``entry_script`` is the repo-relative python entry point each run drives
    (e.g. ``experiments/full_experiment.py``). Returns the failure count (number
    of runs that exited non-zero); the caller decides what to do with it.
    """
    failures = 0
    n = manifest["n_runs"]
    for run in manifest["runs"]:
        cmd = [sys.executable, entry_script, *run["args"]]
        print(f"\n=== [{run['index'] + 1}/{n}] {run['run_name']} ===")
        print("  " + " ".join(cmd))
        result = subprocess.run(cmd, cwd=REPO_ROOT)
        if result.returncode != 0:
            failures += 1
            print(f"  ✗ {run['run_name']} exited with code {result.returncode}")
    return failures


def array_command(manifest: dict, manifest_path: Path, array_script: str) -> list[str]:
    """The ``sbatch`` job-array command for this manifest (pure; no side effects).

    One array task per run: ``--array=0-{n_runs-1}`` matches the dense ``index``
    values the shell array script selects on. ``array_script`` is the
    repo-relative `.sh` that runs one task (e.g. ``scripts/run_sweep.sh``).
    """
    n = manifest["n_runs"]
    return ["sbatch", f"--array=0-{n - 1}", array_script, str(manifest_path)]


def submit_slurm_array(
    manifest: dict, manifest_path: Path, array_script: str
) -> dict | None:
    """Submit the manifest as a SLURM array and close its provenance loop.

    Prints the `sbatch` command, then — if `sbatch` is on PATH — submits it,
    parses the job id, and atomically stamps ``slurm: {sbatch_command, job_id}``
    onto the manifest's last invocation record (the just-submitted array tasks
    read this file, hence the `.json.tmp` replace). When `sbatch` is absent the
    command is only printed (for hand-running on the login node) and the
    manifest is left untouched.

    Returns the recorded ``{"sbatch_command", "job_id"}`` dict, or None when
    nothing was submitted.
    """
    cmd = array_command(manifest, manifest_path, array_script)
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
    submission = {"sbatch_command": shlex.join(cmd), "job_id": job_id}

    # Close the loop in the invocation record: how the manifest entered the
    # queue, not just how it was defined. Atomic replace — the just-submitted
    # array tasks read this file.
    manifest["invocations"][-1]["slurm"] = submission
    tmp_path = manifest_path.with_suffix(".json.tmp")
    with open(tmp_path, "w") as f:
        json.dump(manifest, f, indent=2)
    tmp_path.replace(manifest_path)
    return submission

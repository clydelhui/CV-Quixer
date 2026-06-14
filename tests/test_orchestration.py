"""Tests for experiments/_orchestration.py — the shared launch/manifest seam.

Pins the run-manifest launch contract the three orchestrators (sweep.py,
resume_sweep.py, eval_cutoff_sweep_all.py) delegate to: the local subprocess
fan-out counts failures, the SLURM array command is built from n_runs, and a
successful submission stamps the parsed job id back onto the manifest's last
invocation record (atomically, since the array tasks read that file).

No torch, no cluster — subprocess is monkeypatched.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

# _orchestration lives in experiments/ (not a package); import it the same way
# the other experiment-script tests reach their module — via sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

import _orchestration  # noqa: E402


def _manifest(n: int) -> dict:
    """A minimal run manifest with n densely-indexed runs + one invocation."""
    return {
        "n_runs": n,
        "invocations": [{"argv": ["x"], "command": "uv run python x"}],
        "runs": [
            {"index": i, "run_name": f"run{i}", "args": ["--flag", str(i)]}
            for i in range(n)
        ],
    }


# --- array_command (pure) ------------------------------------------------


def test_array_command_sizes_array_to_n_runs(tmp_path):
    """--array=0-{n-1} matches the dense index range the shell selects on, and
    the passed array script + manifest path are threaded through verbatim."""
    manifest = _manifest(4)
    manifest_path = tmp_path / "sweep_manifest.json"

    cmd = _orchestration.array_command(manifest, manifest_path, "scripts/run_sweep.sh")

    assert cmd == [
        "sbatch", "--array=0-3", "scripts/run_sweep.sh", str(manifest_path)
    ]


# --- launch_local --------------------------------------------------------


def test_launch_local_runs_each_entry_and_counts_failures(tmp_path, monkeypatch):
    """Every run is invoked as `python <entry_script> <args>` from the repo
    root; the return value is the count of non-zero exits."""
    calls = []

    def fake_run(cmd, cwd=None):
        calls.append((cmd, cwd))
        # Fail the middle run (index 1) to exercise the failure tally.
        rc = 1 if cmd[-1] == "1" else 0
        return subprocess.CompletedProcess(cmd, rc)

    monkeypatch.setattr(_orchestration.subprocess, "run", fake_run)

    failures = _orchestration.launch_local(_manifest(3), "experiments/full_experiment.py")

    assert failures == 1
    assert len(calls) == 3
    # Each call drives the requested entry script from the repo root.
    for cmd, cwd in calls:
        assert cmd[0] == sys.executable
        assert cmd[1] == "experiments/full_experiment.py"
        assert cwd == _orchestration.REPO_ROOT


# --- submit_slurm_array --------------------------------------------------


def test_submit_records_job_id_into_last_invocation(tmp_path, monkeypatch):
    """A successful sbatch submission parses 'Submitted batch job <id>' and
    atomically stamps slurm:{sbatch_command, job_id} onto invocations[-1] of
    the on-disk manifest (the array tasks read that file)."""
    manifest = _manifest(2)
    manifest_path = tmp_path / "sweep_manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    monkeypatch.setattr(_orchestration.shutil, "which", lambda _: "/usr/bin/sbatch")

    def fake_run(cmd, cwd=None, capture_output=False, text=False):
        return subprocess.CompletedProcess(cmd, 0, stdout="Submitted batch job 98765\n", stderr="")

    monkeypatch.setattr(_orchestration.subprocess, "run", fake_run)

    submission = _orchestration.submit_slurm_array(
        manifest, manifest_path, "scripts/run_sweep.sh"
    )

    assert submission == {
        "sbatch_command": "sbatch --array=0-1 scripts/run_sweep.sh " + str(manifest_path),
        "job_id": "98765",
    }
    # Stamped both in memory and atomically on disk.
    assert manifest["invocations"][-1]["slurm"] == submission
    on_disk = json.loads(manifest_path.read_text())
    assert on_disk["invocations"][-1]["slurm"] == submission
    # No leftover temp file.
    assert not (tmp_path / "sweep_manifest.json.tmp").exists()


def test_submit_without_sbatch_prints_only_and_leaves_manifest_untouched(
    tmp_path, monkeypatch
):
    """When sbatch is absent, the command is printed for hand-running and the
    manifest is not modified (no half-recorded submission)."""
    manifest = _manifest(2)
    manifest_path = tmp_path / "sweep_manifest.json"
    original = json.dumps(manifest)
    manifest_path.write_text(original)

    monkeypatch.setattr(_orchestration.shutil, "which", lambda _: None)

    submission = _orchestration.submit_slurm_array(
        manifest, manifest_path, "scripts/run_sweep.sh"
    )

    assert submission is None
    assert "slurm" not in manifest["invocations"][-1]
    assert manifest_path.read_text() == original


def test_submit_raises_on_sbatch_failure(tmp_path, monkeypatch):
    """A non-zero sbatch exit surfaces as CalledProcessError (the queue rejected
    the job) rather than silently recording a bogus submission."""
    import pytest

    manifest = _manifest(1)
    manifest_path = tmp_path / "sweep_manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    monkeypatch.setattr(_orchestration.shutil, "which", lambda _: "/usr/bin/sbatch")

    def fake_run(cmd, cwd=None, capture_output=False, text=False):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="sbatch: error: invalid")

    monkeypatch.setattr(_orchestration.subprocess, "run", fake_run)

    with pytest.raises(subprocess.CalledProcessError):
        _orchestration.submit_slurm_array(manifest, manifest_path, "scripts/run_sweep.sh")
    assert "slurm" not in manifest["invocations"][-1]

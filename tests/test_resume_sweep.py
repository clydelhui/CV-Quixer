"""Tests for experiments/resume_sweep.py (JSON only — no torch, no training).

These pin the top-up semantics (see CONTEXT.md "Top-up"): raise selected sweep
runs to a shared *target total* epoch count, resuming from latest.pt where one
exists, restarting from scratch where none does, skipping runs already at the
target — all by replaying each run's original argv from sweep_manifest.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# resume_sweep lives in experiments/ (not a package); import it the same way
# the other experiment-script tests do — via sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

import resume_sweep  # noqa: E402


def make_sweep(tmp_path, runs):
    """Write a synthetic sweep dir: sweep_manifest.json + per-run dirs.

    ``runs`` is a list of dicts: ``run_name``, ``args`` (original argv), and
    optionally ``n_epochs`` (history.json epoch count; None = no history.json)
    and ``ckpt`` (whether checkpoints/latest.pt exists).
    """
    sweep_dir = tmp_path / "sweep_test_2026-01-01_00-00-00"
    sweep_dir.mkdir()
    manifest = {
        "sweep_name": "sweep_test",
        "created_at": "2026-01-01T00:00:00",
        "sweep_dir": str(sweep_dir),
        "n_runs": len(runs),
        "runs": [
            {"index": i, "run_name": r["run_name"], "args": list(r["args"])}
            for i, r in enumerate(runs)
        ],
    }
    with open(sweep_dir / "sweep_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    for r in runs:
        run_dir = sweep_dir / r["run_name"]
        run_dir.mkdir()
        n_epochs = r.get("n_epochs")
        if n_epochs is not None:
            history = {
                "epoch": {"test_acc": [0.5 + 0.01 * e for e in range(n_epochs)]},
                "meta": {"seed": 42},
            }
            with open(run_dir / "history.json", "w") as f:
                json.dump(history, f)
        if r.get("ckpt", False):
            ckpt_dir = run_dir / "checkpoints"
            ckpt_dir.mkdir()
            (ckpt_dir / "latest.pt").write_bytes(b"fake")
    return sweep_dir, manifest


BASE_ARGS = [
    "--observables", "xpxsps", "--seed", "42", "--num-layers", "2",
    "--run-name", "p8000__obs-xpxsps__seed42",
    "--runs-root", "results/sweeps/sweep_test_2026-01-01_00-00-00",
    "--epochs", "3", "--train-fraction", "0.1", "--test-fraction", "0.1",
]


def test_resume_entry_rewrites_epochs_and_appends_resume(tmp_path):
    """A run with a checkpoint below the target is resumed: its original argv is
    replayed verbatim except --epochs is rewritten to the target, and
    --resume <run>/checkpoints/latest.pt is appended."""
    sweep_dir, _ = make_sweep(tmp_path, [
        {"run_name": "p8000__obs-xpxsps__seed42", "args": BASE_ARGS,
         "n_epochs": 3, "ckpt": True},
    ])
    manifest = resume_sweep.build_manifest(sweep_dir, target_epochs=6)

    (entry,) = manifest["runs"]
    assert entry["action"] == "resume"
    args = entry["args"]
    # --epochs rewritten in place, not duplicated.
    assert args.count("--epochs") == 1
    assert args[args.index("--epochs") + 1] == "6"
    # --resume appended, pointing at this run's latest.pt.
    assert args[args.index("--resume") + 1] == str(
        sweep_dir / "p8000__obs-xpxsps__seed42" / "checkpoints" / "latest.pt"
    )
    # Everything else preserved verbatim (order included).
    stripped = list(args)
    i = stripped.index("--resume")
    del stripped[i:i + 2]
    expected = list(BASE_ARGS)
    expected[expected.index("--epochs") + 1] = "6"
    assert stripped == expected


def test_epochs_appended_when_original_args_lack_it(tmp_path):
    """sweep.py omits --epochs from run args when the sweep used the default;
    the top-up must then append --epochs <target> rather than crash."""
    no_epochs = [a for i, a in enumerate(BASE_ARGS)
                 if a != "--epochs" and BASE_ARGS[i - 1] != "--epochs"]
    sweep_dir, _ = make_sweep(tmp_path, [
        {"run_name": "p8000__obs-xpxsps__seed42", "args": no_epochs,
         "n_epochs": 3, "ckpt": True},
    ])
    manifest = resume_sweep.build_manifest(sweep_dir, target_epochs=6)

    (entry,) = manifest["runs"]
    args = entry["args"]
    assert args.count("--epochs") == 1
    assert args[args.index("--epochs") + 1] == "6"


def test_runs_already_at_target_are_skipped(tmp_path):
    """A run whose history already records >= target epochs is excluded from
    the launchable entries (it would be a pure no-op burning an array slot)
    but is still recorded in the manifest's skip list."""
    args_a = list(BASE_ARGS)
    args_b = list(BASE_ARGS)
    args_b[args_b.index("--run-name") + 1] = "p13760__obs-xpxsps__seed42"
    sweep_dir, _ = make_sweep(tmp_path, [
        {"run_name": "p8000__obs-xpxsps__seed42", "args": args_a,
         "n_epochs": 6, "ckpt": True},     # already at target
        {"run_name": "p13760__obs-xpxsps__seed42", "args": args_b,
         "n_epochs": 3, "ckpt": True},     # below target
    ])
    manifest = resume_sweep.build_manifest(sweep_dir, target_epochs=6)

    assert [r["run_name"] for r in manifest["runs"]] == [
        "p13760__obs-xpxsps__seed42"
    ]
    assert manifest["n_runs"] == 1
    (skipped,) = manifest["skipped"]
    assert skipped["run_name"] == "p8000__obs-xpxsps__seed42"
    assert skipped["current_epochs"] == 6


def test_run_without_checkpoint_restarts_fresh(tmp_path):
    """A run that died before completing epoch 1 (no latest.pt — provably no
    per-epoch artefacts) is included WITHOUT --resume: it trains from scratch
    into the same run dir with its original argv plus the new --epochs."""
    sweep_dir, _ = make_sweep(tmp_path, [
        {"run_name": "p8000__obs-xpxsps__seed42", "args": BASE_ARGS,
         "n_epochs": None, "ckpt": False},
    ])
    manifest = resume_sweep.build_manifest(sweep_dir, target_epochs=6)

    (entry,) = manifest["runs"]
    assert entry["action"] == "fresh"
    assert entry["current_epochs"] == 0
    assert "--resume" not in entry["args"]
    assert entry["args"][entry["args"].index("--epochs") + 1] == "6"


def _three_run_sweep(tmp_path):
    runs = []
    for name in ("p8000__obs-xpxsps__seed42", "p8000__obs-xpxsps__seed43",
                 "p13760__obs-x__seed42"):
        args = list(BASE_ARGS)
        args[args.index("--run-name") + 1] = name
        runs.append({"run_name": name, "args": args, "n_epochs": 3, "ckpt": True})
    return make_sweep(tmp_path, runs)


def test_runs_patterns_filter_by_fnmatch(tmp_path):
    """--runs patterns restrict selection by fnmatch on run_name; a run matching
    any pattern is selected, non-matching runs are absent entirely (neither
    launched nor listed as skipped)."""
    sweep_dir, _ = _three_run_sweep(tmp_path)
    manifest = resume_sweep.build_manifest(
        sweep_dir, target_epochs=6, patterns=["p8000__*"]
    )

    assert [r["run_name"] for r in manifest["runs"]] == [
        "p8000__obs-xpxsps__seed42", "p8000__obs-xpxsps__seed43"
    ]
    assert manifest["skipped"] == []


def test_launchable_runs_reindexed_densely(tmp_path):
    """With a skip in the middle of the population, the launchable entries are
    re-indexed 0..K-1 (run_sweep.sh selects by index — a gap would strand an
    array task) and each carries the keys the array script expects."""
    names = ("a__seed42", "b__seed42", "c__seed42")
    runs = []
    for name, n_epochs in zip(names, (3, 6, None)):  # b already at target
        args = list(BASE_ARGS)
        args[args.index("--run-name") + 1] = name
        runs.append({"run_name": name, "args": args,
                     "n_epochs": n_epochs, "ckpt": n_epochs is not None})
    sweep_dir, _ = make_sweep(tmp_path, runs)
    manifest = resume_sweep.build_manifest(sweep_dir, target_epochs=6)

    assert [(r["index"], r["run_name"]) for r in manifest["runs"]] == [
        (0, "a__seed42"), (1, "c__seed42")
    ]
    assert manifest["n_runs"] == 2
    for r in manifest["runs"]:
        assert {"index", "run_name", "args"} <= set(r)


def test_main_dry_run_writes_manifest_and_preserves_original(
    tmp_path, monkeypatch, capsys
):
    """--dry-run writes resume_manifest_<ts>.json, launches nothing, leaves
    sweep_manifest.json byte-identical, and prints a per-run status line."""
    sweep_dir, _ = _three_run_sweep(tmp_path)
    original_bytes = (sweep_dir / "sweep_manifest.json").read_bytes()

    monkeypatch.setattr(sys, "argv", [
        "resume_sweep.py", "--sweep-dir", str(sweep_dir),
        "--epochs", "6", "--dry-run",
    ])
    resume_sweep.main()

    assert (sweep_dir / "sweep_manifest.json").read_bytes() == original_bytes
    (manifest_path,) = sorted(sweep_dir.glob("resume_manifest_*.json"))
    with open(manifest_path) as f:
        manifest = json.load(f)
    assert manifest["target_epochs"] == 6
    assert manifest["n_runs"] == 3

    out = capsys.readouterr().out
    for name in ("p8000__obs-xpxsps__seed42", "p8000__obs-xpxsps__seed43",
                 "p13760__obs-x__seed42"):
        assert name in out
    assert "resume" in out  # planned action shown per run


def test_slurm_command_uses_run_sweep_sh_unchanged(tmp_path):
    """The SLURM submission targets the existing scripts/run_sweep.sh (no new
    array script) with --array sized to the re-indexed launchable count. The
    command is built by the shared orchestration seam, passed resume_sweep's
    RUN_SWEEP_SH so a top-up reuses the sweep array script."""
    import _orchestration

    sweep_dir, _ = _three_run_sweep(tmp_path)
    manifest = resume_sweep.build_manifest(sweep_dir, target_epochs=6)
    manifest_path = sweep_dir / "resume_manifest_test.json"

    cmd = _orchestration.array_command(
        manifest, manifest_path, resume_sweep.RUN_SWEEP_SH
    )

    assert cmd == [
        "sbatch", "--array=0-2", "scripts/run_sweep.sh", str(manifest_path)
    ]

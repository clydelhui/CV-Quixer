"""Tests for cv_quixer.provenance — launch-provenance invocation records.

These pin the Invocation schema (see CONTEXT.md "Invocation"): every experiment
entry point appends ``{launched_at, argv, command, hostname, git_sha,
git_dirty}`` to the artefact it owns, with ``command`` a shell-ready re-run
line derived losslessly from ``argv``.
"""

from __future__ import annotations

import shlex
import string
import sys

from cv_quixer.provenance import invocation_record


def test_record_keys_and_argv_verbatim():
    rec = invocation_record()
    assert set(rec) == {
        "launched_at", "argv", "command", "hostname", "git_sha", "git_dirty",
    }
    assert rec["argv"] == sys.argv
    assert rec["hostname"]
    assert rec["launched_at"]


def test_command_is_rerunnable_and_roundtrips():
    rec = invocation_record()
    parts = shlex.split(rec["command"])
    assert parts[:3] == ["uv", "run", "python"]
    # The shell line parses back to exactly the original argv (quoting is
    # lossless even for args with spaces).
    assert parts[3:] == sys.argv


def test_git_fields_resolve_inside_repo():
    # The test suite runs from a checkout, so git metadata must resolve.
    rec = invocation_record()
    assert rec["git_sha"] is not None
    assert len(rec["git_sha"]) == 40
    assert set(rec["git_sha"]) <= set(string.hexdigits.lower())
    assert isinstance(rec["git_dirty"], bool)


def test_sweep_manifest_carries_invocation(monkeypatch):
    # build_manifest embeds the launching command as invocations[0].
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
    import sweep  # noqa: E402

    fake_argv = [
        "experiments/sweep.py", "--num-heads", "4", "--observables", "xpxsps",
    ]
    monkeypatch.setattr(sys, "argv", fake_argv)

    import argparse

    ns = argparse.Namespace(
        target_params=None, observables=["xpxsps"], seeds=[42], num_layers=[1],
        scaling_knob=None, trunc_lambda=None, decoder_hidden_mult=None,
        query_trunc_lambda=None, poly_init_noise=None, positional_encoding=None,
        pooling=None, block_residual=None,
        model="quantum", epochs=None, train_fraction=None, test_fraction=None,
        subset_seed=42, gate_param_bound=None, sweep_name="sweep_test",
        sweeps_root="results/sweeps", wandb=False,
        **{dest: ([4] if dest == "num_heads" else None)
           for dest, *_ in sweep.ARCH_AXES},
    )
    manifest = sweep.build_manifest(ns)
    (entry,) = manifest["invocations"]
    assert entry["argv"] == fake_argv
    assert entry["command"] == "uv run python " + shlex.join(fake_argv)

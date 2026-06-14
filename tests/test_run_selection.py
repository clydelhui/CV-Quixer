"""Tests for experiments/_run_selection.py — the shared coordinate filter.

Pins the locked semantics (CONTEXT.md "Coordinate filter", ADR-0006): OR within
a flag, AND across flags, absent coordinate -> exclude-with-warning; plus the
two extraction helpers and the registry/flag drift guards. JSON only, no torch.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

# _run_selection lives in experiments/ (not a package); import via sys.path like
# the other experiment-script tests.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

import _run_selection as rs  # noqa: E402


# --- run_matches semantics ---------------------------------------------------

def test_empty_filter_keeps_every_run():
    assert rs.run_matches({"num_modes": 99}, {}) is True
    assert rs.run_matches({}, {}) is True


def test_or_within_a_field():
    assert rs.run_matches({"num_modes": 2}, {"num_modes": {2, 3}}) is True
    assert rs.run_matches({"num_modes": 3}, {"num_modes": {2, 3}}) is True
    assert rs.run_matches({"num_modes": 4}, {"num_modes": {2, 3}}) is False


def test_and_across_fields():
    coords = {"num_modes": 2, "num_heads": 5}
    assert rs.run_matches(coords, {"num_modes": {2, 3}, "num_heads": {5, 10}}) is True
    # num_heads fails -> whole match fails even though num_modes passes.
    assert rs.run_matches(coords, {"num_modes": {2, 3}, "num_heads": {10}}) is False


def test_absent_coordinate_excludes_with_warning():
    with pytest.warns(RuntimeWarning, match="num_seq2seq_blocks"):
        result = rs.run_matches(
            {"num_modes": 2}, {"num_seq2seq_blocks": {2}}, run_name="r0"
        )
    assert result is False


def test_string_field_matches():
    assert rs.run_matches({"model": "quantum"}, {"model": {"quantum"}}) is True
    assert rs.run_matches(
        {"model": "quantum_stacked"}, {"model": {"quantum"}}
    ) is False


# --- coords_from_config_json -------------------------------------------------

def _config(quantum=None, training=None, model="quantum"):
    return {
        "model": model,
        "quantum": {"num_modes": 2, "num_heads": 5, **(quantum or {})},
        "training": {"seed": 42, **(training or {})},
    }


def test_config_extracts_nested_paths():
    coords = rs.coords_from_config_json(_config(model="quantum_stacked"))
    assert coords["num_modes"] == 2          # quantum.*
    assert coords["num_heads"] == 5
    assert coords["seed"] == 42              # training.seed
    assert coords["model"] == "quantum_stacked"  # top-level


def test_config_omits_missing_keys_so_filter_reports_unresolved():
    # No num_seq2seq_blocks key (pre-ADR-0003 run) -> absent from coords, not None.
    coords = rs.coords_from_config_json(_config())
    assert "num_seq2seq_blocks" not in coords
    # observables is config-less and never extracted from config.json.
    assert "observables" not in coords


def test_config_normalizes_block_residual_bool_to_string():
    coords = rs.coords_from_config_json(_config(quantum={"block_residual": False}))
    assert coords["block_residual"] == "off"


# --- coords_from_args --------------------------------------------------------

BASE_ARGS = [
    "--target-params", "8000", "--observables", "xpxsps", "--seed", "42",
    "--num-modes", "3", "--num-heads", "6", "--run-name", "x", "--epochs", "3",
]


def test_args_parses_flags_into_typed_coords():
    coords = rs.coords_from_args(BASE_ARGS)
    assert coords["num_modes"] == 3 and isinstance(coords["num_modes"], int)
    assert coords["num_heads"] == 6
    assert coords["observables"] == "xpxsps"  # config-less field, only source
    assert coords["seed"] == 42
    assert coords["target_params"] == 8000


def test_args_ignores_unrelated_flags():
    coords = rs.coords_from_args(BASE_ARGS)
    assert "run_name" not in coords and "epochs" not in coords


# --- argparse wiring ---------------------------------------------------------

def test_add_and_parse_filter_args_roundtrip():
    parser = argparse.ArgumentParser()
    rs.add_filter_args(parser)
    ns = parser.parse_args(["--num-modes", "2", "3", "--model", "quantum"])
    filters = rs.parse_filter_args(ns)
    assert filters == {"num_modes": {2, 3}, "model": {"quantum"}}
    # Unset flags are omitted entirely (no filter on them).
    assert "num_heads" not in filters


def test_parse_filter_args_empty_when_no_flags_set():
    parser = argparse.ArgumentParser()
    rs.add_filter_args(parser)
    ns = parser.parse_args([])
    assert rs.parse_filter_args(ns) == {}


# --- registry drift guards ---------------------------------------------------

def test_flag_spelling_matches_dash_convention():
    for field in rs.FILTERABLE_FIELDS:
        assert rs.flag_for(field.name) == "--" + field.name.replace("_", "-")


def test_registry_is_superset_of_sweep_arch_axes():
    """Drift guard: every manual sweep axis must be filterable (sweep.ARCH_AXES
    is the third copy of the field list — keep it covered by the registry)."""
    import sweep

    arch_dests = {dest for dest, *_ in sweep.ARCH_AXES}
    registry = {f.name for f in rs.FILTERABLE_FIELDS}
    assert arch_dests <= registry, arch_dests - registry

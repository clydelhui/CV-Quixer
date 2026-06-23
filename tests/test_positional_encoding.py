"""Tests for the positional-encoding type knob (--positional-encoding).

The CNN hypernetwork adds a positional encoding to its flattened conv features
before projecting to gate parameters. Historically this was a hardcoded 2D
sinusoidal PE (row/col split). This knob ablates it across three variants:

  * ``"2d"`` — the historic row/col-split sinusoid (the default; byte-identical);
  * ``"1d"`` — a standard sinusoid over the flattened (row-major) patch index,
    spanning the full feature_dim (ViT-style, no grid awareness);
  * ``"none"`` — an all-zeros buffer, so the additive PE is a no-op.

Load-bearing invariants pinned here:

  * ``"none"`` produces an all-zeros buffer (the add becomes a no-op);
  * ``"1d"``, ``"2d"`` and ``"none"`` are pairwise distinct;
  * the default ``"2d"`` is byte-identical to the pre-knob helper;
  * ``"1d"`` does not require a perfect-square patch count (relaxes the 2d
    assertion), but ``"2d"`` still does;
  * the knob threads to every model variant's PE buffer and changes no
    state_dict keys;
  * positional_encoding is a registered configuration coordinate.

Small circuits (num_modes=2, cutoff_dim=4) keep simulation tractable.
"""

import sys
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

# _run_selection / sweep / report_sweep live in experiments/ (not a package) —
# import via sys.path like the other experiment-script tests.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

from cv_quixer.config.schema import DataConfig, ExperimentConfig, QuantumConfig
from cv_quixer.config.utils import experiment_config_from_dict
from cv_quixer.models.quantum.cv_attention import (
    _compute_2d_sinusoidal_pe,
    _compute_positional_encoding,
)


class TestPositionalEncodingHelper:
    def test_none_is_all_zeros(self):
        pe = _compute_positional_encoding(num_patches=16, feature_dim=16, pe_type="none")
        assert pe.shape == (16, 16)
        assert torch.equal(pe, torch.zeros(16, 16)), "'none' must be an all-zeros buffer"

    def test_variants_are_pairwise_distinct(self):
        none = _compute_positional_encoding(16, 16, "none")
        one = _compute_positional_encoding(16, 16, "1d")
        two = _compute_positional_encoding(16, 16, "2d")
        assert not torch.equal(none, one)
        assert not torch.equal(none, two)
        assert not torch.equal(one, two), "1d and 2d must differ"

    def test_2d_default_is_byte_identical_to_pre_knob_helper(self):
        # The historic helper is the golden: the default path must reproduce it
        # exactly so existing checkpoints + the 2D baseline are untouched.
        golden = _compute_2d_sinusoidal_pe(16, 16)
        assert torch.equal(_compute_positional_encoding(16, 16), golden)
        assert torch.equal(_compute_positional_encoding(16, 16, "2d"), golden)

    def test_1d_does_not_require_square_patch_count(self):
        # 1d treats patches as a flat sequence, so a non-square count is fine —
        # whereas 2d still asserts a perfect square.
        pe = _compute_positional_encoding(num_patches=8, feature_dim=16, pe_type="1d")
        assert pe.shape == (8, 16)
        with pytest.raises(AssertionError):
            _compute_positional_encoding(num_patches=8, feature_dim=16, pe_type="2d")

    def test_invalid_pe_type_raises(self):
        with pytest.raises(ValueError, match="positional_encoding"):
            _compute_positional_encoding(16, 16, "3d")


class TestConfigField:
    def test_default_is_2d(self):
        assert QuantumConfig().positional_encoding == "2d"

    def test_invalid_value_is_rejected(self):
        with pytest.raises(ValueError, match="positional_encoding"):
            QuantumConfig(positional_encoding="3d")

    def test_old_config_without_key_loads_as_2d(self):
        # Pre-knob runs have no positional_encoding key. Like poly_init_noise's
        # off path, an absent key reloads silently as the historic behaviour
        # ("2d") — byte-identical, so no migration is needed.
        cfg = ExperimentConfig(
            model="quantum",
            data=DataConfig(image_size=14, patch_size=7, num_classes=10),
            quantum=QuantumConfig(num_modes=2, cutoff_dim=4, num_heads=2),
        )
        raw = asdict(cfg)
        raw["quantum"].pop("positional_encoding")
        reconstructed = experiment_config_from_dict(raw)  # must not raise
        assert reconstructed.quantum.positional_encoding == "2d"


def _quantum_config(positional_encoding: str = "2d", **overrides) -> QuantumConfig:
    base = dict(
        num_modes=2, num_layers=1, cutoff_dim=4, num_heads=2,
        cnn_channels_1=4, cnn_channels_2=8, cnn_kernel_size=3,
        decoder_hidden_dim=16, poly_degree=2, dtype="complex64",
        cvqnn_num_layers=0, positional_encoding=positional_encoding,
    )
    base.update(overrides)
    return QuantumConfig(**base)


def _pos_enc_buffers(model) -> dict:
    """All PE buffers in a model, keyed by their dotted state-dict path."""
    return {
        name: buf for name, buf in model.named_buffers() if name.endswith("pos_enc")
    }


class TestModelThreading:
    def _build(self, tiny_data_config, positional_encoding, model="quantum"):
        from cv_quixer.models import build_model
        return build_model(ExperimentConfig(
            model=model, data=tiny_data_config,
            quantum=_quantum_config(positional_encoding),
        ))

    @pytest.mark.parametrize("model", ["quantum", "quantum_shared", "quantum_stacked"])
    def test_pe_type_reaches_buffers(self, tiny_data_config, model):
        # Each variant exposes at least one pos_enc buffer (block-1 CNN), and the
        # buffer content changes with the knob across all three models.
        none = _pos_enc_buffers(self._build(tiny_data_config, "none", model))
        two = _pos_enc_buffers(self._build(tiny_data_config, "2d", model))
        assert none and set(none) == set(two), "every variant must own pos_enc buffers"
        for key in none:
            assert torch.equal(none[key], torch.zeros_like(none[key])), \
                f"'none' must zero {key}"
            assert not torch.equal(none[key], two[key]), \
                f"'2d' must differ from 'none' at {key}"

    def test_state_dict_keys_unchanged_by_pe_type(self, tiny_data_config):
        off = self._build(tiny_data_config, "none")
        two = self._build(tiny_data_config, "2d")
        one = self._build(tiny_data_config, "1d")
        assert set(off.state_dict()) == set(two.state_dict()) == set(one.state_dict()), (
            "positional_encoding must not add/rename state_dict keys (still pos_enc)"
        )


class TestCoordinateRegistration:
    def test_filterable_field_registered(self):
        import _run_selection as rs

        field = rs._FIELDS_BY_NAME.get("positional_encoding")
        assert field is not None, "positional_encoding must be a filterable coordinate"
        assert field.py_type is str
        assert field.config_path == ("quantum", "positional_encoding")

    def test_report_treats_it_as_config_identity(self):
        import report_sweep

        assert "positional_encoding" in report_sweep.CONFIG_IDENTITY_FIELDS, (
            "none / 1d / 2d runs of the same architecture must NOT be seed-averaged "
            "together — positional_encoding has to be a configuration-identity field"
        )


class TestSweepAxis:
    def _manifest(self, *argv):
        import sweep
        return sweep.build_manifest(sweep.build_parser().parse_args(list(argv)))

    def test_positional_encoding_is_a_grid_axis(self):
        manifest = self._manifest(
            "--observables", "xpxsps", "--positional-encoding", "none", "1d", "2d",
        )
        assert manifest["n_runs"] == 3, "one run per PE variant"
        assert manifest["axes"]["positional_encoding"] == ["none", "1d", "2d"]

    def test_run_name_markers_and_forwarded_flag(self):
        manifest = self._manifest(
            "--observables", "xpxsps", "--positional-encoding", "none", "1d", "2d",
        )
        markers = {}
        for run in manifest["runs"]:
            pe = run["positional_encoding"]
            markers[pe] = run["run_name"]
            args = run["args"]
            assert args[args.index("--positional-encoding") + 1] == pe
        assert "__penone" in markers["none"]
        assert "__pe1d" in markers["1d"]
        assert "__pe2d" in markers["2d"]

    def test_absent_axis_emits_no_flag_or_marker(self):
        # Inheriting full_experiment's "2d" default must be byte-identical to a
        # no-axis sweep: no --positional-encoding flag, no __pe marker.
        manifest = self._manifest("--observables", "xpxsps", "--num-heads", "4")
        (run,) = manifest["runs"]
        assert "__pe" not in run["run_name"]
        assert "--positional-encoding" not in run["args"]

    def test_standalone_positional_encoding_sweep_is_allowed(self, tmp_path, monkeypatch):
        # positional_encoding is a valid sole manual axis: main()'s
        # under-specified guard must accept it like any other axis.
        import sweep
        monkeypatch.setattr(sys, "argv", [
            "sweep.py", "--positional-encoding", "none", "1d",
            "--observables", "xpxsps", "--sweeps-root", str(tmp_path), "--dry-run",
        ])
        sweep.main()  # must not SystemExit on the "specify an axis" guard

"""Tests for the observable-preset registry and its interaction with the
config system + parameter-budget auto-scaling (the two sweep axes).
"""

import dataclasses
import json
import warnings

import dacite
import pytest

from cv_quixer.config.observable_presets import PRESET_NAMES, resolve_observables
from cv_quixer.config.schema import (
    ExperimentConfig,
    ObservableSpec,
    QuantumConfig,
)

CUTOFF = 6


class TestResolveObservables:
    @pytest.mark.parametrize("name", PRESET_NAMES)
    def test_resolves_to_valid_specs(self, name):
        specs = resolve_observables(name, CUTOFF)
        assert specs and all(isinstance(s, ObservableSpec) for s in specs)
        # Building a QuantumConfig exercises __post_init__ validation/expansion.
        cfg = QuantumConfig(num_modes=2, cutoff_dim=CUTOFF, readout_observables=specs)
        assert len(cfg._observable_plan) >= 2

    def test_unknown_preset_raises(self):
        with pytest.raises(ValueError, match="Unknown observable preset"):
            resolve_observables("not_a_preset", CUTOFF)

    def test_pnr_plan_scales_with_cutoff(self):
        cfg = QuantumConfig(
            num_modes=2, cutoff_dim=CUTOFF,
            readout_observables=resolve_observables("pnr", CUTOFF),
        )
        assert len(cfg._observable_plan) == 2 * CUTOFF  # num_modes × cutoff

    def test_xpxsps_plan_length(self):
        cfg = QuantumConfig(
            num_modes=2, cutoff_dim=CUTOFF,
            readout_observables=resolve_observables("xpxsps", CUTOFF),
        )
        assert len(cfg._observable_plan) == 2 * 4  # 4 observable types × num_modes

    def test_returns_fresh_list(self):
        """Each call returns an independent list (safe to mutate / reuse)."""
        a = resolve_observables("xp", CUTOFF)
        b = resolve_observables("xp", CUTOFF)
        assert a is not b


class TestConfigRoundTrip:
    @pytest.mark.parametrize("name", PRESET_NAMES)
    def test_dacite_round_trip(self, name, data_config):
        """A preset survives the asdict→JSON→dacite round-trip used to persist
        and reload config.json (same path as eval_cutoff_sweep / report_diagnostics).
        """
        cfg = ExperimentConfig(
            quantum=QuantumConfig(
                num_modes=2, cutoff_dim=CUTOFF,
                readout_observables=resolve_observables(name, CUTOFF),
            ),
            data=data_config,
        )
        raw = json.loads(json.dumps(dataclasses.asdict(cfg)))
        restored = dacite.from_dict(
            data_class=ExperimentConfig,
            data=raw,
            config=dacite.Config(strict=False),
        )
        assert len(restored.quantum._observable_plan) == len(
            cfg.quantum._observable_plan
        )


class TestAutoScaleWithPresets:
    @pytest.mark.parametrize("name", ["x", "pnr"])
    def test_target_params_within_tolerance(self, name, data_config):
        """target_params auto-scaling lands within 10% for both a small (`x`)
        and a large (`pnr`) observable plan — the two ends of the budget axis.
        """
        from cv_quixer.models.quantum.cv_quixer import CVQuixer
        from cv_quixer.utils.params import count_parameters

        target = 13_760
        cfg = QuantumConfig(
            num_modes=2, cutoff_dim=CUTOFF, num_heads=4,
            cnn_channels_1=8, cnn_kernel_size=3, decoder_hidden_dim=32,
            poly_degree=3, dtype="complex64", target_params=target,
            readout_observables=resolve_observables(name, CUTOFF),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = CVQuixer(cfg, data_config)
        achieved = count_parameters(model)
        assert abs(achieved - target) / target < 0.10

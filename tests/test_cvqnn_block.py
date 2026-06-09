"""Tests for the CVQNN block W (post-polynomial, pre-readout) and its migration.

W is a fixed, per-head, trainable Killoran circuit applied to the post-selected
polynomial state before observable readout (QuantumConfig.cvqnn_num_layers, ADR-0001).
These tests pin the load-bearing invariants:

  * cvqnn_num_layers=0 ⇒ no W params registered ⇒ state_dict byte-identical to a
    pre-W model (the checkpoint-compat / ablation guarantee);
  * cvqnn_num_layers=1 ⇒ exactly one W param tensor per head added;
  * small-random init keeps the displacement/beamsplitter gate gradients FINITE
    (exact-zero init lands on their NaN-gradient singularity — see ADR-0001);
  * W is non-trivial (changes readouts) and its leakage feeds cvqnn_trunc_loss;
  * the pre-W migration script + loud guard round-trip correctly.

Small circuits (num_modes=2, cutoff_dim=4) keep simulation tractable.
"""

import json
import tempfile
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from cv_quixer.config.schema import DataConfig, ExperimentConfig, QuantumConfig
from cv_quixer.config.utils import experiment_config_from_dict
from cv_quixer.models import build_model
from cv_quixer.models.quantum.cv_attention import (
    _CVQNN_LAYER,
    _GATE_SEQUENCE,
    _INTERFEROMETER_SEQUENCE,
    _build_cvqnn_plan,
)


def _config(cvqnn_num_layers: int, **overrides) -> QuantumConfig:
    base = dict(
        num_modes=2,
        num_layers=1,
        cutoff_dim=4,
        num_heads=2,
        cnn_channels_1=4,
        cnn_channels_2=8,
        cnn_kernel_size=3,
        decoder_hidden_dim=16,
        poly_degree=2,
        dtype="complex64",
        trunc_penalty="norm",
        cvqnn_num_layers=cvqnn_num_layers,
    )
    base.update(overrides)
    return QuantumConfig(**base)


# ---------------------------------------------------------------------------
# Op-plan construction
# ---------------------------------------------------------------------------


class TestBuildCvqnnPlan:
    def test_zero_layers_is_empty(self):
        assert _build_cvqnn_plan(0) == ()

    def test_one_layer_is_canonical_two_interferometer_form(self):
        # Canonical Killoran layer = leading interferometer + the per-patch
        # gate sequence (which itself ends with the second interferometer + D + K).
        assert _build_cvqnn_plan(1) == _INTERFEROMETER_SEQUENCE + _GATE_SEQUENCE
        assert _build_cvqnn_plan(1) == _CVQNN_LAYER

    def test_multi_layer_stacks(self):
        assert _build_cvqnn_plan(3) == _CVQNN_LAYER * 3

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            _build_cvqnn_plan(-1)


# ---------------------------------------------------------------------------
# state_dict parity (checkpoint-compat invariant)
# ---------------------------------------------------------------------------


class TestStateDictParity:
    def test_disabled_registers_no_cvqnn_params(self, tiny_data_config):
        model = build_model(
            ExperimentConfig(model="quantum", data=tiny_data_config,
                             quantum=_config(0))
        )
        assert all("cvqnn_params" not in k for k in model.state_dict())
        # Heads expose cvqnn_params as a plain None attribute (not a Parameter).
        for head in model.cv_attention.heads:
            assert head.cvqnn_params is None

    def test_enabled_adds_exactly_one_param_per_head(self, tiny_data_config):
        model0 = build_model(
            ExperimentConfig(model="quantum", data=tiny_data_config,
                             quantum=_config(0))
        )
        model1 = build_model(
            ExperimentConfig(model="quantum", data=tiny_data_config,
                             quantum=_config(1))
        )
        added = set(model1.state_dict()) - set(model0.state_dict())
        assert added == {
            f"cv_attention.heads.{h}.cvqnn_params"
            for h in range(model1.cv_attention.num_heads)
        }

    def test_disabled_state_dict_loads_into_fresh_disabled_model(self, tiny_data_config):
        cfg = ExperimentConfig(model="quantum", data=tiny_data_config,
                               quantum=_config(0))
        torch.manual_seed(0)
        m_a = build_model(cfg)
        torch.manual_seed(1)
        m_b = build_model(cfg)
        # strict load must succeed (identical key set, no W params).
        m_b.load_state_dict(m_a.state_dict(), strict=True)

    def test_shared_model_also_gets_w_per_head(self, tiny_data_config):
        model = build_model(
            ExperimentConfig(model="quantum_shared", data=tiny_data_config,
                             quantum=_config(1))
        )
        keys = [k for k in model.state_dict() if "cvqnn_params" in k]
        assert len(keys) == model.cv_attention.num_heads


# ---------------------------------------------------------------------------
# Gradient finiteness (the NaN-singularity guard) + non-triviality
# ---------------------------------------------------------------------------


class TestCvqnnGradientsAndEffect:
    @pytest.mark.parametrize("lw", [1, 2])
    def test_small_random_init_gives_finite_grads(self, tiny_data_config, lw):
        torch.manual_seed(0)
        model = build_model(
            ExperimentConfig(model="quantum", data=tiny_data_config,
                             quantum=_config(lw))
        )
        patches = torch.rand(3, model.cv_attention.heads[0].lcu_coeffs.b_real.numel(),
                             tiny_data_config.patch_size ** 2)
        out = model(patches, return_trunc_loss=True)
        loss = out.logits.pow(2).sum() + out.cvqnn_trunc_loss
        loss.backward()
        for head in model.cv_attention.heads:
            g = head.cvqnn_params.grad
            assert g is not None
            assert torch.isfinite(g).all(), "W gradients must be finite (no NaN)"

    def test_w_is_non_trivial(self, tiny_data_config):
        # Same seed → identical LCU/poly/hypernet; only the W block differs.
        torch.manual_seed(0)
        m0 = build_model(ExperimentConfig(model="quantum", data=tiny_data_config,
                                          quantum=_config(0)))
        torch.manual_seed(0)
        m1 = build_model(ExperimentConfig(model="quantum", data=tiny_data_config,
                                          quantum=_config(1)))
        patches = torch.rand(2, 4, tiny_data_config.patch_size ** 2)
        r0 = m0(patches, return_readouts=True, return_trunc_loss=True).readouts
        r1 = m1(patches, return_readouts=True, return_trunc_loss=True).readouts
        assert (r0 - r1).abs().max() > 1e-4, "W (W≠I) must change the readouts"


# ---------------------------------------------------------------------------
# cvqnn_trunc_loss plumbing
# ---------------------------------------------------------------------------


class TestCvqnnTruncLoss:
    def test_zero_when_disabled(self, tiny_data_config):
        model = build_model(ExperimentConfig(model="quantum", data=tiny_data_config,
                                             quantum=_config(0)))
        patches = torch.rand(2, 4, tiny_data_config.patch_size ** 2)
        out = model(patches, return_trunc_loss=True)
        assert out.cvqnn_trunc_loss is not None
        assert float(out.cvqnn_trunc_loss) == 0.0

    def test_nonnegative_and_present_when_enabled(self, tiny_data_config):
        torch.manual_seed(0)
        model = build_model(ExperimentConfig(model="quantum", data=tiny_data_config,
                                             quantum=_config(1)))
        patches = torch.rand(2, 4, tiny_data_config.patch_size ** 2)
        out = model(patches, return_trunc_loss=True)
        assert out.cvqnn_trunc_loss is not None
        assert float(out.cvqnn_trunc_loss.detach()) >= -1e-6  # ≥0 up to float noise

    def test_none_when_not_requested(self, tiny_data_config):
        model = build_model(ExperimentConfig(model="quantum", data=tiny_data_config,
                                             quantum=_config(1)))
        patches = torch.rand(2, 4, tiny_data_config.patch_size ** 2)
        logits = model(patches)  # no return_* flags → bare tensor
        assert isinstance(logits, torch.Tensor)


# ---------------------------------------------------------------------------
# Migration script + loud guard round-trip
# ---------------------------------------------------------------------------


class TestMigrationAndGuard:
    def _old_style_config_dict(self, data_config) -> dict:
        cfg = ExperimentConfig(model="quantum", data=data_config, quantum=_config(0))
        raw = asdict(cfg)
        raw["quantum"].pop("cvqnn_num_layers")
        raw["quantum"].pop("cvqnn_trunc_lambda")
        return raw

    def test_guard_raises_on_unmigrated_config(self, tiny_data_config):
        raw = self._old_style_config_dict(tiny_data_config)
        with pytest.raises(ValueError, match="migrate_add_cvqnn_field"):
            experiment_config_from_dict(raw)

    def test_guard_passes_on_current_config(self, tiny_data_config):
        cfg = ExperimentConfig(model="quantum", data=tiny_data_config, quantum=_config(1))
        reconstructed = experiment_config_from_dict(asdict(cfg))
        assert reconstructed.quantum.cvqnn_num_layers == 1

    def test_migration_injects_prew_values_and_is_idempotent(self, tiny_data_config):
        from experiments.migrate_add_cvqnn_field import migrate_config

        raw = self._old_style_config_dict(tiny_data_config)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "config.json"
            json.dump(raw, open(p, "w"))
            assert migrate_config(p, dry_run=False) == "migrated"
            assert migrate_config(p, dry_run=False) == "skipped"  # idempotent
            migrated = json.load(open(p))
            assert migrated["quantum"]["cvqnn_num_layers"] == 0
            assert migrated["quantum"]["cvqnn_trunc_lambda"] == 0.0
            # Guard now passes and rebuilds as a pre-W model.
            cfg = experiment_config_from_dict(migrated)
            assert cfg.quantum.cvqnn_num_layers == 0

    def test_migrated_config_loads_prew_checkpoint_strict(self, tiny_data_config):
        # Build a pre-W model, save its state_dict, strip the cvqnn_* keys to
        # simulate an old config, migrate, rebuild, and strict-load. This is the
        # end-to-end checkpoint-compat guarantee.
        from experiments.migrate_add_cvqnn_field import migrate_config

        cfg0 = ExperimentConfig(model="quantum", data=tiny_data_config, quantum=_config(0))
        torch.manual_seed(0)
        m0 = build_model(cfg0)
        state = m0.state_dict()

        raw = asdict(cfg0)
        raw["quantum"].pop("cvqnn_num_layers")
        raw["quantum"].pop("cvqnn_trunc_lambda")
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "config.json"
            json.dump(raw, open(p, "w"))
            migrate_config(p, dry_run=False)
            cfg_re = experiment_config_from_dict(json.load(open(p)))
            torch.manual_seed(1)
            m_re = build_model(cfg_re)
            m_re.load_state_dict(state, strict=True)  # must not raise

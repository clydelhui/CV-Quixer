"""Tests for the seq-to-seq stacked CV-Quixer (model="quantum_stacked", ADR-0003).

A stacked model runs num_seq2seq_blocks uniform seq-to-seq blocks (position i's
output token is the readout of W·P(M)·U_{q,i}|0⟩), ending in mean-pooling or a
canonical seq-to-one aggregator block (pooling="quixer"), then the shared
CVDecoder. These tests exercise the public interface only: QuantumConfig,
build_model, and the BaseVisionTransformer forward contract.

Small circuits (num_modes=2, cutoff_dim=4, 4 patches) keep simulation tractable.
"""

import pytest
import torch

from cv_quixer.config.schema import ExperimentConfig, QuantumConfig
from cv_quixer.models import build_model


def _stacked_config(**overrides) -> QuantumConfig:
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
    )
    base.update(overrides)
    return QuantumConfig(**base)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestStackedConfigValidation:
    def test_defaults_construct_and_hold_documented_values(self):
        cfg = _stacked_config()
        assert cfg.num_seq2seq_blocks == 1
        assert cfg.pooling == "mean"
        assert cfg.block_residual is True
        assert cfg.query_trunc_lambda == 0.01

    def test_zero_blocks_rejected_at_construction(self):
        # An aggregator-on-raw-patches model is exactly the existing CVQuixer;
        # the stacked config never describes it (ADR-0003).
        with pytest.raises(ValueError, match="num_seq2seq_blocks"):
            _stacked_config(num_seq2seq_blocks=0)

    def test_unknown_pooling_rejected_at_construction(self):
        with pytest.raises(ValueError, match="pooling"):
            _stacked_config(pooling="banana")

    def test_negative_query_trunc_lambda_rejected_at_construction(self):
        with pytest.raises(ValueError, match="query_trunc_lambda"):
            _stacked_config(query_trunc_lambda=-1.0)


# ---------------------------------------------------------------------------
# Forward contract
# ---------------------------------------------------------------------------


def _build_stacked(data_config, **overrides):
    return build_model(
        ExperimentConfig(
            model="quantum_stacked",
            data=data_config,
            quantum=_stacked_config(**overrides),
        )
    )


class TestStackedForward:
    def test_forward_produces_finite_logits(self, tiny_data_config):
        torch.manual_seed(0)
        model = _build_stacked(tiny_data_config)
        patches = torch.rand(2, 4, tiny_data_config.patch_size ** 2)
        logits = model(patches)
        assert logits.shape == (2, tiny_data_config.num_classes)
        assert torch.isfinite(logits).all()

    def test_backward_gives_finite_grads_on_every_parameter(self, tiny_data_config):
        torch.manual_seed(0)
        model = _build_stacked(tiny_data_config)
        patches = torch.rand(2, 4, tiny_data_config.patch_size ** 2)
        model(patches).pow(2).sum().backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, f"{name} got no gradient"
            assert torch.isfinite(p.grad).all(), f"{name} gradient not finite"

    @pytest.mark.parametrize("pooling", ["mean", "quixer"])
    def test_evaluate_contract_call_supported(self, tiny_data_config, pooling):
        # cv_quixer.evaluation.diagnostics.evaluate() calls every model with
        # exactly these flags — the stacked model must accept them and surface
        # the decoder-input stage's per-head success probs.
        torch.manual_seed(0)
        model = _build_stacked(tiny_data_config, pooling=pooling)
        patches = torch.rand(2, 4, tiny_data_config.patch_size ** 2)
        out = model(patches, return_trunc_loss=True, return_readouts=True,
                    return_success_prob=True)
        assert out.success_probs is not None
        assert len(out.success_probs) == model.config.num_heads


# ---------------------------------------------------------------------------
# Truncation streams
# ---------------------------------------------------------------------------


class TestTruncationStreams:
    def test_stacked_model_returns_all_three_streams(self, tiny_data_config):
        torch.manual_seed(0)
        model = _build_stacked(tiny_data_config)
        patches = torch.rand(2, 4, tiny_data_config.patch_size ** 2)
        out = model(patches, return_trunc_loss=True)
        for stream in ("trunc_loss", "cvqnn_trunc_loss", "query_trunc_loss"):
            val = getattr(out, stream)
            assert val is not None, f"{stream} missing"
            assert float(val.detach()) >= -1e-6, f"{stream} negative"

    def test_canonical_model_has_no_query_stream(self, tiny_data_config):
        torch.manual_seed(0)
        model = build_model(
            ExperimentConfig(
                model="quantum", data=tiny_data_config, quantum=_stacked_config()
            )
        )
        patches = torch.rand(2, 4, tiny_data_config.patch_size ** 2)
        out = model(patches, return_trunc_loss=True)
        assert out.query_trunc_loss is None
        assert out.trunc_loss is not None  # canonical streams unaffected


# ---------------------------------------------------------------------------
# Stacking
# ---------------------------------------------------------------------------


class TestStacking:
    def test_two_blocks_run_and_add_parameters(self, tiny_data_config):
        torch.manual_seed(0)
        m1 = _build_stacked(tiny_data_config)
        torch.manual_seed(0)
        m2 = _build_stacked(tiny_data_config, num_seq2seq_blocks=2)
        patches = torch.rand(2, 4, tiny_data_config.patch_size ** 2)
        logits = m2(patches)
        assert logits.shape == (2, tiny_data_config.num_classes)
        assert torch.isfinite(logits).all()
        assert m2.get_num_parameters() > m1.get_num_parameters()

    def test_residual_toggle_changes_outputs(self, tiny_data_config):
        # Same seed → identical parameters; only the inter-block wiring differs.
        torch.manual_seed(0)
        m_res = _build_stacked(tiny_data_config, num_seq2seq_blocks=2)
        torch.manual_seed(0)
        m_pure = _build_stacked(
            tiny_data_config, num_seq2seq_blocks=2, block_residual=False
        )
        patches = torch.rand(2, 4, tiny_data_config.patch_size ** 2)
        diff = (m_res(patches) - m_pure(patches)).abs().max()
        assert torch.isfinite(m_pure(patches)).all()
        assert diff > 1e-6, "residual on/off must change the forward pass"


# ---------------------------------------------------------------------------
# Aggregator block (pooling="quixer")
# ---------------------------------------------------------------------------


class TestAggregator:
    def test_quixer_pooling_runs_and_adds_parameters(self, tiny_data_config):
        torch.manual_seed(0)
        m_mean = _build_stacked(tiny_data_config)
        torch.manual_seed(0)
        m_agg = _build_stacked(tiny_data_config, pooling="quixer")
        patches = torch.rand(2, 4, tiny_data_config.patch_size ** 2)
        logits = m_agg(patches)
        assert logits.shape == (2, tiny_data_config.num_classes)
        assert torch.isfinite(logits).all()
        assert m_agg.get_num_parameters() > m_mean.get_num_parameters()

    def test_decoder_input_width_is_heads_times_r_under_both_poolings(
        self, tiny_data_config
    ):
        # The shared-decoder-width invariant (ADR-0003): the pre-decoder
        # readout vector is (B, H×R) regardless of pooling — identical to the
        # canonical model's decoder input.
        patches = torch.rand(2, 4, tiny_data_config.patch_size ** 2)
        for pooling in ("mean", "quixer"):
            torch.manual_seed(0)
            model = _build_stacked(tiny_data_config, pooling=pooling)
            out = model(patches, return_readouts=True)
            H = model.config.num_heads
            R = len(model.config._observable_plan)
            assert out.readouts.shape == (2, H * R), pooling


# ---------------------------------------------------------------------------
# Canonical models unaffected (checkpoint compatibility)
# ---------------------------------------------------------------------------


class TestCanonicalModelsUnaffected:
    @pytest.mark.parametrize("model_name", ["quantum", "quantum_shared"])
    def test_new_fields_are_inert_for_canonical_models(
        self, tiny_data_config, model_name
    ):
        # Varying every ADR-0003 field must not change a canonical model's
        # state-dict key set — pre-stacked checkpoints still strict-load.
        torch.manual_seed(0)
        m_default = build_model(
            ExperimentConfig(
                model=model_name, data=tiny_data_config, quantum=_stacked_config()
            )
        )
        torch.manual_seed(1)
        m_varied = build_model(
            ExperimentConfig(
                model=model_name,
                data=tiny_data_config,
                quantum=_stacked_config(
                    num_seq2seq_blocks=3,
                    pooling="quixer",
                    block_residual=False,
                    query_trunc_lambda=0.5,
                ),
            )
        )
        assert set(m_default.state_dict()) == set(m_varied.state_dict())
        m_varied.load_state_dict(m_default.state_dict(), strict=True)

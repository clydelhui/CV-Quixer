"""Tests for the coefficient-ablation knob (--coeff-ablation).

A CV-Quixer head learns two sets of combination coefficients: the per-position
LCU coefficients b_i (M = Σ_i b_i U_i) and the per-degree polynomial
coefficients c_j (P(M) = Σ_j c_j M^j). This knob freezes them to fixed uniform
values to ablate the learned weighting structure (CONTEXT.md "Coefficient
ablation", ADR-0008). Three cumulative levels:

  * ``"none"``     — the default; b_i and c_j trained as normal (byte-identical
    to a pre-knob model);
  * ``"lcu"``      — freeze b_i = 1/N (requires_grad=False), removing per-position
    weighting; c_j still trained;
  * ``"lcu_poly"`` — additionally freeze c_j = 1 (all-ones, NOT the [1,0,..]
    init, which would make P(M)=I and discard the LCU).

The frozen coefficients are gauge-/renorm-inert as trainable scalars (ADR-0008),
so they are frozen rather than tied to one trainable parameter. They live in the
shared head base (_CVHeadBase), so one change point covers quantum /
quantum_shared / quantum_stacked.

Small circuits (num_modes=2, cutoff_dim=4) keep simulation tractable.
"""

import sys
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

# _run_selection / sweep / report_sweep live in experiments/ (not a package).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

from cv_quixer.config.schema import DataConfig, ExperimentConfig, QuantumConfig
from cv_quixer.config.utils import experiment_config_from_dict
from cv_quixer.models.quantum.cv_attention import (
    LCUSumCoefficients,
    PolynomialCoefficients,
)


class TestConfigField:
    def test_default_is_none(self):
        assert QuantumConfig().coeff_ablation == "none"

    def test_invalid_value_is_rejected(self):
        with pytest.raises(ValueError, match="coeff_ablation"):
            QuantumConfig(coeff_ablation="lcu_only")

    @pytest.mark.parametrize("value", ["none", "lcu", "lcu_poly"])
    def test_valid_values_accepted(self, value):
        assert QuantumConfig(coeff_ablation=value).coeff_ablation == value

    def test_old_config_without_key_loads_as_none(self):
        # Pre-knob runs have no coeff_ablation key. Like positional_encoding's
        # default path, an absent key reloads silently as the historic behaviour
        # ("none") — byte-identical, so no migration is needed.
        cfg = ExperimentConfig(
            model="quantum",
            data=DataConfig(image_size=14, patch_size=7, num_classes=10),
            quantum=QuantumConfig(num_modes=2, cutoff_dim=4, num_heads=2),
        )
        raw = asdict(cfg)
        raw["quantum"].pop("coeff_ablation")
        reconstructed = experiment_config_from_dict(raw)  # must not raise
        assert reconstructed.quantum.coeff_ablation == "none"


class TestLCUFreeze:
    def test_trainable_default_is_uniform_and_grad(self):
        lcu = LCUSumCoefficients(4)
        assert lcu.b_real.requires_grad and lcu.b_imag.requires_grad
        assert torch.allclose(lcu.b_real, torch.full((4,), 0.25))
        assert torch.allclose(lcu.b_imag, torch.zeros(4))

    def test_frozen_is_uniform_and_no_grad(self):
        lcu = LCUSumCoefficients(4, trainable=False)
        assert not lcu.b_real.requires_grad and not lcu.b_imag.requires_grad
        # Frozen LCU is the existing 1/N init with grad off.
        assert torch.allclose(lcu.b_real, torch.full((4,), 0.25))
        assert torch.allclose(lcu.b_imag, torch.zeros(4))
        # forward() still yields the complex coefficient tensor.
        b = lcu()
        assert torch.allclose(b.real, torch.full((4,), 0.25))
        assert torch.allclose(b.imag, torch.zeros(4))


class TestPolyFreeze:
    def test_trainable_default_is_identity_init_and_grad(self):
        poly = PolynomialCoefficients(3)
        assert poly.c.requires_grad
        assert torch.allclose(poly.c, torch.tensor([1.0, 0.0, 0.0, 0.0]))

    def test_frozen_is_all_ones_and_no_grad(self):
        poly = PolynomialCoefficients(3, trainable=False)
        assert not poly.c.requires_grad
        # All-ones, NOT the [1,0,..] init (which would make P(M)=I).
        assert torch.allclose(poly.c, torch.ones(4))

    def test_frozen_ignores_poly_init_noise(self):
        # A frozen polynomial is deterministic all-ones regardless of noise.
        poly = PolynomialCoefficients(3, poly_init_noise=0.5, trainable=False)
        assert torch.allclose(poly.c, torch.ones(4))


def _quantum_config(coeff_ablation: str = "none", **overrides) -> QuantumConfig:
    base = dict(
        num_modes=2, num_layers=1, cutoff_dim=4, num_heads=2,
        cnn_channels_1=4, cnn_channels_2=8, cnn_kernel_size=3,
        decoder_hidden_dim=16, poly_degree=2, dtype="complex64",
        cvqnn_num_layers=0, coeff_ablation=coeff_ablation,
    )
    base.update(overrides)
    return QuantumConfig(**base)


def _build(tiny_data_config, coeff_ablation, model="quantum"):
    from cv_quixer.models import build_model
    return build_model(ExperimentConfig(
        model=model, data=tiny_data_config,
        quantum=_quantum_config(coeff_ablation),
    ))


def _grad_flags(model):
    """requires_grad for every lcu/poly coefficient param, keyed by suffix."""
    lcu = [p.requires_grad for n, p in model.named_parameters()
           if n.endswith("lcu_coeffs.b_real") or n.endswith("lcu_coeffs.b_imag")]
    poly = [p.requires_grad for n, p in model.named_parameters()
            if n.endswith("poly_coeffs.c")]
    return lcu, poly


class TestModelThreading:
    @pytest.mark.parametrize("model", ["quantum", "quantum_shared", "quantum_stacked"])
    def test_none_leaves_all_trainable(self, tiny_data_config, model):
        lcu, poly = _grad_flags(_build(tiny_data_config, "none", model))
        assert lcu and poly, "every variant must expose lcu + poly coeff params"
        assert all(lcu) and all(poly), "'none' must keep all coeffs trainable"

    @pytest.mark.parametrize("model", ["quantum", "quantum_shared", "quantum_stacked"])
    def test_lcu_freezes_only_lcu(self, tiny_data_config, model):
        lcu, poly = _grad_flags(_build(tiny_data_config, "lcu", model))
        assert lcu and not any(lcu), "'lcu' must freeze every LCU coeff"
        assert poly and all(poly), "'lcu' must leave polynomial coeffs trainable"

    @pytest.mark.parametrize("model", ["quantum", "quantum_shared", "quantum_stacked"])
    def test_lcu_poly_freezes_both(self, tiny_data_config, model):
        lcu, poly = _grad_flags(_build(tiny_data_config, "lcu_poly", model))
        assert lcu and not any(lcu), "'lcu_poly' must freeze every LCU coeff"
        assert poly and not any(poly), "'lcu_poly' must freeze every polynomial coeff"

    @pytest.mark.parametrize("ablation", ["none", "lcu", "lcu_poly"])
    def test_forward_runs_finite(self, tiny_data_config, ablation):
        torch.manual_seed(0)
        model = _build(tiny_data_config, ablation)
        n_patches = (14 // 7) ** 2
        x = torch.randn(2, n_patches, 7 * 7)
        out = model(x)
        assert out.shape == (2, 10)
        assert torch.isfinite(out).all()

    def test_grad_flows_through_frozen_coeffs_to_hypernet(self, tiny_data_config):
        # ADR-0008: freezing b_i / c_j must NOT block the gradient that reaches the
        # trainable per-patch unitaries U_i (the hypernet) THROUGH the LCU.
        torch.manual_seed(0)
        model = _build(tiny_data_config, "lcu_poly")
        n_patches = (14 // 7) ** 2
        x = torch.randn(2, n_patches, 7 * 7)
        model(x).sum().backward()
        hypernet_grads = [
            p.grad for n, p in model.named_parameters()
            if "hypernet" in n.lower() and p.requires_grad
        ]
        assert hypernet_grads, "expected trainable hypernet params"
        assert any(g is not None and g.abs().sum() > 0 for g in hypernet_grads), (
            "gradient must reach the hypernet through the frozen coefficients"
        )


class TestParamCountAndCompat:
    def test_param_count_drops_by_frozen_coeffs(self, tiny_data_config):
        from cv_quixer.utils.params import count_parameters
        none = _build(tiny_data_config, "none")
        lcu = _build(tiny_data_config, "lcu")
        lcu_poly = _build(tiny_data_config, "lcu_poly")
        H = none.config.num_heads
        N = (14 // 7) ** 2
        d1 = none.config.poly_degree + 1
        c_none = count_parameters(none)
        c_lcu = count_parameters(lcu)
        c_lcu_poly = count_parameters(lcu_poly)
        # 'lcu' frees 2N reals per head (b_real + b_imag).
        assert c_none - c_lcu == H * 2 * N
        # 'lcu_poly' additionally frees (d+1) reals per head (c).
        assert c_lcu - c_lcu_poly == H * d1

    def test_none_coeffs_match_historic_init(self, tiny_data_config):
        # The "none" path must reproduce the pre-knob init exactly (b_i=1/N,
        # c=[1,0,..]) so existing checkpoints reload byte-identically.
        model = _build(tiny_data_config, "none")
        N = (14 // 7) ** 2
        for name, p in model.named_parameters():
            if name.endswith("lcu_coeffs.b_real"):
                assert torch.allclose(p, torch.full((N,), 1.0 / N))
            if name.endswith("poly_coeffs.c"):
                expected = torch.zeros_like(p)
                expected[0] = 1.0
                assert torch.allclose(p, expected)


class TestCoordinateRegistration:
    def test_filterable_field_registered(self):
        import _run_selection as rs

        field = rs._FIELDS_BY_NAME.get("coeff_ablation")
        assert field is not None, "coeff_ablation must be a filterable coordinate"
        assert field.py_type is str
        assert field.config_path == ("quantum", "coeff_ablation")

    def test_report_treats_it_as_config_identity(self):
        import report_sweep

        assert "coeff_ablation" in report_sweep.CONFIG_IDENTITY_FIELDS, (
            "none / lcu / lcu_poly runs of the same architecture must NOT be "
            "seed-averaged together — coeff_ablation has to be config-identity"
        )


class TestSweepAxis:
    def _manifest(self, *argv):
        import sweep
        return sweep.build_manifest(sweep.build_parser().parse_args(list(argv)))

    def test_coeff_ablation_is_a_grid_axis(self):
        manifest = self._manifest(
            "--observables", "xpxsps", "--coeff-ablation", "none", "lcu", "lcu_poly",
        )
        assert manifest["n_runs"] == 3, "one run per coeff_ablation variant"
        assert manifest["axes"]["coeff_ablation"] == ["none", "lcu", "lcu_poly"]

    def test_run_name_markers_and_forwarded_flag(self):
        manifest = self._manifest(
            "--observables", "xpxsps", "--coeff-ablation", "lcu", "lcu_poly",
        )
        markers = {}
        for run in manifest["runs"]:
            ca = run["coeff_ablation"]
            markers[ca] = run["run_name"]
            args = run["args"]
            assert args[args.index("--coeff-ablation") + 1] == ca
        assert "__calcu" in markers["lcu"]
        assert "__calcu_poly" in markers["lcu_poly"]

    def test_absent_axis_emits_no_flag_or_marker(self):
        manifest = self._manifest("--observables", "xpxsps", "--num-heads", "4")
        (run,) = manifest["runs"]
        assert "__ca" not in run["run_name"]
        assert "--coeff-ablation" not in run["args"]

    def test_standalone_coeff_ablation_sweep_is_allowed(self, tmp_path, monkeypatch):
        # coeff_ablation is a valid sole manual axis: main()'s under-specified
        # guard must accept it like any other axis.
        import sweep
        monkeypatch.setattr(sys, "argv", [
            "sweep.py", "--coeff-ablation", "lcu", "lcu_poly",
            "--observables", "xpxsps", "--sweeps-root", str(tmp_path), "--dry-run",
        ])
        sweep.main()  # must not SystemExit on the "specify an axis" guard

"""Tests for configurable observable readouts.

Covers:
    1. New observable matrices (x², p²) — analytic correctness.
    2. New CVCircuit measurement methods — vacuum and Fock-state checks.
    3. ObservableSpec config validation and expansion to evaluation plan.
    4. Backward compatibility: legacy `readout_observable` string ↔ new list.
    5. Decoder input dimension matches the expanded plan.
    6. Gradient flow through new observables to CNN hypernetwork params.
    7. vmap correctness — forward pass shapes for mixed observable plans.
    8. Checkpoint round-trip equivalence (load legacy-trained state into a
       model configured with the explicit equivalent observable list).
"""

import torch
import pytest

import dacite

from cv_quixer.config.schema import (
    ObservableSpec,
    QuantumConfig,
)
from cv_quixer.models.quantum.cv_quixer import CVQuixer
from cv_quixer.quantum import CVCircuit, FockState
from cv_quixer.quantum.ops import (
    quadrature_p_matrix,
    quadrature_p_squared_matrix,
    quadrature_x_matrix,
    quadrature_x_squared_matrix,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


# data_config and tiny_data_config are provided by tests/conftest.py (session scope).


def _small_config(**overrides) -> QuantumConfig:
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
    )
    base.update(overrides)
    return QuantumConfig(**base)


def _fock_state(occupations: list[int], cutoff_dim: int) -> FockState:
    """Build a pure Fock state |n_0, n_1, …⟩."""
    n = len(occupations)
    shape = (cutoff_dim,) * n
    data = torch.zeros(shape, dtype=torch.complex128)
    data[tuple(occupations)] = 1.0
    return FockState(data, n, cutoff_dim)


# ---------------------------------------------------------------------------
# 1. Observable matrices
# ---------------------------------------------------------------------------


class TestObservableMatrices:
    @pytest.mark.parametrize("D", [4, 8])
    def test_x_squared_diagonal_equals_n_plus_half(self, D):
        # ⟨n|x̂²|n⟩ = n + 1/2 for the harmonic oscillator
        xx = quadrature_x_squared_matrix(D).diagonal().real
        expected = torch.arange(D, dtype=torch.float64) + 0.5
        # The truncation at the boundary loses some weight, but for n < D-1
        # the analytic identity holds within numerical precision.
        assert torch.allclose(xx[:-1], expected[:-1], atol=1e-10)

    @pytest.mark.parametrize("D", [4, 8])
    def test_p_squared_diagonal_equals_n_plus_half(self, D):
        pp = quadrature_p_squared_matrix(D).diagonal().real
        expected = torch.arange(D, dtype=torch.float64) + 0.5
        assert torch.allclose(pp[:-1], expected[:-1], atol=1e-10)

    def test_x_squared_matches_x_at_x(self):
        D = 8
        x = quadrature_x_matrix(D)
        xx = quadrature_x_squared_matrix(D)
        assert torch.allclose(xx, x @ x, atol=1e-12)

    def test_p_squared_matches_p_at_p(self):
        D = 8
        p = quadrature_p_matrix(D)
        pp = quadrature_p_squared_matrix(D)
        assert torch.allclose(pp, p @ p, atol=1e-12)

    def test_x_squared_is_hermitian(self):
        xx = quadrature_x_squared_matrix(6)
        assert torch.allclose(xx, xx.conj().T, atol=1e-12)

    def test_p_squared_is_hermitian(self):
        pp = quadrature_p_squared_matrix(6)
        assert torch.allclose(pp, pp.conj().T, atol=1e-12)


# ---------------------------------------------------------------------------
# 2. Circuit measurement methods
# ---------------------------------------------------------------------------


class TestCircuitMeasurements:
    def test_prob_n_photons_vacuum_concentrated_at_zero(self):
        circuit = CVCircuit(num_modes=2, cutoff_dim=4)
        state = FockState.vacuum(num_modes=2, cutoff_dim=4)
        assert circuit.measure_prob_n_photons(0, 0, state).item() == pytest.approx(1.0, abs=1e-10)
        for n in range(1, 4):
            assert circuit.measure_prob_n_photons(0, n, state).item() == pytest.approx(0.0, abs=1e-10)
            assert circuit.measure_prob_n_photons(1, n, state).item() == pytest.approx(0.0, abs=1e-10)

    def test_prob_n_photons_fock_state(self):
        circuit = CVCircuit(num_modes=2, cutoff_dim=4)
        # |2, 1⟩ — 2 photons in mode 0, 1 photon in mode 1
        state = _fock_state([2, 1], cutoff_dim=4)
        assert circuit.measure_prob_n_photons(0, 2, state).item() == pytest.approx(1.0, abs=1e-10)
        assert circuit.measure_prob_n_photons(0, 0, state).item() == pytest.approx(0.0, abs=1e-10)
        assert circuit.measure_prob_n_photons(1, 1, state).item() == pytest.approx(1.0, abs=1e-10)
        assert circuit.measure_prob_n_photons(1, 0, state).item() == pytest.approx(0.0, abs=1e-10)

    def test_x_squared_on_fock_state(self):
        # On |n⟩, ⟨x̂²⟩ = n + 1/2
        circuit = CVCircuit(num_modes=1, cutoff_dim=8)
        for n in range(0, 6):  # stay away from cutoff
            state = _fock_state([n], cutoff_dim=8)
            measured = circuit.measure_quadrature_x_squared(0, state).item()
            assert measured == pytest.approx(n + 0.5, abs=1e-10)

    def test_p_squared_on_fock_state(self):
        circuit = CVCircuit(num_modes=1, cutoff_dim=8)
        for n in range(0, 6):
            state = _fock_state([n], cutoff_dim=8)
            measured = circuit.measure_quadrature_p_squared(0, state).item()
            assert measured == pytest.approx(n + 0.5, abs=1e-10)

    def test_x_squared_consistent_with_manual_trace(self):
        circuit = CVCircuit(num_modes=2, cutoff_dim=5)
        torch.manual_seed(0)
        raw = torch.randn(5, 5, dtype=torch.complex128)
        raw = raw / raw.norm()
        state = FockState(raw, 2, 5)
        x = quadrature_x_matrix(5, dtype=torch.complex128)
        rho = state.reduced_density_matrix(0)
        manual = torch.trace(rho @ x @ x).real.item()
        measured = circuit.measure_quadrature_x_squared(0, state).item()
        assert measured == pytest.approx(manual, abs=1e-10)


# ---------------------------------------------------------------------------
# 3. Config validation + expansion
# ---------------------------------------------------------------------------


class TestObservableSpecValidation:
    def test_default_is_x_per_mode(self):
        cfg = _small_config()
        assert cfg.readout_observable is None
        assert cfg.readout_observables is None
        assert len(cfg._observable_plan) == cfg.num_modes
        assert all(e.type == "x" and e.n is None for e in cfg._observable_plan)
        assert [e.mode for e in cfg._observable_plan] == list(range(cfg.num_modes))

    def test_explicit_x_all_matches_default(self):
        a = _small_config()
        b = _small_config(readout_observables=[ObservableSpec(type="x", mode="all")])
        assert a._observable_plan == b._observable_plan

    def test_both_legacy_and_new_set_raises(self):
        with pytest.raises(ValueError, match="not both"):
            _small_config(
                readout_observable="quadrature_x",
                readout_observables=[ObservableSpec(type="x", mode="all")],
            )

    def test_invalid_legacy_string_raises(self):
        with pytest.raises(ValueError, match="readout_observable"):
            _small_config(readout_observable="not_a_real_observable")

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError, match="Unknown observable type"):
            _small_config(readout_observables=[ObservableSpec(type="bogus")])

    def test_mode_out_of_range_raises(self):
        with pytest.raises(ValueError, match="out of range"):
            _small_config(readout_observables=[ObservableSpec(type="x", mode=99)])

    def test_mode_string_other_than_all_raises(self):
        with pytest.raises(ValueError, match="must be 'all'"):
            _small_config(readout_observables=[ObservableSpec(type="x", mode="bogus")])

    def test_prob_n_without_n_raises(self):
        with pytest.raises(ValueError, match="requires 'n'"):
            _small_config(readout_observables=[ObservableSpec(type="prob_n", mode=0)])

    def test_prob_n_n_out_of_range_raises(self):
        with pytest.raises(ValueError, match="n=99 out of range"):
            _small_config(
                readout_observables=[ObservableSpec(type="prob_n", mode=0, n=99)],
            )

    def test_n_on_non_prob_n_raises(self):
        with pytest.raises(ValueError, match="must not specify 'n'"):
            _small_config(readout_observables=[ObservableSpec(type="x", mode=0, n=3)])

    def test_all_modes_expansion(self):
        cfg = _small_config(
            readout_observables=[ObservableSpec(type="n", mode="all")],
        )
        assert [e.mode for e in cfg._observable_plan] == list(range(cfg.num_modes))
        assert all(e.type == "n" for e in cfg._observable_plan)

    def test_list_modes_and_n_lexicographic_order(self):
        cfg = _small_config(
            num_modes=3,
            cutoff_dim=4,
            readout_observables=[
                ObservableSpec(type="prob_n", mode=[0, 2], n=[1, 3]),
            ],
        )
        expected = [
            ("prob_n", 0, 1),
            ("prob_n", 0, 3),
            ("prob_n", 2, 1),
            ("prob_n", 2, 3),
        ]
        actual = [(e.type, e.mode, e.n) for e in cfg._observable_plan]
        assert actual == expected

    def test_mixed_plan_total_length(self):
        cfg = _small_config(
            num_modes=2,
            cutoff_dim=4,
            readout_observables=[
                ObservableSpec(type="x", mode="all"),         # 2
                ObservableSpec(type="x_squared", mode=[0, 1]),  # 2
                ObservableSpec(type="prob_n", mode="all", n=[0, 1]),  # 4
                ObservableSpec(type="n", mode=0),             # 1
            ],
        )
        assert len(cfg._observable_plan) == 2 + 2 + 4 + 1


# ---------------------------------------------------------------------------
# 4. Legacy / backward compatibility
# ---------------------------------------------------------------------------


class TestLegacyCompatibility:
    def test_legacy_quadrature_x_equals_explicit_list(self):
        legacy = _small_config(readout_observable="quadrature_x")
        explicit = _small_config(
            readout_observables=[ObservableSpec(type="x", mode="all")]
        )
        assert legacy._observable_plan == explicit._observable_plan

    def test_legacy_photon_number_equals_explicit_list(self):
        legacy = _small_config(readout_observable="photon_number")
        explicit = _small_config(
            readout_observables=[ObservableSpec(type="n", mode="all")]
        )
        assert legacy._observable_plan == explicit._observable_plan

    def test_legacy_pnr_distribution_equals_explicit_list(self):
        legacy = _small_config(readout_observable="pnr_distribution")
        explicit = _small_config(
            readout_observables=[
                ObservableSpec(
                    type="prob_n", mode="all", n=list(range(legacy.cutoff_dim))
                ),
            ],
        )
        assert legacy._observable_plan == explicit._observable_plan

    def test_legacy_pnr_distribution_order_preserved(self):
        """Legacy pnr_distribution flattens (num_modes, cutoff_dim) by row.

        Iteration order must be (outer = mode, inner = n) so trained decoder
        weights stay aligned after the refactor.
        """
        cfg = _small_config(num_modes=2, cutoff_dim=3, readout_observable="pnr_distribution")
        expected = [
            ("prob_n", 0, 0),
            ("prob_n", 0, 1),
            ("prob_n", 0, 2),
            ("prob_n", 1, 0),
            ("prob_n", 1, 1),
            ("prob_n", 1, 2),
        ]
        actual = [(e.type, e.mode, e.n) for e in cfg._observable_plan]
        assert actual == expected


# ---------------------------------------------------------------------------
# 5. Decoder input dimension
# ---------------------------------------------------------------------------


class TestDecoderInputDim:
    def test_default_decoder_in_features(self, tiny_data_config):
        cfg = _small_config()
        model = CVQuixer(cfg, tiny_data_config)
        first_linear = model.decoder.net[0]
        assert first_linear.in_features == cfg.num_heads * cfg.num_modes

    @pytest.mark.parametrize(
        "specs, expected_per_head",
        [
            ([ObservableSpec(type="x", mode="all")], 2),                    # num_modes
            ([ObservableSpec(type="x_squared", mode="all")], 2),
            ([ObservableSpec(type="p_squared", mode=0)], 1),
            ([ObservableSpec(type="prob_n", mode="all", n=[0, 1])], 4),     # 2*2
            (
                [
                    ObservableSpec(type="x", mode="all"),
                    ObservableSpec(type="prob_n", mode=0, n=[0, 1, 2]),
                ],
                5,
            ),
        ],
    )
    def test_decoder_in_features_matches_plan(self, tiny_data_config, specs, expected_per_head):
        cfg = _small_config(readout_observables=specs)
        model = CVQuixer(cfg, tiny_data_config)
        first_linear = model.decoder.net[0]
        assert first_linear.in_features == cfg.num_heads * expected_per_head


# ---------------------------------------------------------------------------
# 6. Forward pass shape / no-NaN under mixed observable plans
# ---------------------------------------------------------------------------


class TestForwardShapeMixed:
    @pytest.mark.parametrize(
        "specs",
        [
            [ObservableSpec(type="x_squared", mode="all")],
            [ObservableSpec(type="p_squared", mode="all")],
            [ObservableSpec(type="prob_n", mode=0, n=2)],
            [
                ObservableSpec(type="x", mode="all"),
                ObservableSpec(type="x_squared", mode=[0, 1]),
                ObservableSpec(type="prob_n", mode="all", n=[0, 1]),
                ObservableSpec(type="n", mode=0),
            ],
        ],
    )
    def test_forward_shape_and_finite(self, tiny_data_config, specs):
        cfg = _small_config(readout_observables=specs)
        model = CVQuixer(cfg, tiny_data_config)
        patch_dim = tiny_data_config.patch_size ** 2
        num_patches = (tiny_data_config.image_size // tiny_data_config.patch_size) ** 2
        patches = torch.randn(2, num_patches, patch_dim)
        logits = model(patches)
        assert logits.shape == (2, tiny_data_config.num_classes)
        assert torch.isfinite(logits).all()


# ---------------------------------------------------------------------------
# 7. Gradient flow through new observables
# ---------------------------------------------------------------------------


class TestGradientFlow:
    @pytest.mark.parametrize(
        "specs",
        [
            [ObservableSpec(type="x_squared", mode="all")],
            [ObservableSpec(type="p_squared", mode="all")],
            [ObservableSpec(type="prob_n", mode="all", n=[1, 2])],
            [
                ObservableSpec(type="x_squared", mode="all"),
                ObservableSpec(type="prob_n", mode=0, n=2),
            ],
        ],
    )
    def test_gradient_reaches_hypernetwork(self, tiny_data_config, specs):
        """Backward pass must populate non-None grads on CNN hypernetwork params.

        Mirrors the existing PNR gradient test: the default polynomial init
        collapses to c_0|ψ_in⟩, which is CNN-independent. Set c_1=0.5 to
        activate the data-dependent branch.
        """
        cfg = _small_config(readout_observables=specs)
        model = CVQuixer(cfg, tiny_data_config)
        with torch.no_grad():
            for head in model.cv_attention.heads:
                head.poly_coeffs.c.data[1] = 0.5

        patch_dim = tiny_data_config.patch_size ** 2
        num_patches = (tiny_data_config.image_size // tiny_data_config.patch_size) ** 2
        patches = torch.randn(2, num_patches, patch_dim)
        logits = model(patches)
        logits.sum().backward()

        cnn_params = [
            p for n, p in model.named_parameters()
            if "hypernetwork" in n and p.requires_grad
        ]
        assert cnn_params, "No CNN hypernetwork parameters found"
        assert all(p.grad is not None for p in cnn_params), (
            "Some CNN hypernetwork params received no gradient"
        )
        assert any(p.grad.abs().sum().item() > 0 for p in cnn_params), (
            "Readout did not propagate gradient to the hypernetwork"
        )


# ---------------------------------------------------------------------------
# 8. Checkpoint round-trip equivalence (legacy ↔ explicit)
# ---------------------------------------------------------------------------


class TestCheckpointRoundTrip:
    @pytest.mark.parametrize(
        "legacy, explicit_specs",
        [
            ("quadrature_x", [ObservableSpec(type="x", mode="all")]),
            ("photon_number", [ObservableSpec(type="n", mode="all")]),
            (
                "pnr_distribution",
                [ObservableSpec(type="prob_n", mode="all", n=[0, 1, 2, 3])],
            ),
        ],
    )
    def test_legacy_state_dict_loads_into_explicit_model(
        self, tiny_data_config, legacy, explicit_specs
    ):
        cfg_legacy = _small_config(readout_observable=legacy, cutoff_dim=4)
        cfg_explicit = _small_config(readout_observables=explicit_specs, cutoff_dim=4)

        torch.manual_seed(0)
        model_legacy = CVQuixer(cfg_legacy, tiny_data_config)
        sd = model_legacy.state_dict()

        torch.manual_seed(0)
        model_explicit = CVQuixer(cfg_explicit, tiny_data_config)
        # Must load without missing or unexpected keys
        result = model_explicit.load_state_dict(sd, strict=True)
        # Strict load returns IncompatibleKeys(missing_keys=[], unexpected_keys=[])
        assert not result.missing_keys
        assert not result.unexpected_keys

        patch_dim = tiny_data_config.patch_size ** 2
        num_patches = (tiny_data_config.image_size // tiny_data_config.patch_size) ** 2
        patches = torch.randn(2, num_patches, patch_dim)

        with torch.no_grad():
            out_legacy = model_legacy(patches)
            out_explicit = model_explicit(patches)
        # fp32 quantum sim — generous tolerance for ordering-equivalent computations
        assert torch.allclose(out_legacy, out_explicit, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# 9. YAML / dacite parsing of the new field
# ---------------------------------------------------------------------------


class TestDaciteLoading:
    def test_dacite_parses_list_of_dicts(self):
        raw = {
            "num_modes": 2,
            "cutoff_dim": 4,
            "readout_observables": [
                {"type": "x", "mode": "all"},
                {"type": "x_squared", "mode": [0, 1]},
                {"type": "prob_n", "mode": "all", "n": [0, 1]},
                {"type": "n", "mode": 0},
            ],
        }
        cfg = dacite.from_dict(
            QuantumConfig, raw, config=dacite.Config(strict=False)
        )
        assert len(cfg._observable_plan) == 2 + 2 + 4 + 1

    def test_dacite_parses_legacy_string(self):
        raw = {"num_modes": 2, "cutoff_dim": 4, "readout_observable": "pnr_distribution"}
        cfg = dacite.from_dict(
            QuantumConfig, raw, config=dacite.Config(strict=False)
        )
        assert len(cfg._observable_plan) == 2 * 4

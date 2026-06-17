"""Tests for the polynomial-coefficient init perturbation (--poly-init-noise).

The polynomial coefficients are initialised to c = [1, 0, …, 0] so that
P(M) = I at the start of training. With c_{j≥1} = 0 the input-dependent operator
M = Σ b_i U_i is bypassed, the readout is input-independent at init, and the loss
is structurally pinned at ln(num_classes) — the *uniform-predictor collapse*
(CONTEXT.md). `poly_init_noise` seeds c_{j≥1} ≠ 0 to break that symmetry.

Load-bearing invariants pinned here:

  * poly_init_noise > 0 seeds c_{j≥1} with O(eps) noise (c_0 stays 1);
  * poly_init_noise == 0.0 (the default) is byte-identical to today — c stays
    exactly [1, 0, …] AND no RNG is consumed (a naive eps*randn draw would shift
    the seeded stream and break checkpoint parity);
  * the noise is seeded/reproducible, and per-head construction breaks head
    symmetry;
  * adding the knob changes no state_dict keys (still self.c);
  * poly_init_noise is a registered configuration coordinate.

Small circuits (num_modes=2, cutoff_dim=4) keep simulation tractable.
"""

import sys
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

# _run_selection / report_sweep live in experiments/ (not a package) — import via
# sys.path like the other experiment-script tests.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

from cv_quixer.config.schema import DataConfig, ExperimentConfig, QuantumConfig
from cv_quixer.config.utils import experiment_config_from_dict
from cv_quixer.models.quantum.cv_attention import PolynomialCoefficients


class TestPolynomialCoefficientsInit:
    def test_noise_seeds_higher_order_coeffs(self):
        torch.manual_seed(0)
        poly = PolynomialCoefficients(degree=3, poly_init_noise=0.05)
        c = poly().detach()
        assert c[0].item() == 1.0
        assert (c[1:].abs() > 0).all(), "c_{j>=1} must be seeded nonzero"
        # O(eps): well within a few standard deviations of eps=0.05.
        assert c[1:].abs().max().item() < 0.5

    def test_default_is_identity_init(self):
        poly = PolynomialCoefficients(degree=3)
        c = poly().detach()
        expected = torch.zeros(4)
        expected[0] = 1.0
        assert torch.equal(c, expected), "default must stay c = [1, 0, …] exactly"

    def test_default_consumes_no_rng(self):
        # The byte-identical guard: with the noise off, construction must not draw
        # from the seeded RNG — otherwise every downstream init (other heads, the
        # decoder) shifts and checkpoint parity with a pre-feature model breaks.
        before = torch.get_rng_state()
        PolynomialCoefficients(degree=3)               # default off
        PolynomialCoefficients(degree=3, poly_init_noise=0.0)  # explicit off
        after = torch.get_rng_state()
        assert torch.equal(before, after), "no RNG may be consumed when noise is off"

    def test_noise_is_seeded_and_reproducible(self):
        torch.manual_seed(7)
        a = PolynomialCoefficients(degree=3, poly_init_noise=0.05)().detach()
        torch.manual_seed(7)
        b = PolynomialCoefficients(degree=3, poly_init_noise=0.05)().detach()
        assert torch.equal(a, b), "same seed must reproduce the same noise"
        # Consecutive draws under one seed differ (each head sees fresh noise).
        torch.manual_seed(7)
        first = PolynomialCoefficients(degree=3, poly_init_noise=0.05)().detach()
        second = PolynomialCoefficients(degree=3, poly_init_noise=0.05)().detach()
        assert not torch.equal(first[1:], second[1:]), "successive draws must differ"


class TestConfigField:
    def test_default_is_off(self):
        assert QuantumConfig().poly_init_noise == 0.0

    def test_negative_noise_is_rejected(self):
        # It is a noise std — a negative value (typo) must fail loudly, not
        # silently seed sign-flipped noise and leak a minus into the __pin marker.
        with pytest.raises(ValueError, match="poly_init_noise"):
            QuantumConfig(poly_init_noise=-0.05)

    def test_old_config_without_key_loads_silently_as_off(self):
        # Unlike the CVQNN block (which raises a loud migration guard on a missing
        # key), an absent poly_init_noise reloads silently as 0.0 — off is
        # byte-identical, so no migration is needed.
        cfg = ExperimentConfig(
            model="quantum",
            data=DataConfig(image_size=14, patch_size=7, num_classes=10),
            quantum=QuantumConfig(num_modes=2, cutoff_dim=4, num_heads=2),
        )
        raw = asdict(cfg)
        raw["quantum"].pop("poly_init_noise")
        reconstructed = experiment_config_from_dict(raw)  # must not raise
        assert reconstructed.quantum.poly_init_noise == 0.0


def _quantum_config(poly_init_noise: float = 0.0, **overrides) -> QuantumConfig:
    base = dict(
        num_modes=2, num_layers=1, cutoff_dim=4, num_heads=2,
        cnn_channels_1=4, cnn_channels_2=8, cnn_kernel_size=3,
        decoder_hidden_dim=16, poly_degree=3, dtype="complex64",
        cvqnn_num_layers=0, poly_init_noise=poly_init_noise,
    )
    base.update(overrides)
    return QuantumConfig(**base)


class TestModelThreading:
    def _build(self, tiny_data_config, poly_init_noise, model="quantum"):
        from cv_quixer.models import build_model
        return build_model(ExperimentConfig(
            model=model, data=tiny_data_config,
            quantum=_quantum_config(poly_init_noise),
        ))

    def test_noise_reaches_every_head(self, tiny_data_config):
        torch.manual_seed(0)
        model = self._build(tiny_data_config, 0.05)
        for head in model.cv_attention.heads:
            c = head.poly_coeffs().detach()
            assert c[0].item() == 1.0
            assert (c[1:].abs() > 0).all(), "every head's c_{j>=1} must be seeded"

    def test_heads_get_different_noise(self, tiny_data_config):
        torch.manual_seed(0)
        model = self._build(tiny_data_config, 0.05)
        c0 = model.cv_attention.heads[0].poly_coeffs().detach()
        c1 = model.cv_attention.heads[1].poly_coeffs().detach()
        assert not torch.equal(c0[1:], c1[1:]), "per-head noise must break symmetry"

    def test_shared_model_also_threads_noise(self, tiny_data_config):
        torch.manual_seed(0)
        model = self._build(tiny_data_config, 0.05, model="quantum_shared")
        for head in model.cv_attention.heads:
            assert (head.poly_coeffs().detach()[1:].abs() > 0).all()

    def test_off_leaves_identity_coeffs(self, tiny_data_config):
        torch.manual_seed(0)
        model = self._build(tiny_data_config, 0.0)
        for head in model.cv_attention.heads:
            c = head.poly_coeffs().detach()
            expected = torch.zeros_like(c)
            expected[0] = 1.0
            assert torch.equal(c, expected), "off must leave c = [1, 0, …]"

    def test_state_dict_keys_unchanged_by_noise(self, tiny_data_config):
        torch.manual_seed(0)
        off = self._build(tiny_data_config, 0.0)
        torch.manual_seed(0)
        on = self._build(tiny_data_config, 0.05)
        assert set(off.state_dict()) == set(on.state_dict()), (
            "poly_init_noise must not add/rename state_dict keys (still self.c)"
        )
        assert any(k.endswith("poly_coeffs.c") for k in on.state_dict())


class TestCoordinateRegistration:
    def test_filterable_field_registered(self):
        import _run_selection as rs

        field = rs._FIELDS_BY_NAME.get("poly_init_noise")
        assert field is not None, "poly_init_noise must be a filterable coordinate"
        assert field.py_type is float
        assert field.config_path == ("quantum", "poly_init_noise")

    def test_report_treats_it_as_config_identity(self):
        import report_sweep

        assert "poly_init_noise" in report_sweep.CONFIG_IDENTITY_FIELDS, (
            "re-rolls (poly_init_noise on) must not be seed-averaged with their "
            "originals (off) — it has to be a configuration-identity coordinate"
        )


class TestSweepAxis:
    def _manifest(self, *argv):
        import sweep
        return sweep.build_manifest(sweep.build_parser().parse_args(list(argv)))

    def test_poly_init_noise_is_a_grid_axis(self):
        manifest = self._manifest(
            "--observables", "xpxsps", "--poly-init-noise", "0.05", "0.1",
        )
        assert manifest["n_runs"] == 2, "one run per eps value"
        assert manifest["axes"]["poly_init_noise"] == [0.05, 0.1]

    def test_run_name_marker_and_forwarded_flag(self):
        manifest = self._manifest(
            "--observables", "xpxsps", "--poly-init-noise", "0.05",
        )
        (run,) = manifest["runs"]
        assert "__pin0.05" in run["run_name"], "eps must mark the run dir name"
        args = run["args"]
        assert args[args.index("--poly-init-noise") + 1] == "0.05"

    def test_absent_axis_emits_no_flag_or_marker(self):
        # Inheriting full_experiment's default must be byte-identical to a no-axis
        # sweep: no --poly-init-noise flag, no __pin marker.
        manifest = self._manifest("--observables", "xpxsps", "--num-heads", "4")
        (run,) = manifest["runs"]
        assert "__pin" not in run["run_name"]
        assert "--poly-init-noise" not in run["args"]

    def test_standalone_poly_init_noise_sweep_is_allowed(self, tmp_path, monkeypatch):
        # poly_init_noise is a valid sole axis (canonical model): main()'s
        # under-specified guard must accept it like any other manual axis.
        import sweep
        monkeypatch.setattr(sys, "argv", [
            "sweep.py", "--poly-init-noise", "0.05", "--observables", "xpxsps",
            "--sweeps-root", str(tmp_path), "--dry-run",
        ])
        sweep.main()  # must not SystemExit on the "specify an axis" guard


class TestSharedHelpers:
    def test_read_exclude_file_delegates_to_shared_parser(self):
        import _run_selection
        import report_sweep
        assert report_sweep.read_exclude_file is _run_selection.read_run_names_file

    def test_reroll_prefix_is_single_source(self):
        import _run_selection
        import report_sweep
        import rerun_sweep
        assert report_sweep.REROLL_PREFIX is _run_selection.REROLL_PREFIX
        assert rerun_sweep.REROLL_PREFIX is _run_selection.REROLL_PREFIX


def _make_sweep(tmp_path, runs):
    """Synthetic sweep dir: sweep_manifest.json + per-run dirs (cf. test_resume_sweep).

    ``runs`` items: ``run_name``, ``args`` (original argv), optional ``reroll_exists``
    (whether a reroll__…__pin dir already sits in the sweep dir).
    """
    import json as _json
    sweep_dir = tmp_path / "grid_test_2026-01-01_00-00-00"
    sweep_dir.mkdir()
    manifest = {
        "sweep_name": "grid_test",
        "sweep_dir": str(sweep_dir),
        "n_runs": len(runs),
        "runs": [
            {"index": i, "run_name": r["run_name"], "args": list(r["args"])}
            for i, r in enumerate(runs)
        ],
    }
    with open(sweep_dir / "sweep_manifest.json", "w") as f:
        _json.dump(manifest, f)
    for r in runs:
        (sweep_dir / r["run_name"]).mkdir()
    return sweep_dir


def _orig_args(run_name):
    return [
        "--observables", "xpxsps_pnr", "--seed", "42", "--num-layers", "1",
        "--run-name", run_name, "--runs-root", "results/sweeps/grid_test",
        "--epochs", "4", "--poly-degree", "1",
    ]


class TestRerunSweep:
    def test_reroll_entry_injects_noise_and_provenance_no_resume(self, tmp_path):
        import rerun_sweep
        sweep_dir = _make_sweep(tmp_path, [
            {"run_name": "manual__obs__pd1__dnl3", "args": _orig_args("manual__obs__pd1__dnl3")},
        ])
        manifest = rerun_sweep.build_manifest(
            sweep_dir, poly_init_noises=[0.05],
            patterns=["manual__obs__pd1__dnl3"],
        )
        (entry,) = manifest["runs"]
        args = entry["args"]
        assert "--resume" not in args, "a re-roll restarts fresh — never resumes"
        assert args[args.index("--poly-init-noise") + 1] == "0.05"
        assert args[args.index("--reroll-of") + 1] == "manual__obs__pd1__dnl3"
        new_name = args[args.index("--run-name") + 1]
        assert new_name == "reroll__manual__obs__pd1__dnl3__pin0.05"
        assert entry["run_name"] == new_name
        # Original epochs preserved (verbatim replay).
        assert args[args.index("--epochs") + 1] == "4"

    def test_include_file_selects_only_listed_runs(self, tmp_path):
        import rerun_sweep
        sweep_dir = _make_sweep(tmp_path, [
            {"run_name": "dead_a", "args": _orig_args("dead_a")},
            {"run_name": "healthy_b", "args": _orig_args("healthy_b")},
        ])
        listing = sweep_dir / "low_accuracy_runs.txt"
        listing.write_text("# collapsed\ndead_a   0.10 0.10\n")
        manifest = rerun_sweep.build_manifest(
            sweep_dir, poly_init_noises=[0.05], runs_file=listing,
        )
        names = {e["run_name"] for e in manifest["runs"]}
        assert names == {"reroll__dead_a__pin0.05"}, "only listed runs re-rolled"

    def test_multiple_eps_is_cartesian(self, tmp_path):
        import rerun_sweep
        sweep_dir = _make_sweep(tmp_path, [
            {"run_name": "dead_a", "args": _orig_args("dead_a")},
        ])
        manifest = rerun_sweep.build_manifest(
            sweep_dir, poly_init_noises=[0.05, 0.1], patterns=["dead_a"],
        )
        names = {e["run_name"] for e in manifest["runs"]}
        assert names == {"reroll__dead_a__pin0.05", "reroll__dead_a__pin0.1"}
        assert [e["index"] for e in manifest["runs"]] == [0, 1]

    def test_existing_reroll_dir_is_skipped(self, tmp_path):
        import rerun_sweep
        sweep_dir = _make_sweep(tmp_path, [
            {"run_name": "dead_a", "args": _orig_args("dead_a")},
        ])
        (sweep_dir / "reroll__dead_a__pin0.05").mkdir()
        manifest = rerun_sweep.build_manifest(
            sweep_dir, poly_init_noises=[0.05], patterns=["dead_a"],
        )
        assert manifest["runs"] == []
        assert any(s["run_name"] == "reroll__dead_a__pin0.05"
                   for s in manifest["skipped"])

    def test_epochs_override(self, tmp_path):
        import rerun_sweep
        sweep_dir = _make_sweep(tmp_path, [
            {"run_name": "dead_a", "args": _orig_args("dead_a")},
        ])
        manifest = rerun_sweep.build_manifest(
            sweep_dir, poly_init_noises=[0.05], patterns=["dead_a"], target_epochs=8,
        )
        (entry,) = manifest["runs"]
        assert entry["args"][entry["args"].index("--epochs") + 1] == "8"

    def test_no_selection_raises(self, tmp_path):
        import rerun_sweep
        sweep_dir = _make_sweep(tmp_path, [
            {"run_name": "dead_a", "args": _orig_args("dead_a")},
        ])
        with pytest.raises(ValueError):
            rerun_sweep.build_manifest(sweep_dir, poly_init_noises=[0.05])


def _row(run_name, *, reroll_of=None, n_epochs=4, best=0.5):
    return {"run_name": run_name, "reroll_of": reroll_of,
            "n_epochs": n_epochs, "best_test_acc": best}


class TestRerollReporting:
    def test_ignore_drops_reroll_rows(self):
        import report_sweep
        rows = [_row("orig_a"), _row("reroll__orig_a__pin0.05", reroll_of="orig_a")]
        kept = report_sweep.select_for_rerolls(rows, "ignore")
        assert {r["run_name"] for r in kept} == {"orig_a"}

    def test_replace_substitutes_reroll_for_its_original(self):
        import report_sweep
        rows = [
            _row("orig_a"), _row("reroll__orig_a__pin0.05", reroll_of="orig_a"),
            _row("orig_b"),  # untouched — no re-roll
        ]
        kept = report_sweep.select_for_rerolls(rows, "replace")
        assert {r["run_name"] for r in kept} == {
            "reroll__orig_a__pin0.05", "orig_b",
        }

    def test_compare_keeps_only_paired_configs(self):
        import report_sweep
        rows = [
            _row("orig_a"), _row("reroll__orig_a__pin0.05", reroll_of="orig_a"),
            _row("orig_b"),  # no re-roll → excluded from the comparison
        ]
        kept = report_sweep.select_for_rerolls(rows, "compare")
        assert {r["run_name"] for r in kept} == {
            "orig_a", "reroll__orig_a__pin0.05",
        }

    def test_common_max_epoch_is_the_min(self):
        import report_sweep
        rows = [_row("orig_a", n_epochs=4),
                _row("reroll__orig_a__pin0.05", reroll_of="orig_a", n_epochs=6)]
        assert report_sweep._common_max_epoch(rows) == 4

    def test_old_run_poly_init_noise_defaults_to_zero(self, tmp_path):
        import report_sweep
        run_dir = tmp_path / "orig_a"
        run_dir.mkdir()
        # An old, pre-feature run: meta has no poly_init_noise key.
        history = {"epoch": {"test_acc": [0.1, 0.2], "train_acc": [0.1, 0.2]},
                   "meta": {"seed": 42, "model": "quantum"}}
        import json as _json
        with open(run_dir / "history.json", "w") as f:
            _json.dump(history, f)
        row = report_sweep._load_run(run_dir)
        assert row["poly_init_noise"] == 0.0, "absent poly_init_noise must read as 0.0"
        assert row["reroll_of"] is None

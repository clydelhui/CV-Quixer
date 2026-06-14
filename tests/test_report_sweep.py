"""Tests for experiments/report_sweep.py + report_sweep_compare.py (JSON only).

These pin the configuration-identity grouping that fixes the manual-sweep
collapse: cross-run figures must seed-average only true seed repeats, never
distinct architectures that share the manual-mode placeholder coordinates
(target_params=-1 / default scaling_knob).
"""

from __future__ import annotations

import csv
import json
import sys
import warnings
from pathlib import Path

import pytest

# The report scripts live in experiments/ (not a package); import them the same
# way report_sweep_compare.py imports report_sweep — via sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

import report_sweep  # noqa: E402
import report_sweep_compare  # noqa: E402

# One complete history["meta"] as full_experiment.py writes it (every
# CONFIG_IDENTITY_FIELDS member populated), overridden per test.
BASE_META = {
    "model": "quantum",
    "observables_name": "xpxsps",
    "target_params": -1,
    "scaling_knob": "num_heads",
    "achieved_params": 10_000,
    "num_layers": 2,
    "trunc_lambda": 0.1,
    "seed": 42,
    "total_runtime_sec": 100.0,
    "device": "cpu",
    "num_modes": 2,
    "num_heads": 4,
    "cutoff_dim": 6,
    "poly_degree": 3,
    "cnn_channels_1": 8,
    "cnn_channels_2": 8,
    "cnn_kernel_size": 3,
    "decoder_hidden_dim": 64,
    "cnn_num_conv_layers": 2,
    "hypernet_num_linear_layers": 1,
    "decoder_num_layers": 2,
    "decoder_hidden_mult": None,
    "cvqnn_num_layers": 1,
    "num_seq2seq_blocks": 1,
    "pooling": "mean",
    "block_residual": True,
    "query_trunc_lambda": 0.01,
    "cvqnn_trunc_lambda": 0.01,
}


def make_run(sweep_dir, run_name, *, test_acc=(0.5, 0.6), **meta_overrides):
    """Write a minimal history.json for one synthetic run."""
    meta = {**BASE_META, **meta_overrides}
    meta.setdefault("best_test_acc", max(test_acc))
    meta.setdefault("best_epoch", test_acc.index(max(test_acc)) + 1)
    run_dir = sweep_dir / run_name
    run_dir.mkdir(parents=True)
    history = {
        "meta": meta,
        "epoch": {
            "test_acc": list(test_acc),
            "train_acc": list(test_acc),
            "train_loss": [1.0] * len(test_acc),
            "test_loss": [1.0] * len(test_acc),
        },
    }
    (run_dir / "history.json").write_text(json.dumps(history))
    return run_dir


def run_report_sweep(sweep_dir, monkeypatch, *extra):
    monkeypatch.setattr(
        sys, "argv",
        ["report_sweep.py", "--sweep-dir", str(sweep_dir),
         "--skip-per-run-figures", *extra],
    )
    report_sweep.main()


# ---------------------------------------------------------------------------
# Configuration-identity grouping
# ---------------------------------------------------------------------------


def test_budget_mode_seed_groups(tmp_path):
    """Budget sweeps keep one seed-averaged group per (budget, observable)."""
    sweep = tmp_path / "sweep"
    for tp, nh in [(8000, 3), (20000, 7)]:
        for obs in ("x", "xpxsps"):
            for seed, acc, ach in [(42, 0.60, tp + 10), (43, 0.70, tp - 10)]:
                make_run(
                    sweep, f"p{tp}__obs-{obs}__seed{seed}",
                    test_acc=(acc - 0.1, acc),
                    target_params=tp, num_heads=nh, observables_name=obs,
                    seed=seed, achieved_params=ach,
                )
    rows = report_sweep.load_sweep(sweep)
    groups = report_sweep._config_groups(rows)
    assert len(groups) == 4
    for g in groups.values():
        assert g["acc"][0] == pytest.approx(0.65)
        assert g["acc"][1] == pytest.approx(0.05)
    assert sorted(g["x"] for g in groups.values()) == [8000, 8000, 20000, 20000]


def test_manual_mode_configs_not_collapsed(tmp_path):
    """The bug: manual configs all share (obs, knob, -1) and averaged into one."""
    sweep = tmp_path / "sweep"
    acc = 0.5
    for nh in (4, 6):
        for nm in (2, 3):
            for seed in (42, 43):
                acc += 0.02
                make_run(
                    sweep, f"manual__obs-xpxsps__seed{seed}__nh{nh}__nm{nm}",
                    test_acc=(0.4, acc),
                    num_heads=nh, num_modes=nm, seed=seed,
                    achieved_params=1000 * nh + 10 * nm,
                )
    rows = report_sweep.load_sweep(sweep)
    assert all(r["target_params"] == -1 for r in rows)
    groups = report_sweep._config_groups(rows)
    assert len(groups) == 4  # one per architecture, not one blended point


def test_mixed_models_grouped_separately(tmp_path):
    sweep = tmp_path / "sweep"
    make_run(sweep, "manual__obs-xpxsps__seed42", test_acc=(0.5, 0.6))
    make_run(
        sweep, "manual__obs-xpxsps__seed42__stacked",
        test_acc=(0.5, 0.7), model="quantum_stacked",
    )
    rows = report_sweep.load_sweep(sweep)
    groups = report_sweep._config_groups(rows)
    assert len(groups) == 2
    series = {
        report_sweep._series_key(k, ["model", "observables"]) for k in groups
    }
    assert series == {("quantum", "xpxsps"), ("quantum_stacked", "xpxsps")}


def test_varying_fields():
    base = {f: BASE_META.get(f) for f in report_sweep.CONFIG_IDENTITY_FIELDS}
    base["observables"] = "xpxsps"
    k1 = report_sweep._config_key(base)
    k2 = report_sweep._config_key({**base, "num_heads": 6})
    k3 = report_sweep._config_key({**base, "num_heads": 6, "num_modes": 3})
    assert report_sweep._varying_fields([k1, k2]) == ["num_heads"]
    assert set(report_sweep._varying_fields([k1, k2, k3])) == {
        "num_heads", "num_modes",
    }
    assert report_sweep._varying_fields([k1]) == []


# ---------------------------------------------------------------------------
# Identity drift guard
# ---------------------------------------------------------------------------


def test_identity_drift_guard_warns(tmp_path):
    sweep = tmp_path / "sweep"
    # Same identity fields, but the run names carry an axis marker (a fake
    # __gb knob) that CONFIG_IDENTITY_FIELDS does not track.
    make_run(sweep, "manual__obs-xpxsps__seed42__gb1.5", seed=42)
    make_run(sweep, "manual__obs-xpxsps__seed43", seed=43)
    rows = report_sweep.load_sweep(sweep)
    with pytest.warns(RuntimeWarning, match="configuration identity"):
        report_sweep._check_identity_drift(rows)


def test_no_drift_warning_for_seed_repeats(tmp_path):
    sweep = tmp_path / "sweep"
    make_run(sweep, "manual__obs-xpxsps__seed42", seed=42)
    make_run(sweep, "manual__obs-xpxsps__seed43", seed=43)
    rows = report_sweep.load_sweep(sweep)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        report_sweep._check_identity_drift(rows)


# ---------------------------------------------------------------------------
# Vary rule (figures render only when their axis takes ≥2 values)
# ---------------------------------------------------------------------------


def test_single_config_skips_cross_run_figures(tmp_path, monkeypatch):
    sweep = tmp_path / "sweep"
    make_run(sweep, "manual__obs-xpxsps__seed42", seed=42)
    make_run(sweep, "manual__obs-xpxsps__seed43", seed=43)
    run_report_sweep(sweep, monkeypatch)
    figs = sweep / "figures"
    assert (sweep / "summary.csv").is_file()
    assert not (figs / "acc_vs_params.png").exists()
    assert not (figs / "acc_by_observable.png").exists()
    assert not (figs / "acc_vs_trunc_lambda.png").exists()


def test_varying_axis_renders_its_figure(tmp_path, monkeypatch):
    sweep = tmp_path / "sweep"
    for layers, ach in [(2, 9000), (3, 11000)]:
        make_run(
            sweep, f"manual__obs-xpxsps__seed42__L{layers}",
            num_layers=layers, achieved_params=ach,
        )
    run_report_sweep(sweep, monkeypatch)
    figs = sweep / "figures"
    assert (figs / "acc_vs_params.png").is_file()
    assert (figs / "acc_vs_num_layers.png").is_file()
    assert not (figs / "acc_by_observable.png").exists()  # single preset
    assert not (figs / "acc_vs_trunc_lambda.png").exists()  # single λ
    assert not (figs / "acc_vs_num_heads.png").exists()  # constant field


def test_acc_vs_field_all_else_equal_lines(tmp_path, monkeypatch):
    """acc_vs_<field> connects all-else-equal chains when other axes also vary.

    A 2x2 (num_heads x num_modes) sweep: acc_vs_num_modes must draw one line per
    num_heads value (2 chains), each connecting that value's 2 num_modes points
    — not a single scatter (the pre-trend-line behaviour, where num_modes is not
    the sole varying axis would have suppressed the line).
    """
    sweep = tmp_path / "sweep"
    acc = 0.5
    for nh in (4, 6):
        for nm in (2, 3):
            acc += 0.02
            make_run(
                sweep, f"manual__obs-xpxsps__seed42__nh{nh}__nm{nm}",
                test_acc=(0.4, acc), num_heads=nh, num_modes=nm,
                achieved_params=1000 * nh + 10 * nm,
            )

    import matplotlib.axes

    calls: list[dict] = []
    orig_errorbar = matplotlib.axes.Axes.errorbar

    def spy(self, x, y, *a, **kw):
        calls.append(
            {"x": list(x), "label": kw.get("label"), "ls": kw.get("linestyle")}
        )
        return orig_errorbar(self, x, y, *a, **kw)

    monkeypatch.setattr(matplotlib.axes.Axes, "errorbar", spy)

    rows = report_sweep.load_sweep(sweep)
    fig_dir = sweep / "figures"
    fig_dir.mkdir()
    report_sweep.plot_acc_vs_hyperparam(rows, fig_dir)

    assert (fig_dir / "acc_vs_num_modes.png").is_file()
    assert (fig_dir / "acc_vs_num_heads.png").is_file()
    # Isolate the num_modes figure's chains (x sweeps {2, 3}): one per num_heads.
    modes_chains = [c for c in calls if sorted(c["x"]) == [2, 3]]
    assert len(modes_chains) == 2
    assert all(len(c["x"]) == 2 and c["ls"] == "-" for c in modes_chains)
    assert {c["label"] for c in modes_chains} == {"4", "6"}


def test_acc_vs_field_excludes_dependent_decoder_hidden_dim(tmp_path, monkeypatch):
    """A multiplier-sized decoder_hidden_dim must not break num_modes lines.

    With decoder_hidden_mult set, decoder_hidden_dim is slaved to (num_modes,
    num_heads), so it co-varies with the axis being plotted. It must be excluded
    from the all-else-equal key, otherwise every num_modes chain is a lone point.
    """
    sweep = tmp_path / "sweep"
    for nh in (4, 6):
        for nm in (2, 3):
            make_run(
                sweep, f"manual__obs-xpxsps__seed42__nh{nh}__nm{nm}",
                test_acc=(0.4, 0.5), num_heads=nh, num_modes=nm,
                decoder_hidden_mult=4.0,
                decoder_hidden_dim=100 * nh * nm,  # slaved to both axes
                achieved_params=1000 * nh + nm,
            )

    rows = report_sweep.load_sweep(sweep)
    assert report_sweep._dependent_fields(rows) == {"decoder_hidden_dim"}

    import matplotlib.axes

    calls: list[dict] = []
    orig_errorbar = matplotlib.axes.Axes.errorbar

    def spy(self, x, y, *a, **kw):
        calls.append({"x": list(x), "label": kw.get("label")})
        return orig_errorbar(self, x, y, *a, **kw)

    monkeypatch.setattr(matplotlib.axes.Axes, "errorbar", spy)

    fig_dir = sweep / "figures"
    fig_dir.mkdir()
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the duplicate-x guard must not fire
        report_sweep.plot_acc_vs_hyperparam(rows, fig_dir)

    # num_modes: one 2-point line per num_heads (decoder_hidden_dim excluded).
    modes_chains = [c for c in calls if sorted(c["x"]) == [2, 3]]
    assert len(modes_chains) == 2
    assert {c["label"] for c in modes_chains} == {"4", "6"}  # labelled by num_heads
    # decoder_hidden_dim is dependent → no connected line (all single-point).
    assert (fig_dir / "acc_vs_decoder_hidden_dim.png").is_file()
    dhd_chains = [c for c in calls if c["x"] and c["x"][0] in (800, 1200, 1800)]
    assert all(len(c["x"]) == 1 for c in dhd_chains)


def test_acc_vs_field_legend_suppressed_when_many_lines(tmp_path, monkeypatch):
    """Past MAX_LEGEND_CHAINS trend lines the legend is dropped, lines kept."""
    sweep = tmp_path / "sweep"
    n_chains = report_sweep.MAX_LEGEND_CHAINS + 1
    # One chain per cnn_channels_2 value (the "other" varying field), each with
    # two num_modes points so it is a genuine 2-point line.
    for c2 in range(8, 8 + n_chains):
        for nm in (2, 3):
            make_run(
                sweep, f"manual__obs-xpxsps__seed42__c2{c2}__nm{nm}",
                test_acc=(0.4, 0.5), cnn_channels_2=c2, num_modes=nm,
                achieved_params=100 * c2 + nm,
            )

    import matplotlib.axes

    legend_calls: list = []
    orig_legend = matplotlib.axes.Axes.legend

    def spy(self, *a, **kw):
        legend_calls.append((a, kw))
        return orig_legend(self, *a, **kw)

    monkeypatch.setattr(matplotlib.axes.Axes, "legend", spy)

    rows = report_sweep.load_sweep(sweep)
    fig_dir = sweep / "figures"
    fig_dir.mkdir()
    report_sweep.plot_acc_vs_hyperparam(rows, fig_dir)

    # acc_vs_num_modes has n_chains (>cap) lines → no legend; acc_vs_cnn_channels_2
    # has 2 lines (one per num_modes) → legend drawn. So exactly one legend call.
    assert (fig_dir / "acc_vs_num_modes.png").is_file()
    assert (fig_dir / "acc_vs_cnn_channels_2.png").is_file()
    assert len(legend_calls) == 1


def test_two_observables_render_bar_figure(tmp_path, monkeypatch):
    sweep = tmp_path / "sweep"
    for obs, ach in [("x", 9000), ("xpxsps", 11000)]:
        make_run(
            sweep, f"manual__obs-{obs}__seed42",
            observables_name=obs, achieved_params=ach,
        )
    run_report_sweep(sweep, monkeypatch)
    assert (sweep / "figures" / "acc_by_observable.png").is_file()


def test_lambda_axis_renders_lambda_figure(tmp_path, monkeypatch):
    sweep = tmp_path / "sweep"
    for lam in (0.01, 0.1):
        make_run(sweep, f"manual__obs-xpxsps__seed42__tl{lam}", trunc_lambda=lam)
    run_report_sweep(sweep, monkeypatch)
    assert (sweep / "figures" / "acc_vs_trunc_lambda.png").is_file()


# ---------------------------------------------------------------------------
# --series-by flag
# ---------------------------------------------------------------------------


def test_series_by_rejects_unknown_field(tmp_path, monkeypatch):
    sweep = tmp_path / "sweep"
    make_run(sweep, "manual__obs-xpxsps__seed42")
    monkeypatch.setattr(
        sys, "argv",
        ["report_sweep.py", "--sweep-dir", str(sweep), "--series-by", "bogus"],
    )
    with pytest.raises(SystemExit):
        report_sweep.main()


def test_series_by_custom_field(tmp_path, monkeypatch):
    sweep = tmp_path / "sweep"
    for hll, ach in [(1, 9000), (2, 11000)]:
        make_run(
            sweep, f"manual__obs-xpxsps__seed42__hll{hll}",
            hypernet_num_linear_layers=hll, achieved_params=ach,
        )
    run_report_sweep(sweep, monkeypatch, "--series-by", "hypernet_num_linear_layers")
    assert (sweep / "figures" / "acc_vs_params.png").is_file()


# ---------------------------------------------------------------------------
# report_sweep_compare.py
# ---------------------------------------------------------------------------


def test_compare_two_sweeps(tmp_path, monkeypatch):
    manual = tmp_path / "sweep_manual"
    budget = tmp_path / "sweep_budget"
    for nh, ach in [(4, 9000), (6, 11000)]:
        for seed in (42, 43):
            make_run(
                manual, f"manual__obs-xpxsps__seed{seed}__nh{nh}",
                num_heads=nh, seed=seed, achieved_params=ach,
            )
    for tp in (8000, 20000):
        make_run(
            budget, f"p{tp}__obs-xpxsps__seed42",
            target_params=tp, achieved_params=tp, model="quantum_shared",
        )
    out = tmp_path / "out"
    monkeypatch.setattr(
        sys, "argv",
        ["report_sweep_compare.py",
         "--sweep-dir", str(manual), "--sweep-dir", str(budget),
         "--label", "manual", "--label", "budget",
         "--out-dir", str(out)],
    )
    report_sweep_compare.main()
    csv_lines = (out / "comparison.csv").read_text().strip().splitlines()
    assert len(csv_lines) == 1 + 6  # header + 6 runs
    assert (out / "figures" / "acc_vs_params_compare.png").is_file()
    # The manual sweep's two architectures stay distinct points (no collapse).
    rows = report_sweep_compare.load_sweeps(
        [manual, budget], ["manual", "budget"]
    )
    per = report_sweep_compare._per_label_groups(rows)
    assert len([1 for (lbl, _k) in per if lbl == "manual"]) == 2
    assert len([1 for (lbl, _k) in per if lbl == "budget"]) == 2


# ---------------------------------------------------------------------------
# Epoch-fair reporting: --max-epoch cap + heterogeneous-epoch warning
# ---------------------------------------------------------------------------


def test_max_epoch_caps_derived_metrics(tmp_path):
    """With max_epoch=N, best/best-epoch/final/n_epochs come from the first N
    epochs of the history series — not meta's all-epochs values — so a run
    topped up beyond its neighbours can be compared fairly at epoch N."""
    sweep = tmp_path / "sweep"
    make_run(sweep, "p8000__obs-xpxsps__seed42",
             test_acc=(0.5, 0.6, 0.8, 0.9))  # peaks late; meta best = 0.9

    (row,) = report_sweep.load_sweep(sweep, max_epoch=2)

    assert row["best_test_acc"] == pytest.approx(0.6)
    assert row["best_epoch"] == 2
    assert row["final_test_acc"] == pytest.approx(0.6)
    assert row["final_train_acc"] == pytest.approx(0.6)
    assert row["n_epochs"] == 2


def test_run_below_cap_is_kept_with_warning(tmp_path):
    """A run with fewer epochs than the cap still appears in the rows (a
    mid-top-up report must not silently lose runs), with a RuntimeWarning
    naming the run and its actual epoch count."""
    sweep = tmp_path / "sweep"
    make_run(sweep, "p8000__obs-xpxsps__seed42", test_acc=(0.5,))

    with pytest.warns(RuntimeWarning, match=r"p8000__obs-xpxsps__seed42.*1"):
        (row,) = report_sweep.load_sweep(sweep, max_epoch=2)

    assert row["n_epochs"] == 1
    assert row["best_test_acc"] == pytest.approx(0.5)


def test_heterogeneous_epoch_counts_warn(tmp_path):
    """Comparing runs trained for different numbers of epochs is unfair (best
    acc over more epochs is advantaged) — the report must say so loudly."""
    sweep = tmp_path / "sweep"
    make_run(sweep, "p8000__obs-xpxsps__seed42", test_acc=(0.5, 0.6))
    make_run(sweep, "p8000__obs-xpxsps__seed43",
             test_acc=(0.5, 0.6, 0.7, 0.8), seed=43)
    rows = report_sweep.load_sweep(sweep)

    with pytest.warns(RuntimeWarning, match="epoch count"):
        report_sweep._check_epoch_heterogeneity(rows)


def test_homogeneous_epoch_counts_do_not_warn(tmp_path):
    """No crying wolf: equal epoch counts (the normal, fair case) are silent."""
    sweep = tmp_path / "sweep"
    make_run(sweep, "p8000__obs-xpxsps__seed42", test_acc=(0.5, 0.6))
    make_run(sweep, "p8000__obs-xpxsps__seed43", test_acc=(0.5, 0.7), seed=43)
    rows = report_sweep.load_sweep(sweep)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        report_sweep._check_epoch_heterogeneity(rows)


def test_main_max_epoch_caps_summary_and_warns_on_heterogeneity(
    tmp_path, monkeypatch
):
    """End-to-end: a lopsided sweep warns through main(); with --max-epoch the
    summary table carries the capped metrics."""
    sweep = tmp_path / "sweep"
    make_run(sweep, "p8000__obs-xpxsps__seed42", test_acc=(0.5, 0.6))
    make_run(sweep, "p8000__obs-xpxsps__seed43",
             test_acc=(0.5, 0.6, 0.8, 0.9), seed=43)

    with pytest.warns(RuntimeWarning, match="epoch count"):
        run_report_sweep(sweep, monkeypatch)

    run_report_sweep(sweep, monkeypatch, "--max-epoch", "2")
    rows = list(csv.DictReader(open(sweep / "summary.csv")))
    by_seed = {r["seed"]: r for r in rows}
    assert float(by_seed["43"]["best_test_acc"]) == pytest.approx(0.6)
    assert by_seed["43"]["n_epochs"] == "2"
    assert by_seed["42"]["n_epochs"] == "2"


def test_compare_inherits_max_epoch_and_heterogeneity_guard(
    tmp_path, monkeypatch
):
    """report_sweep_compare honours --max-epoch through the shared loaders and
    warns when epoch counts differ across the combined sweeps."""
    sweep_a, sweep_b = tmp_path / "a", tmp_path / "b"
    make_run(sweep_a, "p8000__obs-xpxsps__seed42", test_acc=(0.5, 0.6))
    make_run(sweep_b, "p8000__obs-xpxsps__seed42",
             test_acc=(0.5, 0.6, 0.8, 0.9), model="quantum_shared")
    out = tmp_path / "out"

    def run_compare(*extra):
        monkeypatch.setattr(
            sys, "argv",
            ["report_sweep_compare.py",
             "--sweep-dir", str(sweep_a), "--sweep-dir", str(sweep_b),
             "--label", "a", "--label", "b", "--out-dir", str(out), *extra],
        )
        report_sweep_compare.main()

    with pytest.warns(RuntimeWarning, match="epoch count"):
        run_compare()

    run_compare("--max-epoch", "2")
    rows = list(csv.DictReader(open(out / "comparison.csv")))
    by_label = {r["sweep_label"]: r for r in rows}
    assert float(by_label["b"]["best_test_acc"]) == pytest.approx(0.6)
    assert by_label["b"]["n_epochs"] == "2"

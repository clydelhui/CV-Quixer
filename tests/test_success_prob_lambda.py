"""Success-probability figure inputs in experiments/report_diagnostics.py.

The λ/β derivation and the two shape/stage helpers the success-probability
figures depend on — all numpy-only — plus a synthetic-artefact round trip
covering the stacked model's rank-3 `success_probs` and block-prefixed
coefficient keys (ADR-0002, ADR-0003).
"""

import numpy as np
import pytest


def test_derive_lcu_lambda_hand_computed():
    from experiments.report_diagnostics import _derive_lcu_lambda

    # head 0: b = [3+0j, 0+4j] → α = 3 + 4 = 7;  c = [1, -2, 0.5]
    #   λ = |1|·7⁰ + |−2|·7¹ + |0.5|·7² = 1 + 14 + 24.5 = 39.5
    # head 1: b = [1+0j, 0+0j] → α = 1;          c = [0, 1, 0]   → λ = 1
    lcu = np.array([[[3.0, 0.0], [0.0, 4.0]],
                    [[1.0, 0.0], [0.0, 0.0]]], dtype=np.float32)
    poly = np.array([[1.0, -2.0, 0.5],
                     [0.0, 1.0, 0.0]], dtype=np.float32)
    lam = _derive_lcu_lambda(lcu, poly)
    assert lam.shape == (2,)
    assert lam.dtype == np.float64
    np.testing.assert_allclose(lam, [39.5, 1.0], rtol=1e-12)


# ---------------------------------------------------------------------------
# _success_prob_matrix — canonical (N, H) vs stacked (N, H, N_positions)
# ---------------------------------------------------------------------------


def test_success_prob_matrix_passes_2d_through():
    from experiments.report_diagnostics import _success_prob_matrix

    raw = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    out = _success_prob_matrix(raw)
    assert out.shape == (2, 2)
    assert out.dtype == np.float64
    np.testing.assert_allclose(out, raw)


def test_success_prob_matrix_folds_positions_into_samples():
    """(N, H, P) → (N*P, H): column h stays head h, every (sample, position)
    pair becomes one row."""
    from experiments.report_diagnostics import _success_prob_matrix

    # value = 100*sample + 10*head + position, so every element is identifiable.
    n, h, p = 2, 3, 4
    raw = np.array([[[100 * s + 10 * head + pos for pos in range(p)]
                     for head in range(h)]
                    for s in range(n)], dtype=np.float64)
    out = _success_prob_matrix(raw)
    assert out.shape == (n * p, h)
    # Row (s*p + pos) holds sample s at position pos, one column per head.
    for s in range(n):
        for pos in range(p):
            for head in range(h):
                assert out[s * p + pos, head] == 100 * s + 10 * head + pos


def test_success_prob_matrix_rejects_other_ranks():
    from experiments.report_diagnostics import (
        MissingArtefactError,
        _success_prob_matrix,
    )

    with pytest.raises(MissingArtefactError, match="expected"):
        _success_prob_matrix(np.zeros(5))
    with pytest.raises(MissingArtefactError, match="expected"):
        _success_prob_matrix(np.zeros((2, 3, 4, 5)))


# ---------------------------------------------------------------------------
# _decoder_input_stage — which namespace feeds the decoder
# ---------------------------------------------------------------------------


def test_decoder_input_stage_canonical_is_flat():
    from experiments.report_diagnostics import _decoder_input_stage

    assert _decoder_input_stage({"lcu_coeffs": 1, "poly_coeffs": 2}) == ("", "", "")


def test_decoder_input_stage_is_last_block_without_aggregator():
    from experiments.report_diagnostics import _decoder_input_stage

    diag = {"block0_lcu_coeffs": 1, "block1_lcu_coeffs": 2,
            "mean_photon_number": 3}
    prefix, suffix, _label = _decoder_input_stage(diag)
    assert (prefix, suffix) == ("block1_", "_block1")


def test_decoder_input_stage_prefers_the_aggregator():
    from experiments.report_diagnostics import _decoder_input_stage

    diag = {"block0_lcu_coeffs": 1, "block1_lcu_coeffs": 2,
            "agg_lcu_coeffs": 3}
    prefix, suffix, _label = _decoder_input_stage(diag)
    assert (prefix, suffix) == ("agg_", "_agg")


# ---------------------------------------------------------------------------
# Round trip on synthetic stacked artefacts
# ---------------------------------------------------------------------------


N_SAMPLES, N_HEADS, N_POSITIONS, N_EPOCHS = 6, 3, 4, 2


def _write_stacked_run(root, *, n_blocks=2):
    """A minimal stacked run dir: rank-3 success_probs + block-prefixed coeffs.

    Mirrors what `full_experiment.py` writes for model="quantum_stacked" under
    pooling="mean" — enough for the two success-probability figures, nothing
    more.
    """
    from cv_quixer.evaluation import artefact_schema as schema

    rng = np.random.default_rng(0)
    (root / "predictions").mkdir(parents=True)
    (root / "diagnostics").mkdir(parents=True)
    for epoch in range(1, N_EPOCHS + 1):
        np.savez_compressed(
            root / "predictions" / schema.prediction_filename(epoch),
            y_true=np.zeros(N_SAMPLES, dtype=np.int64),
            y_pred=np.zeros(N_SAMPLES, dtype=np.int64),
            y_probs=np.full((N_SAMPLES, 10), 0.1, dtype=np.float32),
            readouts=np.zeros((N_SAMPLES, N_HEADS), dtype=np.float32),
            # Raw norms in (0, 1]; the ratio to β² lands well below 1.
            success_probs=rng.uniform(
                0.05, 1.0, (N_SAMPLES, N_HEADS, N_POSITIONS)
            ).astype(np.float32),
        )
        diag = {}
        for b in range(n_blocks):
            diag[f"block{b}_{schema.LCU_COEFFS}"] = rng.normal(
                size=(N_HEADS, N_POSITIONS, 2)
            ).astype(np.float32)
            diag[f"block{b}_{schema.POLY_COEFFS}"] = rng.normal(
                size=(N_HEADS, 3)
            ).astype(np.float32)
        np.savez_compressed(
            root / "diagnostics" / schema.diagnostics_filename(epoch), **diag
        )
    return root


def _write_stage_sidecars(root, *, n_blocks=2, trunc=(0.4, 0.2, 0.01)):
    """Per-stage sidecars as `eval_block_stages.py` writes them."""
    from cv_quixer.evaluation import artefact_schema as schema

    rng = np.random.default_rng(7)
    for epoch in range(1, N_EPOCHS + 1):
        for b in range(n_blocks):
            # Block b's streams differ so the per-block figure has something to
            # separate; their mean is what the model would have recorded.
            scale = 1.0 + b
            np.savez_compressed(
                root / "predictions"
                / schema.stage_prediction_filename(epoch, f"block{b}"),
                **{schema.SUCCESS_PROBS: rng.uniform(
                    0.05, 1.0, (N_SAMPLES, N_HEADS, N_POSITIONS)
                ).astype(np.float32)},
                **{k: np.float32(v * scale)
                   for k, v in zip(schema.STAGE_TRUNC_KEYS, trunc)},
            )
    return root


def _load_npz(path):
    with np.load(path) as npz:
        return dict(npz)


def _fake_run(root, epoch, fig_dir):
    from cv_quixer.evaluation import artefact_schema as schema

    return {
        "run_dir": root,
        "epoch": epoch,
        "fig_dir": fig_dir,
        "predictions": _load_npz(
            root / "predictions" / schema.prediction_filename(epoch)
        ),
        "diagnostics": _load_npz(
            root / "diagnostics" / schema.diagnostics_filename(epoch)
        ),
        "history": {"epoch": {
            "test_acc": [0.1] * N_EPOCHS,
            "test_loss": [2.3] * N_EPOCHS,
            # Mean over blocks of the sidecar values _write_stage_sidecars
            # writes (scale 1 and 2), i.e. what the model would have recorded.
            "test_trunc_loss": [0.4 * 1.5] * N_EPOCHS,
            "test_query_trunc_loss": [0.2 * 1.5] * N_EPOCHS,
            "test_cvqnn_trunc_loss": [0.01 * 1.5] * N_EPOCHS,
        }},
    }


def test_stacked_histogram_renders_the_decoder_input_stage(tmp_path):
    """Without sidecars only the decoder-input stage has success probs, so a
    2-block run yields exactly one file — suffixed for its stage."""
    from experiments.report_diagnostics import plot_success_prob_histogram

    root = _write_stacked_run(tmp_path / "run")
    fig_dir = tmp_path / "figures"
    fig_dir.mkdir()
    plot_success_prob_histogram(_fake_run(root, N_EPOCHS, fig_dir))
    assert (fig_dir / "success_prob_histogram_block1.png").is_file()
    assert not (fig_dir / "success_prob_histogram_block0.png").exists()


def test_stacked_trajectory_renders_the_decoder_input_stage(tmp_path):
    from experiments.report_diagnostics import plot_success_prob_trajectory

    root = _write_stacked_run(tmp_path / "run")
    fig_dir = tmp_path / "figures"
    fig_dir.mkdir()
    plot_success_prob_trajectory(_fake_run(root, N_EPOCHS, fig_dir))
    assert (fig_dir / "success_prob_trajectory_block1.png").is_file()
    assert not (fig_dir / "success_prob_trajectory_block0.png").exists()


def test_sidecars_add_the_earlier_stage(tmp_path):
    """With sidecars present every stage renders, one file each."""
    from experiments.report_diagnostics import (
        plot_success_prob_histogram,
        plot_success_prob_trajectory,
    )

    root = _write_stage_sidecars(_write_stacked_run(tmp_path / "run"))
    fig_dir = tmp_path / "figures"
    fig_dir.mkdir()
    run = _fake_run(root, N_EPOCHS, fig_dir)
    plot_success_prob_histogram(run)
    plot_success_prob_trajectory(run)
    for b in (0, 1):
        assert (fig_dir / f"success_prob_histogram_block{b}.png").is_file()
        assert (fig_dir / f"success_prob_trajectory_block{b}.png").is_file()


def test_stage_prediction_filename_round_trip():
    from cv_quixer.evaluation import artefact_schema as schema

    assert schema.stage_prediction_filename(7, "block0_") == "epoch_0007_block0.npz"
    assert schema.stage_prediction_filename(7, "block0") == "epoch_0007_block0.npz"
    assert schema.stage_prediction_filename(12, "agg_") == "epoch_0012_agg.npz"
    with pytest.raises(ValueError, match="stage prefix"):
        schema.stage_prediction_filename(1, "")


# ---------------------------------------------------------------------------
# beta_trajectory — coefficients only, no predictions needed
# ---------------------------------------------------------------------------


def test_beta_trajectory_renders_one_file_per_stage(tmp_path):
    from experiments.report_diagnostics import plot_beta_trajectory

    root = _write_stacked_run(tmp_path / "run")
    fig_dir = tmp_path / "figures"
    fig_dir.mkdir()
    plot_beta_trajectory(_fake_run(root, N_EPOCHS, fig_dir))
    for b in (0, 1):
        assert (fig_dir / f"beta_trajectory_block{b}.png").is_file()


def test_beta_trajectory_keeps_canonical_filename(tmp_path):
    """A flat-key (canonical) run keeps the historic unsuffixed name."""
    from cv_quixer.evaluation import artefact_schema as schema
    from experiments.report_diagnostics import plot_beta_trajectory

    root = tmp_path / "run"
    (root / "diagnostics").mkdir(parents=True)
    rng = np.random.default_rng(3)
    for epoch in range(1, N_EPOCHS + 1):
        np.savez_compressed(
            root / "diagnostics" / schema.diagnostics_filename(epoch),
            **{schema.LCU_COEFFS: rng.normal(
                size=(N_HEADS, N_POSITIONS, 2)).astype(np.float32),
               schema.POLY_COEFFS: rng.normal(
                size=(N_HEADS, 3)).astype(np.float32)},
        )
    fig_dir = tmp_path / "figures"
    fig_dir.mkdir()
    run = {"run_dir": root, "epoch": N_EPOCHS, "fig_dir": fig_dir,
           "predictions": None,
           "diagnostics": _load_npz(
               root / "diagnostics" / schema.diagnostics_filename(N_EPOCHS)),
           "history": {"epoch": {"test_loss": [2.3] * N_EPOCHS}}}
    plot_beta_trajectory(run)
    assert (fig_dir / "beta_trajectory.png").is_file()


# ---------------------------------------------------------------------------
# Per-block truncation curves
# ---------------------------------------------------------------------------


def test_trunc_streams_per_block_renders_one_file_per_stream(tmp_path):
    from experiments.report_diagnostics import plot_trunc_streams_per_block

    root = _write_stage_sidecars(_write_stacked_run(tmp_path / "run"))
    fig_dir = tmp_path / "figures"
    fig_dir.mkdir()
    plot_trunc_streams_per_block(_fake_run(root, N_EPOCHS, fig_dir))
    for name in ("trunc_loss_curve_per_block",
                 "query_trunc_loss_curve_per_block",
                 "cvqnn_trunc_loss_curve_per_block"):
        assert (fig_dir / f"{name}.png").is_file()


def test_trunc_streams_per_block_skips_without_sidecars(tmp_path):
    from experiments.report_diagnostics import plot_trunc_streams_per_block

    root = _write_stacked_run(tmp_path / "run")
    fig_dir = tmp_path / "figures"
    fig_dir.mkdir()
    plot_trunc_streams_per_block(_fake_run(root, N_EPOCHS, fig_dir))
    assert not list(fig_dir.glob("*_per_block.png"))


def test_trunc_streams_per_block_skips_an_all_zero_stream(tmp_path):
    """The convention the query / W trunc curves already use for L_W = 0."""
    from experiments.report_diagnostics import plot_trunc_streams_per_block

    root = _write_stage_sidecars(_write_stacked_run(tmp_path / "run"),
                                 trunc=(0.4, 0.0, 0.01))
    fig_dir = tmp_path / "figures"
    fig_dir.mkdir()
    plot_trunc_streams_per_block(_fake_run(root, N_EPOCHS, fig_dir))
    assert (fig_dir / "trunc_loss_curve_per_block.png").is_file()
    assert not (fig_dir / "query_trunc_loss_curve_per_block.png").exists()


def test_per_block_trunc_mean_reproduces_the_recorded_aggregate(tmp_path):
    """`StackedCVQuixer.forward` reduces each stream to a flat mean over
    blocks, so the sidecars must average back to the recorded value. Pins the
    invariant the eval script's self-check relies on."""
    from cv_quixer.evaluation import artefact_schema as schema
    from experiments.report_diagnostics import _diag_stages, _load_stage_trunc

    root = _write_stage_sidecars(_write_stacked_run(tmp_path / "run"))
    diag = _load_npz(root / "diagnostics" / schema.diagnostics_filename(1))
    per_stream = _load_stage_trunc(root, _diag_stages(diag))
    recorded = _fake_run(root, 1, tmp_path)["history"]["epoch"]
    for key, field in (
        (schema.PATCH_TRUNC, "test_trunc_loss"),
        (schema.QUERY_TRUNC, "test_query_trunc_loss"),
        (schema.W_TRUNC, "test_cvqnn_trunc_loss"),
    ):
        by_stage = per_stream[key]
        derived = np.mean([by_stage[p][1] for p in sorted(by_stage)])
        np.testing.assert_allclose(derived, recorded[field][0], rtol=1e-6)


def test_stacked_ratio_never_exceeds_one(tmp_path):
    """‖P(M)|ψ⟩‖ ≤ β for unit-norm input, so the plotted ratio is a
    probability — the property the upper-bound caption rests on.

    Checked against the derivation rather than the figure: β here comes from
    the same coefficients the model would have used, and the raw norms are
    capped at 1 (an exactly-unit-norm output, the best case).
    """
    from cv_quixer.evaluation import artefact_schema as schema
    from experiments.report_diagnostics import (
        _decoder_input_stage,
        _derive_lcu_lambda,
        _success_prob_matrix,
    )

    root = _write_stacked_run(tmp_path / "run")
    run = _fake_run(root, N_EPOCHS, tmp_path)
    prefix, _, _ = _decoder_input_stage(run["diagnostics"])
    raw = _success_prob_matrix(run["predictions"][schema.SUCCESS_PROBS])
    beta = _derive_lcu_lambda(
        run["diagnostics"][f"{prefix}{schema.LCU_COEFFS}"],
        run["diagnostics"][f"{prefix}{schema.POLY_COEFFS}"],
    )
    assert (beta >= 1.0).all()          # β ≥ |c_0|·α⁰ whenever the coeffs are O(1)
    assert (raw / beta[None, :] ** 2 <= 1.0).all()

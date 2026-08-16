"""Thesis-ready per-run figures for a trained CV-Quixer run.

`report_diagnostics.py` is the project's figure generator and stays that way —
it is a *diagnostic* tool, and its labels say so: raw npz keys as legend entries
(`head0`, `block0_head0`, `c_0`), plain-text maths (`|b_i|`), unexpanded
acronyms, ten-entry legends inside the axes, and a per-head panel row that comes
out 35 inches wide at the ten heads these runs use. This script is a thin
presentation layer over it: it imports that module's loaders and metric helpers
and reimplements none of them, owning only the styling.

A curated subset — the figures that earn a place in a thesis, not the per-batch
debugging curves:

    loss_curve                             accuracy_curve
    truncation_losses                      confusion_matrix
    per_class_accuracy_curve               calibration_reliability
    photon_number_per_mode                 state_norm_histogram
    success_prob_histogram                 success_prob_trajectory
    lcu_coefficients_heatmap               polynomial_coefficients_trajectory

Output lands in `<run>/figures/thesis/` as PNG (review) + PDF (LaTeX), leaving
the existing `figures/*.png` untouched.

Most figures read `predictions/` or `diagnostics/` npz, which a `light` artefact
pull leaves on the cluster (see scripts/pull_results.sh). Missing inputs are
reported as skips, never failures, so the same command is useful on a laptop
holding only `history.json` and on the cluster holding everything.

Usage:
    # every run of >=25 epochs across the three model-variant sweeps
    uv run python experiments/thesis_run_figures.py \\
        --sweep-dir results/sweeps/high_epoch_quantum_<ts>/ \\
        --sweep-dir results/sweeps/high_epoch_shared_<ts>/ \\
        --sweep-dir results/sweeps/high_epoch_stacked_<ts>/ \\
        --min-epochs 25

    # one run, one figure
    uv run python experiments/thesis_run_figures.py \\
        --run-dir results/runs/full_fashionmnist_<ts>/ --only loss_curve
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import sys
import traceback
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _run_selection import read_run_names_file
from _thesis_style import (
    FIGSIZE,
    MODEL_COLORS,
    grid_axes,
    head_colors,
    outside_legend,
    save,
    show_path,
    style_axes,
    titles,
)

# Reuse the run loader, the per-epoch artefact readers, the stage detection and
# every metric derivation. This script adds no data handling of its own.
from report_diagnostics import (
    _SUCCESS_PROB_FLOOR,
    MissingArtefactError,
    _accuracy_from,
    _cross_entropy_from,
    _derive_lcu_lambda,
    _diag_stages,
    _load_all_diagnostics,
    _per_class_acc_from,
    load_run,
)
from report_diagnostics import _require_predictions as _load_predictions_npz

from cv_quixer.evaluation import artefact_schema as schema
from cv_quixer.evaluation.labels import class_names


class SkipFigure(RuntimeError):
    """This figure does not apply to this run (missing input, or N/A)."""


# ---------------------------------------------------------------------------
# Display-name layer
# ---------------------------------------------------------------------------


def _run_subtitle(run: dict) -> str:
    """One grey line identifying the run a figure belongs to.

    Sixteen runs' figures end up in one document, so each must say which
    architecture it came from without relying on the surrounding caption.
    """
    q = run["config"].quantum
    meta = run["history"].get("meta", {})
    model = str(meta.get("model") or "quantum")
    params = meta.get("achieved_params") or meta.get("n_params")
    bits = [
        model,
        f"{q.num_heads} heads",
        f"{q.num_modes} modes",
        f"cutoff {q.cutoff_dim}",
        f"degree {q.poly_degree}",
    ]
    if getattr(q, "num_seq2seq_blocks", 1) and model == "quantum_stacked":
        bits.append(f"{q.num_seq2seq_blocks} blocks")
    if params:
        bits.append(f"{int(params) / 1e6:.2f}M parameters")
    bits.append(f"epoch {run['epoch']}")
    return " · ".join(bits)


def _stage_display(prefix: str, n_blocks: int) -> str:
    """Human stage name for a diagnostics namespace.

    `_diag_stages` yields 0-indexed keys (`block0_`), while the model and the
    docs describe blocks as 1..n — so display is shifted by one to match the
    architecture the thesis describes.
    """
    if prefix.startswith("block"):
        idx = int(prefix[len("block"):].rstrip("_"))
        return f"block {idx + 1} of {n_blocks}" if n_blocks > 1 else "block 1"
    if prefix.startswith("agg"):
        return "aggregator block"
    return ""


def _stage_suffix_title(prefix: str, n_blocks: int) -> str:
    stage = _stage_display(prefix, n_blocks)
    return f" — {stage}" if stage else ""


def _head_label(key: str) -> str:
    """`head3` / `block0_head3` / `agg_head3` → readable legend text."""
    name = key.replace("_state_norms", "")
    if name.startswith("block"):
        block, _, head = name.partition("_")
        idx = int(block[len("block"):])
        return f"Block {idx + 1}, head {head[len('head'):]}"
    if name.startswith("agg_"):
        return f"Aggregator, head {name.split('head')[-1]}"
    if name.startswith("head"):
        return f"Head {name[len('head'):]}"
    return name


def _n_blocks(run: dict) -> int:
    return int(getattr(run["config"].quantum, "num_seq2seq_blocks", 1) or 1)


# ---------------------------------------------------------------------------
# Epoch series: npz-derived where possible, training log otherwise
# ---------------------------------------------------------------------------


def _stream_epochs(run_dir: Path, side: str, n_epochs: int, reduce_fns: dict):
    """Apply scalar reductions to each epoch's predictions npz, one at a time.

    Deliberately not `_load_all_per_epoch_predictions`: that returns every
    epoch's arrays at once, and the *train*-side files are the ~94 MB/epoch
    ones, so a 25-epoch run would hold ~2.4 GB live purely to compute a handful
    of scalars. Reading and discarding epoch by epoch keeps the peak at one
    file. Raises MissingArtefactError from the first absent epoch, exactly as
    the batch loader would.
    """
    out = {name: [] for name in reduce_fns}
    for e in range(1, n_epochs + 1):
        preds = _load_predictions_npz(run_dir, e, side)
        for name, fn in reduce_fns.items():
            out[name].append(fn(preds))
        del preds
    return out


def _epoch_series(run: dict) -> dict:
    """Per-epoch loss/accuracy, derived from the predictions npz when present.

    `report_diagnostics` treats the npz as canonical and the training log as a
    cross-check (CLAUDE.md: `history["epoch"]` is "never canonical for
    figures"). That holds here — but a `light` artefact pull carries only
    `history.json`, so rather than emit nothing this falls back to the logged
    values and records which source was used, surfaced in the figure subtitle.
    """
    eh = run["history"]["epoch"]
    n_epochs = len(eh.get("test_acc") or [])
    reductions = {"loss": _cross_entropy_from, "acc": _accuracy_from}
    try:
        test = _stream_epochs(run["run_dir"], "test", n_epochs, reductions)
        train = _stream_epochs(run["run_dir"], "train", n_epochs, reductions)
    except MissingArtefactError:
        return {
            "epochs": list(range(1, n_epochs + 1)),
            "train_loss": list(eh.get("train_loss") or []),
            "test_loss": list(eh.get("test_loss") or []),
            "train_acc": list(eh.get("train_acc") or []),
            "test_acc": list(eh.get("test_acc") or []),
            "derived": False,
        }
    return {
        "epochs": list(range(1, n_epochs + 1)),
        "train_loss": train["loss"],
        "test_loss": test["loss"],
        "train_acc": train["acc"],
        "test_acc": test["acc"],
        "derived": True,
    }


def _source_note(series: dict) -> str:
    return "" if series["derived"] else " · values from the training log"


def _best_epoch(run: dict) -> int | None:
    return run["history"].get("meta", {}).get("best_epoch")


# ---------------------------------------------------------------------------
# Figures: training dynamics
# ---------------------------------------------------------------------------


def fig_loss_curve(run: dict) -> None:
    s = _epoch_series(run)
    if not s["epochs"]:
        raise SkipFigure("history.json has no epochs")
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(s["epochs"], s["train_loss"], marker="o", markersize=3.5,
            color="#0072B2", label="Training")
    ax.plot(s["epochs"], s["test_loss"], marker="s", markersize=3.5,
            color="#D55E00", label="Test")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    titles(fig, ax, "Training and test loss",
           _run_subtitle(run) + _source_note(s))
    style_axes(ax)
    ax.legend(frameon=False)
    save(fig, run["fig_dir"], "loss_curve")


def fig_accuracy_curve(run: dict) -> None:
    s = _epoch_series(run)
    if not s["epochs"]:
        raise SkipFigure("history.json has no epochs")
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(s["epochs"], s["train_acc"], marker="o", markersize=3.5,
            color="#0072B2", label="Training")
    ax.plot(s["epochs"], s["test_acc"], marker="s", markersize=3.5,
            color="#D55E00", label="Test")
    best = _best_epoch(run)
    if best and best in s["epochs"]:
        acc = s["test_acc"][s["epochs"].index(best)]
        ax.axvline(best, color="#888888", ls="--", lw=0.9)
        ax.annotate(f"best: {acc:.4f} (epoch {best})", (best, acc),
                    textcoords="offset points", xytext=(-8, 10),
                    ha="right", fontsize=8, color="#444444")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    titles(fig, ax, "Training and test accuracy",
           _run_subtitle(run) + _source_note(s))
    style_axes(ax)
    ax.legend(frameon=False, loc="lower right")
    save(fig, run["fig_dir"], "accuracy_curve")


# The three truncation streams, as (train key, test key, panel title). Each
# measures probability leaking out of the truncated Fock space at a different
# stage, so they belong in one figure rather than three near-identical ones.
_TRUNC_STREAMS = [
    ("trunc_loss", "test_trunc_loss",
     "Per-patch unitaries $U_i$"),
    ("cvqnn_trunc_loss", "test_cvqnn_trunc_loss",
     "Fixed CV quantum circuit $W$"),
    ("query_trunc_loss", "test_query_trunc_loss",
     "Query unitaries $U_{q,i}$"),
]


def fig_truncation_losses(run: dict) -> None:
    """All truncation-leakage streams that this run actually exercises.

    A stream that is identically zero means the stage is absent (no CVQNN block
    when `cvqnn_num_layers=0`; no query unitaries outside the stacked model), so
    it is dropped rather than drawn as a flat zero line.
    """
    eh = run["history"]["epoch"]
    active = []
    for train_key, test_key, label in _TRUNC_STREAMS:
        tr = list(eh.get(train_key) or [])
        te = list(eh.get(test_key) or [])
        if (tr and np.any(np.asarray(tr) != 0)) or (
                te and np.any(np.asarray(te) != 0)):
            active.append((tr, te, label))
    if not active:
        raise SkipFigure("no non-zero truncation streams in history.json")

    fig, axes = plt.subplots(
        len(active), 1, figsize=(7.5, 2.4 * len(active) + 1.2),
        sharex=True, squeeze=False,
    )
    for ax, (tr, te, label) in zip(axes[:, 0], active):
        x = range(1, max(len(tr), len(te)) + 1)
        if tr:
            ax.plot(list(x)[:len(tr)], tr, marker="o", markersize=3,
                    color="#0072B2", label="Training")
        if te:
            ax.plot(list(x)[:len(te)], te, marker="s", markersize=3,
                    color="#D55E00", label="Test")
        ax.set_ylabel("Leaked probability")
        ax.set_title(label, fontsize=10, pad=4)
        style_axes(ax)
    axes[-1, 0].set_xlabel("Epoch")
    axes[0, 0].legend(frameon=False, fontsize=8)
    titles(fig, None, "Fock-space truncation loss by stage", _run_subtitle(run))
    save(fig, run["fig_dir"], "truncation_losses", rect=(0, 0, 1, 0.9))


# ---------------------------------------------------------------------------
# Figures: classification performance
# ---------------------------------------------------------------------------


def _require_predictions(run: dict) -> dict:
    if run["predictions"] is None:
        raise SkipFigure(
            f"predictions/epoch_{run['epoch']:04d}.npz not present "
            "(pull --tier excl_train_ckpt, or render on the cluster)"
        )
    return run["predictions"]


def fig_confusion_matrix(run: dict) -> None:
    preds = _require_predictions(run)
    classes = class_names(run["config"])
    cm = np.zeros((len(classes), len(classes)))
    y_true = preds["y_true"].astype(np.int64)
    y_pred = preds["y_pred"].astype(np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    row = cm.sum(axis=1, keepdims=True)
    norm = cm / np.maximum(row, 1)

    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(classes, fontsize=8)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    for i in range(len(classes)):
        for j in range(len(classes)):
            if norm[i, j] >= 0.005:
                ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center",
                        fontsize=7,
                        color="white" if norm[i, j] > 0.6 else "black")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Fraction of true class", fontsize=9)
    titles(fig, ax, "Test confusion matrix (row-normalised)", _run_subtitle(run))
    ax.grid(False)
    save(fig, run["fig_dir"], "confusion_matrix", rect=(0, 0, 1, 0.93))


def fig_per_class_accuracy_curve(run: dict) -> None:
    classes = class_names(run["config"])
    n_epochs = len(run["history"]["epoch"].get("test_acc") or [])
    try:
        streamed = _stream_epochs(
            run["run_dir"], "test", n_epochs,
            {"per_class": lambda p: _per_class_acc_from(p, len(classes))},
        )
    except MissingArtefactError as exc:
        raise SkipFigure(str(exc).split(" — ")[0]) from exc
    epochs = list(range(1, n_epochs + 1))
    acc = np.stack(streamed["per_class"])
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    cmap = plt.get_cmap("tab10")
    for c, name in enumerate(classes):
        ax.plot(epochs, acc[:, c], marker="o", markersize=3,
                color=cmap(c % 10), label=name)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test accuracy (recall)")
    titles(fig, ax, "Per-class test accuracy", _run_subtitle(run))
    style_axes(ax)
    outside_legend(ax, title="Class")
    save(fig, run["fig_dir"], "per_class_accuracy_curve")


def fig_calibration_reliability(run: dict) -> None:
    preds = _require_predictions(run)
    y_true = preds["y_true"]
    y_probs = preds["y_probs"]
    conf = y_probs.max(axis=-1)
    correct = (y_probs.argmax(axis=-1) == y_true).astype(np.float64)
    bins = np.linspace(0, 1, 16)
    ids = np.digitize(conf, bins) - 1
    xs, ys, ns = [], [], []
    for b in range(len(bins) - 1):
        mask = ids == b
        xs.append(conf[mask].mean() if mask.any()
                  else (bins[b] + bins[b + 1]) / 2)
        ys.append(correct[mask].mean() if mask.any() else np.nan)
        ns.append(int(mask.sum()))

    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    ax.plot([0, 1], [0, 1], ls="--", color="#888888", lw=1,
            label="Perfect calibration")
    ax.plot(xs, ys, marker="o", markersize=4, color="#0072B2",
            label="This model")
    for x, y, n in zip(xs, ys, ns):
        if n > 0 and not np.isnan(y):
            ax.annotate(f"{n:,}", (x, y), textcoords="offset points",
                        xytext=(3, 3), fontsize=6, color="#666666")
    ax.set_xlabel("Predicted confidence (maximum softmax probability)")
    ax.set_ylabel("Observed accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    titles(fig, ax, "Calibration reliability",
           _run_subtitle(run) + " · annotations = samples per bin")
    style_axes(ax)
    ax.legend(frameon=False, loc="upper left")
    save(fig, run["fig_dir"], "calibration_reliability")


# ---------------------------------------------------------------------------
# Figures: quantum diagnostics
# ---------------------------------------------------------------------------


def _require_diagnostics(run: dict) -> dict:
    if run["diagnostics"] is None:
        raise SkipFigure(
            f"diagnostics/epoch_{run['epoch']:04d}.npz not present "
            "(pull --tier excl_train_ckpt, or render on the cluster)"
        )
    return run["diagnostics"]


def fig_photon_number_per_mode(run: dict) -> None:
    diag = _require_diagnostics(run)
    if "mean_photon_number" not in diag:
        raise SkipFigure("diagnostics npz has no mean_photon_number")
    arr = np.asarray(diag["mean_photon_number"])      # (heads, modes)
    num_heads, num_modes = arr.shape
    cutoff = run["config"].quantum.cutoff_dim
    colors = head_colors(num_heads)

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    x = np.arange(num_modes)
    width = 0.8 / num_heads
    for h in range(num_heads):
        ax.bar(x + h * width, arr[h], width=width, color=colors[h],
               label=f"Head {h}")
    ax.set_xticks(x + width * (num_heads - 1) / 2)
    ax.set_xticklabels([f"Mode {k}" for k in range(num_modes)])
    ax.set_ylabel(r"Mean photon number $\langle \hat n_k \rangle$")
    ax.axhline(cutoff - 1, color="#C1272D", ls="--", lw=1,
               label=f"Truncation limit ({cutoff - 1})")
    titles(fig, ax, "Mean photon number per mode", _run_subtitle(run))
    style_axes(ax)
    outside_legend(ax)
    save(fig, run["fig_dir"], "photon_number_per_mode")


def fig_state_norm_histogram(run: dict) -> None:
    diag = _require_diagnostics(run)
    keys = sorted(k for k in diag if k.endswith("_state_norms"))
    if not keys:
        raise SkipFigure("diagnostics npz has no state-norm arrays")
    vals = np.concatenate([np.asarray(diag[k]).ravel() for k in keys])
    spread = float(vals.max() - vals.min())
    # A near point-mass at 1.0 (before truncation has built up) would otherwise
    # get 40 bins spread across sub-float32 noise.
    edges = np.linspace(0.999, 1.001, 21) if spread < 1e-4 else 40
    colors = head_colors(len(keys))

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    for k, color in zip(keys, colors):
        # Step outlines, not filled bars: up to ten overlapping filled
        # histograms hide each other regardless of alpha.
        ax.hist(np.asarray(diag[k]).ravel(), bins=edges, histtype="step",
                linewidth=1.4, color=color, label=_head_label(k))
    ax.axvline(1.0, color="#C1272D", ls="--", lw=1, label="Unit norm")
    ax.set_xlabel(r"Output state norm $\|\psi\|^2$")
    ax.set_ylabel("Count")
    titles(fig, ax, "Output state norm across the diagnostic subset",
           _run_subtitle(run))
    style_axes(ax)
    outside_legend(ax)
    save(fig, run["fig_dir"], "state_norm_histogram")


def fig_success_prob_histogram(run: dict) -> None:
    preds = _require_predictions(run)
    if schema.SUCCESS_PROBS not in preds:
        raise SkipFigure(
            "predictions npz has no success_probs key "
            "(not recorded for this model variant)"
        )
    diag = _require_diagnostics(run)
    if schema.LCU_COEFFS not in diag or schema.POLY_COEFFS not in diag:
        raise SkipFigure("diagnostics npz lacks lcu/poly coeffs (needed for λ)")

    raw = np.asarray(preds[schema.SUCCESS_PROBS], dtype=np.float64)   # (N, H)
    lam = _derive_lcu_lambda(diag[schema.LCU_COEFFS],
                             diag[schema.POLY_COEFFS])                # (H,)
    ratio = raw / lam[None, :] ** 2
    fail = (raw < _SUCCESS_PROB_FLOOR).mean(axis=0)
    pos = ratio[ratio > 0]
    log_x = pos.size > 0 and float(pos.max()) / float(pos.min()) > 100.0
    colors = head_colors(ratio.shape[1])

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    if log_x:
        bins = np.logspace(np.log10(pos.min()), np.log10(pos.max()), 41)
        ax.set_xscale("log")
    else:
        bins = 40
    for h in range(ratio.shape[1]):
        vals = ratio[:, h]
        if log_x:
            vals = vals[vals > 0]
        ax.hist(vals, bins=bins, histtype="step", linewidth=1.4,
                color=colors[h], label=f"Head {h}")
    ax.set_xlabel(
        r"Post-selection success probability "
        r"$\|P(M)|\psi\rangle\|^2 / \lambda^2$"
    )
    ax.set_ylabel("Count")
    # The per-head λ and failure rate would make a ten-entry legend unreadable;
    # they belong in a compact note instead.
    note = (f"Subnormalisation λ: {lam.min():.3g}–{lam.max():.3g}   ·   "
            f"post-selection failures: {fail.min():.2%}–{fail.max():.2%}")
    titles(fig, ax, "LCU/QSVT post-selection success probability",
           _run_subtitle(run) + " · test set")
    ax.annotate(note, xy=(0.5, -0.22), xycoords="axes fraction",
                ha="center", fontsize=8, color="#555555")
    style_axes(ax)
    outside_legend(ax)
    save(fig, run["fig_dir"], "success_prob_histogram",
         rect=(0, 0.05, 1, 0.945))


def fig_success_prob_trajectory(run: dict) -> None:
    # Diagnostics first: they are small (written over the diagnostic subset),
    # and without per-epoch λ there is nothing to plot — no point streaming the
    # much larger predictions files to find that out.
    diag_all = _load_all_diagnostics(run["run_dir"])
    n_epochs = len(run["history"]["epoch"].get("test_acc") or [])
    have_lambda = [
        e for e in range(1, n_epochs + 1)
        if diag_all.get(e) is not None
        and schema.LCU_COEFFS in diag_all[e]
        and schema.POLY_COEFFS in diag_all[e]
    ]
    if len(have_lambda) < 2:
        raise SkipFigure("fewer than 2 epochs have coeffs to derive λ")

    # One predictions file live at a time (see _stream_epochs).
    usable, stats = [], []
    for e in have_lambda:
        path = (run["run_dir"] / "predictions"
                / schema.prediction_filename(e, train=False))
        if not path.is_file():
            continue
        with np.load(path) as npz:
            if schema.SUCCESS_PROBS not in npz:
                continue
            raw = np.asarray(npz[schema.SUCCESS_PROBS], dtype=np.float64)
        lam = _derive_lcu_lambda(diag_all[e][schema.LCU_COEFFS],
                                 diag_all[e][schema.POLY_COEFFS])
        ratio = raw / lam[None, :] ** 2
        stats.append((ratio.mean(axis=0),
                      np.percentile(ratio, 10, axis=0),
                      np.percentile(ratio, 90, axis=0)))
        usable.append(e)
    if len(usable) < 2:
        raise SkipFigure(
            "fewer than 2 epochs carry success_probs "
            "(not recorded for this model variant, or predictions/ absent)"
        )

    mean = np.stack([s[0] for s in stats])
    p10 = np.stack([s[1] for s in stats])
    p90 = np.stack([s[2] for s in stats])
    num_heads = mean.shape[1]
    colors = head_colors(num_heads)
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    for h in range(num_heads):
        ax.plot(usable, mean[:, h], marker="o", markersize=3,
                color=colors[h], label=f"Head {h}")
        ax.fill_between(usable, p10[:, h], p90[:, h],
                        color=colors[h], alpha=0.15, linewidth=0)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"$\|P(M)|\psi\rangle\|^2 / \lambda^2$")
    titles(fig, ax, "Post-selection success probability across training",
           _run_subtitle(run) + " · mean with 10th–90th percentile band")
    style_axes(ax)
    outside_legend(ax)
    save(fig, run["fig_dir"], "success_prob_trajectory")


def fig_lcu_coefficients_heatmap(run: dict) -> None:
    diag = _require_diagnostics(run)
    stages = _diag_stages(diag)
    n_blocks = _n_blocks(run)
    drawn = 0
    for prefix, suffix, _label in stages:
        key = f"{prefix}{schema.LCU_COEFFS}"
        if key not in diag:
            continue
        arr = np.asarray(diag[key])                       # (heads, patches, 2)
        magnitude = np.sqrt((arr ** 2).sum(axis=-1))
        fig, ax = plt.subplots(figsize=(9.0, 4.2))
        im = ax.imshow(magnitude, aspect="auto", cmap="viridis")
        ax.set_xlabel("Patch index $i$")
        ax.set_ylabel("Attention head")
        ax.set_yticks(range(magnitude.shape[0]))
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(r"Coefficient magnitude $|b_i|$", fontsize=9)
        titles(fig, ax,
               "Learned LCU coefficient magnitudes",
               _run_subtitle(run) + _stage_suffix_title(prefix, n_blocks))
        ax.grid(False)
        save(fig, run["fig_dir"], f"lcu_coefficients_heatmap{suffix}")
        drawn += 1
    if not drawn:
        raise SkipFigure("diagnostics npz has no lcu_coeffs")


def fig_polynomial_coefficients_trajectory(run: dict) -> None:
    n_epochs = len(run["history"]["epoch"].get("test_acc") or [])
    try:
        diag_all = _load_all_diagnostics(run["run_dir"], n_epochs=n_epochs)
    except (MissingArtefactError, FileNotFoundError) as exc:
        raise SkipFigure(f"diagnostics/ not fully present: {exc}") from exc
    if not diag_all:
        raise SkipFigure("no diagnostics npz found")
    stages = _diag_stages(diag_all[min(diag_all)])
    n_blocks = _n_blocks(run)
    epochs = sorted(diag_all)
    drawn = 0
    for prefix, suffix, _label in stages:
        key = f"{prefix}{schema.POLY_COEFFS}"
        if any(key not in diag_all[e] for e in epochs):
            continue
        arr = np.stack([np.asarray(diag_all[e][key]) for e in epochs])
        _, num_heads, degree_plus_1 = arr.shape
        # Wrapped grid, not the single 3.5*num_heads-inch row report_diagnostics
        # draws — that is 35 inches wide at the 10 heads these runs use.
        fig, axes = grid_axes(num_heads, ncols=5)
        cmap = plt.get_cmap("plasma")
        for h, ax in enumerate(axes):
            for j in range(degree_plus_1):
                ax.plot(epochs, arr[:, h, j], marker="o", markersize=2.5,
                        linewidth=1.2,
                        color=cmap(j / max(degree_plus_1 - 1, 1) * 0.85),
                        label=f"$c_{{{j}}}$")
            ax.set_title(f"Head {h}", fontsize=9, pad=3)
            style_axes(ax)
        for ax in axes[-min(5, num_heads):]:
            ax.set_xlabel("Epoch")
        axes[0].set_ylabel("Coefficient value")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=degree_plus_1,
                   frameon=False, fontsize=9,
                   title="Polynomial coefficient", title_fontsize=9)
        titles(fig, None, "Polynomial coefficients across training",
               _run_subtitle(run) + _stage_suffix_title(prefix, n_blocks))
        save(fig, run["fig_dir"],
             f"polynomial_coefficients_trajectory{suffix}",
             rect=(0, 0.09, 1, 0.90))
        drawn += 1
    if not drawn:
        raise SkipFigure("diagnostics npz has no poly_coeffs for every epoch")


# ---------------------------------------------------------------------------
# Registry + driver
# ---------------------------------------------------------------------------

FIGURES = {
    "loss_curve": fig_loss_curve,
    "accuracy_curve": fig_accuracy_curve,
    "truncation_losses": fig_truncation_losses,
    "confusion_matrix": fig_confusion_matrix,
    "per_class_accuracy_curve": fig_per_class_accuracy_curve,
    "calibration_reliability": fig_calibration_reliability,
    "photon_number_per_mode": fig_photon_number_per_mode,
    "state_norm_histogram": fig_state_norm_histogram,
    "success_prob_histogram": fig_success_prob_histogram,
    "success_prob_trajectory": fig_success_prob_trajectory,
    "lcu_coefficients_heatmap": fig_lcu_coefficients_heatmap,
    "polynomial_coefficients_trajectory": fig_polynomial_coefficients_trajectory,
}


def render_run(run_dir: Path, epoch_arg: str, only: list[str]) -> tuple[int, int]:
    """Render the selected figures for one run. Returns (written, skipped)."""
    run = load_run(run_dir, epoch_arg)
    # Own subdirectory so the report_diagnostics suite is never overwritten.
    run["fig_dir"] = run_dir.resolve() / "figures" / "thesis"
    run["fig_dir"].mkdir(parents=True, exist_ok=True)

    written = skipped = 0
    for name in (only or list(FIGURES)):
        try:
            FIGURES[name](run)
            written += 1
        except (SkipFigure, MissingArtefactError) as exc:
            first = str(exc).split(" — ")[0]
            print(f"  - {name}: {first}")
            skipped += 1
        except Exception as exc:  # noqa: BLE001 — one bad figure must not stop the rest
            warnings.warn(
                f"{run_dir.name}/{name} failed: {type(exc).__name__}: {exc}",
                RuntimeWarning, stacklevel=2,
            )
            traceback.print_exc(limit=3)
            skipped += 1
    return written, skipped


def _epochs_of(run_dir: Path) -> int:
    history = run_dir / "history.json"
    if not history.is_file():
        return 0
    try:
        with open(history) as f:
            return len(json.load(f).get("epoch", {}).get("test_acc") or [])
    except (json.JSONDecodeError, OSError):
        return 0


def discover_runs(args: argparse.Namespace) -> list[Path]:
    """Resolve --run-dir / --sweep-dir (+ filters) into a run-dir list."""
    runs: list[Path] = [Path(d) for d in (args.run_dir or [])]
    for sweep in (args.sweep_dir or []):
        sweep = Path(sweep)
        if not sweep.is_dir():
            raise SystemExit(f"sweep dir not found: {sweep}")
        for p in sorted(sweep.iterdir()):
            if p.is_dir() and p.name not in ("figures", "subsets"):
                runs.append(p)

    names = read_run_names_file(Path(args.runs_file)) if args.runs_file else None
    selected = []
    for r in runs:
        if not (r / "config.json").is_file():
            continue
        if names is not None and r.name not in names:
            continue
        if args.runs and not any(
                fnmatch.fnmatch(r.name, pat) for pat in args.runs):
            continue
        if args.min_epochs and _epochs_of(r) < args.min_epochs:
            continue
        selected.append(r)

    # Round-robin stripe, not contiguous blocks: runs differ several-fold in
    # cost (heads, modes, blocks all vary), and a sweep dir lists them in an
    # order that groups similar architectures together — so contiguous slicing
    # would hand one task all the expensive ones.
    if args.num_shards > 1:
        selected = selected[args.shard::args.num_shards]
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render the thesis-ready per-run figure suite.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-dir", action="append", help="a single run dir (repeatable)")
    parser.add_argument("--sweep-dir", action="append",
                        help="render every run under this sweep (repeatable)")
    parser.add_argument("--runs", action="append",
                        help="fnmatch pattern on run name (repeatable)")
    parser.add_argument("--runs-file", help="file of run names, one per line")
    parser.add_argument("--min-epochs", type=int, default=0,
                        help="skip runs with fewer completed epochs")
    parser.add_argument("--epoch", default="best",
                        help="best | final | <N> — which epoch's artefacts to use")
    parser.add_argument("--only", action="append", choices=list(FIGURES),
                        help="render just this figure (repeatable)")
    parser.add_argument("--num-shards", type=int, default=1,
                        help="split the selected runs across this many jobs")
    parser.add_argument("--shard", type=int, default=0,
                        help="0-based index of this shard (see --num-shards)")
    args = parser.parse_args()

    if not args.run_dir and not args.sweep_dir:
        parser.error("need at least one --run-dir or --sweep-dir")
    if args.num_shards < 1:
        parser.error("--num-shards must be >= 1")
    if not (0 <= args.shard < args.num_shards):
        parser.error(f"--shard must be in [0, {args.num_shards - 1}]")

    runs = discover_runs(args)
    if not runs:
        # An empty shard is normal when tasks outnumber runs, not an error.
        if args.num_shards > 1:
            print(f"shard {args.shard}/{args.num_shards}: no runs — nothing to do")
            return
        raise SystemExit("no runs matched the selection")

    shard_note = (f" [shard {args.shard + 1}/{args.num_shards}]"
                  if args.num_shards > 1 else "")
    print(f"Rendering thesis figures for {len(runs)} run(s), "
          f"epoch={args.epoch}{shard_note}\n")
    total_w = total_s = 0
    failed: list[str] = []
    for i, run_dir in enumerate(runs, 1):
        print(f"[{i}/{len(runs)}] {show_path(run_dir)}")
        try:
            w, s = render_run(run_dir, args.epoch, args.only)
        except Exception as exc:  # noqa: BLE001 — keep going through the batch
            print(f"  ✗ {type(exc).__name__}: {exc}")
            failed.append(run_dir.name)
            continue
        total_w += w
        total_s += s
    print(f"\n{total_w} figure(s) written, {total_s} skipped, "
          f"{len(failed)} run(s) failed.")
    if failed:
        for name in failed:
            print(f"  ✗ {name}")
        sys.exit(1)


if __name__ == "__main__":
    main()

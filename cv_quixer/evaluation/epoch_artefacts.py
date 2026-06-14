"""Build and persist the per-epoch **Epoch artefacts** bundle (CONTEXT.md).

This is the deep, torch-using half of the Epoch-artefacts seam. It owns the one
thing the live writers (`experiments/full_experiment.py`,
`experiments/eval_cutoff_sweep.py`) used to duplicate and had already drifted on:

  * the model-variant coefficient-snapshot dispatch (canonical lcu/poly +
    optional ``cvqnn_params`` vs. the stacked block-prefixed keys, ADR-0003),
  * the ``quantum_diagnostics`` pass with its swallow-and-warn degradation, and
  * the npz key set + on-disk layout (via `artefact_schema`).

It deliberately does **not** run ``evaluate`` — that never drifted and stays in
the callers, which want their own per-split timing / progress / peak-mem windows.
``build_epoch_artefacts`` takes the already-computed ``evaluate`` result dicts.

The frozen `experiments/backfill_artefacts.py` is intentionally NOT a consumer
(see memory ``project_backfill_archived``).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cv_quixer.evaluation import artefact_schema as schema
from cv_quixer.evaluation.diagnostics import (
    quantum_diagnostics,
    snapshot_coefficients,
    snapshot_cvqnn_params,
    snapshot_stacked_coefficients,
)


def _prediction_subset(eval_result: dict | None) -> dict | None:
    """The npz subset of an ``evaluate()`` result: the per-sample arrays only
    (metrics like loss/acc are dropped). ``success_probs`` is included iff the
    model produced it (LCU post-selection only)."""
    if eval_result is None:
        return None
    preds = {
        schema.Y_TRUE:  eval_result["y_true"],
        schema.Y_PRED:  eval_result["y_pred"],
        schema.Y_PROBS: eval_result["y_probs"],
        schema.READOUTS: eval_result["readouts"],
    }
    if eval_result.get(schema.SUCCESS_PROBS) is not None:
        preds[schema.SUCCESS_PROBS] = eval_result[schema.SUCCESS_PROBS]
    return preds


@dataclass
class EpochArtefacts:
    """One epoch's raw output payload: the full ``evaluate()`` result dicts plus
    the merged diagnostics dict. ``test_preds`` / ``train_preds`` are the npz
    subsets; ``write`` persists them under the standard run-dir layout."""

    test_eval: dict
    train_eval: dict | None
    diagnostics: dict

    @property
    def test_preds(self) -> dict:
        return _prediction_subset(self.test_eval)

    @property
    def train_preds(self) -> dict | None:
        return _prediction_subset(self.train_eval)

    def write(self, run_dir, epoch: int) -> None:
        """Persist `predictions/epoch_NNNN[_train].npz` +
        `diagnostics/epoch_NNNN.npz` under ``run_dir`` (filenames from
        `artefact_schema`). Creates the subdirs if absent."""
        run_dir = Path(run_dir)
        preds_dir = run_dir / "predictions"
        diag_dir = run_dir / "diagnostics"
        preds_dir.mkdir(parents=True, exist_ok=True)
        diag_dir.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            preds_dir / schema.prediction_filename(epoch, train=False),
            **self.test_preds,
        )
        if self.train_preds is not None:
            np.savez_compressed(
                preds_dir / schema.prediction_filename(epoch, train=True),
                **self.train_preds,
            )
        np.savez_compressed(
            diag_dir / schema.diagnostics_filename(epoch),
            **self.diagnostics,
        )


def _coefficient_snapshots(model) -> dict[str, np.ndarray]:
    """Model-variant coeff-snapshot dispatch. Stacked models (ADR-0003) get the
    block-prefixed key set; canonical models get flat lcu/poly plus
    ``cvqnn_params`` when the W block is on. Detection is model-driven (the same
    ``getattr(model, "blocks", ...)`` test ``quantum_diagnostics`` uses)."""
    if getattr(model, "blocks", None) is not None:
        return snapshot_stacked_coefficients(model)
    lcu, poly = snapshot_coefficients(model)
    snaps: dict[str, np.ndarray] = {
        schema.LCU_COEFFS: lcu,
        schema.POLY_COEFFS: poly,
    }
    cvqnn = snapshot_cvqnn_params(model)   # None when cvqnn_num_layers == 0
    if cvqnn is not None:
        snaps[schema.CVQNN_PARAMS] = cvqnn
    return snaps


def build_epoch_artefacts(model, device, *, test_eval, train_eval=None,
                          diag_loader=None,
                          diag_progress: bool | str = "diagnostics"
                          ) -> EpochArtefacts:
    """Assemble one epoch's artefacts from already-computed ``evaluate()`` dicts.

    Runs the coefficient-snapshot dispatch and (when ``diag_loader`` is given)
    the ``quantum_diagnostics`` pass, merging the coeff snapshots into the diag
    dict. A diagnostics-pass *exception* is caught and degraded to coeff-only
    (the established "don't let a diagnostic failure kill a multi-hour run"
    behaviour) — this is NOT a non-finite-value swallow (ADR-0004 unaffected).
    """
    coeff_snaps = _coefficient_snapshots(model)
    diagnostics: dict = dict(coeff_snaps)
    if diag_loader is not None:
        try:
            _, _, diag_raw = quantum_diagnostics(
                model, diag_loader, device, progress=diag_progress,
            )
            diag_raw.update(coeff_snaps)
            diagnostics = diag_raw
        except Exception as e:  # noqa: BLE001 — deliberate degrade, see docstring
            warnings.warn(
                f"quantum_diagnostics failed: {type(e).__name__}: {e}. "
                "Saving only coefficient snapshots for this epoch.",
                RuntimeWarning,
            )
            diagnostics = dict(coeff_snaps)
    return EpochArtefacts(
        test_eval=test_eval, train_eval=train_eval, diagnostics=diagnostics,
    )


# The eval-derived per-epoch history fields, single-sourced so the live writers
# (and eval_cutoff's synth history) agree. Maps history key -> evaluate() key.
_EVAL_HISTORY_FIELDS: tuple[tuple[str, str], ...] = (
    ("test_loss",             "loss"),
    ("test_acc",              "acc"),
    ("test_trunc_loss",       "trunc_loss"),
    ("test_cvqnn_trunc_loss", "cvqnn_trunc_loss"),
    ("test_query_trunc_loss", "query_trunc_loss"),
)


def eval_epoch_metrics(test_eval, train_eval=None) -> dict[str, float]:
    """Project ``evaluate()`` results onto the eval-derived history fields.

    Covers ONLY the eval-derived fields (test loss/acc/trunc/cvqnn-trunc/
    query-trunc, plus train loss/acc when a train eval is given). Train-time
    trunc streams come from the training loop, not ``evaluate()``, and stay the
    caller's own concern. The write mechanism (append vs index vs synth) is also
    the caller's — this only single-sources the field *set*.
    """
    metrics = {hk: float(test_eval[ek]) for hk, ek in _EVAL_HISTORY_FIELDS}
    if train_eval is not None:
        metrics["train_loss"] = float(train_eval["loss"])
        metrics["train_acc"] = float(train_eval["acc"])
    return metrics

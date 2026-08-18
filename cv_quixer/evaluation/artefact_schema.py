"""Single source of the per-epoch artefact npz key/filename contract.

This module is the **torch-free** half of the Epoch-artefacts seam (see
CONTEXT.md "Epoch artefacts"). The writers (`experiments/full_experiment.py`,
`experiments/eval_cutoff_sweep.py`, via `cv_quixer.evaluation.epoch_artefacts`)
and the reader (`experiments/report_diagnostics.py`) both reference the key
names and filenames defined here, so the npz schema lives in exactly one place.

Deliberately imports nothing heavy: `report_diagnostics`'s default path stays
torch-free (it defers `import torch` to its full-inference branch), so this
module must never pull torch in transitively.
"""

from __future__ import annotations

# --- predictions/*.npz keys -------------------------------------------------
Y_TRUE = "y_true"
Y_PRED = "y_pred"
Y_PROBS = "y_probs"
READOUTS = "readouts"
# Optional — present only for models with LCU post-selection (ADR-0002).
SUCCESS_PROBS = "success_probs"

# --- per-stage sidecar npz keys (stacked model, ADR-0003) -------------------
# `predictions/epoch_NNNN.npz` describes the *decoder-input* stage only, because
# StackedCVQuixer.forward discards the earlier blocks' values. A sidecar written
# by `experiments/eval_block_stages.py` carries one seq-to-seq stage's own
# SUCCESS_PROBS plus its three truncation scalars (0-d arrays), recomputed from
# a checkpoint over the same test split. Purely additive: a run without sidecars
# behaves exactly as before.
PATCH_TRUNC = "patch_trunc"
QUERY_TRUNC = "query_trunc"
W_TRUNC = "w_trunc"

#: The truncation streams a stage sidecar carries, in figure order.
STAGE_TRUNC_KEYS: tuple[str, ...] = (PATCH_TRUNC, QUERY_TRUNC, W_TRUNC)

#: The keys every predictions npz (test or train) always carries.
PREDICTION_KEYS_REQUIRED: tuple[str, ...] = (Y_TRUE, Y_PRED, Y_PROBS, READOUTS)

# --- diagnostics/*.npz coefficient keys -------------------------------------
# Canonical models write LCU_COEFFS/POLY_COEFFS (+ CVQNN_PARAMS when W is on);
# the stacked model writes block-prefixed variants (ADR-0003), so these flat
# names are the canonical-model contract only.
LCU_COEFFS = "lcu_coeffs"
POLY_COEFFS = "poly_coeffs"
CVQNN_PARAMS = "cvqnn_params"


def prediction_filename(epoch: int, *, train: bool = False) -> str:
    """`epoch_NNNN.npz` (test) or `epoch_NNNN_train.npz` (train)."""
    suffix = "_train" if train else ""
    return f"epoch_{epoch:04d}{suffix}.npz"


def stage_prediction_filename(epoch: int, stage_prefix: str) -> str:
    """`epoch_NNNN_block0.npz` — one seq-to-seq stage's sidecar (ADR-0003).

    Args:
        epoch:        1-indexed epoch number.
        stage_prefix: a `_diag_stages` key prefix, e.g. ``"block0_"`` or
                      ``"agg_"``. The trailing underscore is optional.
    """
    stage = stage_prefix.rstrip("_")
    if not stage:
        raise ValueError(
            "stage_prediction_filename needs a stage prefix; the canonical "
            "(empty-prefix) namespace lives in prediction_filename()."
        )
    return f"epoch_{epoch:04d}_{stage}.npz"


def diagnostics_filename(epoch: int) -> str:
    """`epoch_NNNN.npz` for the per-epoch diagnostics bundle."""
    return f"epoch_{epoch:04d}.npz"

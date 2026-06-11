"""NaN-forensics instrumentation for training runs.

Always-on, cheap-by-construction debugging support wired into
``experiments/full_experiment.py``:

- per-batch debug stream (per-head trunc losses, per-group grad norms,
  gate-param min-abs / exactly-zero counts, per-head success-prob ranges)
  accumulated in memory and persisted to ``<run>/debug/stream.npz``;
- a one-time forensic dump per new non-finite event (pre-step state_dict,
  optimizer state, the offending batch, all grads) so a single batch replays
  the event offline;
- an immediate re-run of the offending batch under
  ``torch.autograd.set_detect_anomaly(True)`` to name the exact backward node
  that produced the first NaN/Inf.

Motivation: the 2026-06 sweep NaN post-mortem. The beamsplitter/displacement
analytic Fock formulas have NaN-singular *gradients* at exactly-zero gate
parameters (finite forward), and the post-selection renorm guard leaks NaN
backward through ``clamp(NaN)``. This module exists to catch the trigger
in vivo with enough context to prove (or refute) that chain.

Everything here is reusable/unit-testable; the experiment script only wires.
"""

from __future__ import annotations

import json
import math
import re
import traceback
import warnings
from pathlib import Path

import numpy as np
import torch


# Cap on per-run forensic event dumps — each is small (model + optimizer +
# one batch), but a pathological run could otherwise dump every batch.
MAX_EVENT_DUMPS = 8


# ---------------------------------------------------------------------------
# Init fingerprint
# ---------------------------------------------------------------------------


def init_fingerprint(model: torch.nn.Module) -> dict:
    """JSON-friendly fingerprint of every parameter tensor.

    Captures shape, float64 sum, and the first four values of each named
    parameter. Purpose: compare *initialisations* across runs/architectures
    (e.g. whether two sweep runs share per-head init draws), so it should be
    written once at model build, before any optimizer step.
    """
    fp: dict[str, dict] = {}
    with torch.no_grad():
        for name, p in model.named_parameters():
            t = p.detach().cpu()
            if t.is_complex():
                t = torch.view_as_real(t)
            t64 = t.double().reshape(-1)
            fp[name] = {
                "shape": list(p.shape),
                "sum": float(t64.sum()),
                "first": [float(v) for v in t64[:4]],
            }
    return fp


# ---------------------------------------------------------------------------
# Gradient groups
# ---------------------------------------------------------------------------

_HEAD_PARAM_RE = re.compile(r"(?:^|\.)heads\.(\d+)\.([A-Za-z_]+)")

# Canonical bucket names for the per-head submodules. Anything unrecognised
# keeps its raw attribute name, so new submodules are never silently merged.
_COMPONENT_BUCKETS = {
    "hypernetwork": "hypernet",   # CNN head (CVQuixer)
    "linear": "hypernet",         # LinearCVHead (SharedCVQuixer) — same role
    "lcu_coeffs": "lcu",
    "poly_coeffs": "poly",
    "cvqnn_params": "cvqnn",
}


def build_grad_groups(
    model: torch.nn.Module,
) -> dict[str, list[torch.nn.Parameter]]:
    """Partition ``model.named_parameters()`` into named gradient groups.

    Per-head groups (``head{h}/hypernet``, ``head{h}/lcu``, ``head{h}/poly``,
    ``head{h}/cvqnn``) separate the streams the NaN post-mortem showed to
    behave differently (hypernet dies first; lcu/poly one step later via the
    renorm clamp leak; cvqnn never). Everything else (decoder, shared CNN,
    stacked-model blocks) falls into prefix buckets so the partition always
    covers every parameter exactly once.
    """
    groups: dict[str, list[torch.nn.Parameter]] = {}
    for name, p in model.named_parameters():
        m = _HEAD_PARAM_RE.search(name)
        if m is not None:
            h, comp = int(m.group(1)), m.group(2)
            bucket = f"head{h}/{_COMPONENT_BUCKETS.get(comp, comp)}"
        elif name.startswith("decoder."):
            bucket = "decoder"
        else:
            # e.g. cv_attention.patch_cnn.* (shared CNN) or stacked-model
            # block params — group by the first two path components.
            bucket = ".".join(name.split(".")[:2])
        groups.setdefault(bucket, []).append(p)
    return groups


def grad_group_norms(
    groups: dict[str, list[torch.nn.Parameter]],
) -> dict[str, float]:
    """L2 norm of the current gradients of each group (NaN-propagating).

    Params with ``grad is None`` contribute nothing; a group with no grads at
    all reports 0.0. Non-finite grads yield a non-finite group norm — that is
    the event-detection signal, so no sanitisation here.
    """
    out: dict[str, float] = {}
    for name, params in groups.items():
        norms = [
            p.grad.detach().norm(2) for p in params if p.grad is not None
        ]
        # One device sync per group (not per param).
        out[name] = (
            float(torch.stack(norms).norm(2).item()) if norms else 0.0
        )
    return out


# ---------------------------------------------------------------------------
# Per-batch debug stream
# ---------------------------------------------------------------------------


class DebugStreamWriter:
    """Accumulates per-batch debug records and persists them as one npz.

    Records are dicts of scalars / 1-D / 2-D numpy arrays with a fixed schema
    across batches (the first record fixes it). ``save()`` rewrites
    ``stream.npz`` + ``stream_meta.json`` wholesale (same pattern as
    history.json: cheap, crash-tolerant at epoch granularity). On resume,
    pass the existing path to ``load()`` to extend rather than overwrite.
    """

    def __init__(self, debug_dir: Path, meta: dict | None = None) -> None:
        self.debug_dir = Path(debug_dir)
        self.path = self.debug_dir / "stream.npz"
        self.meta_path = self.debug_dir / "stream_meta.json"
        self.meta = dict(meta or {})
        self._records: dict[str, list] = {}

    def load(self) -> None:
        """Extend from an existing stream file (resume support)."""
        if self.path.exists():
            with np.load(self.path) as z:
                self._records = {k: list(z[k]) for k in z.files}

    def append(self, record: dict) -> None:
        for key, val in record.items():
            self._records.setdefault(key, []).append(np.asarray(val))

    def __len__(self) -> int:
        return len(next(iter(self._records.values()))) if self._records else 0

    def save(self) -> None:
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        arrays = {k: np.stack(v) for k, v in self._records.items()}
        np.savez_compressed(self.path, **arrays)
        with open(self.meta_path, "w") as f:
            json.dump(self.meta, f, indent=2)


# ---------------------------------------------------------------------------
# Event detection / forensic dump / anomaly replay
# ---------------------------------------------------------------------------


def nonfinite_heads(record: dict, num_heads: int) -> set[int]:
    """Heads whose per-head streams in a debug record are non-finite.

    Checks every 1-D array of length ``num_heads`` plus every grad-group key
    of the form ``head{h}/...`` in the ``grad_groups`` sub-dict.
    """
    bad: set[int] = set()
    for key, val in record.items():
        if key == "grad_groups":
            for gname, gval in val.items():
                m = re.match(r"head(\d+)/", gname)
                if m is not None and not math.isfinite(gval):
                    bad.add(int(m.group(1)))
            continue
        arr = np.asarray(val, dtype=np.float64)
        if arr.ndim >= 1 and arr.shape[0] == num_heads:
            mask = ~np.isfinite(arr.reshape(num_heads, -1)).all(axis=1)
            bad.update(int(h) for h in np.nonzero(mask)[0])
    return bad


def dump_nan_event(
    event_dir: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    patches: torch.Tensor,
    labels: torch.Tensor,
    record: dict,
    *,
    step: int,
    epoch: int,
) -> None:
    """Write the forensic snapshot for one non-finite event.

    Must be called *after* ``loss.backward()`` and *before*
    ``optimizer.step()``: the dumped parameters are then the last finite
    state, and the dumped grads are the first non-finite ones — together
    they replay the event offline in a single batch.
    """
    event_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "epoch": epoch,
            "model_state_dict": {
                k: v.detach().cpu() for k, v in model.state_dict().items()
            },
            "optimizer_state_dict": optimizer.state_dict(),
            "patches": patches.detach().cpu(),
            "labels": labels.detach().cpu(),
            "grads": {
                n: p.grad.detach().cpu()
                for n, p in model.named_parameters()
                if p.grad is not None
            },
        },
        event_dir / "snapshot.pt",
    )
    with open(event_dir / "record.json", "w") as f:
        json.dump(_jsonable(record), f, indent=2)


def anomaly_replay(loss_fn, model: torch.nn.Module, out_path: Path) -> str:
    """Re-run one batch under autograd anomaly detection; write the report.

    ``loss_fn`` must recompute the full training loss for the offending batch
    (closure over the batch tensors). The model parameters are still the
    pre-step (finite) ones, so the replay deterministically reproduces the
    event, and anomaly mode names the exact backward node that produced the
    first NaN (e.g. ``PowBackward0`` from ``beamsplitter_matrix``).

    Leaves ``model``'s grads in an undefined state — the caller must zero and
    recompute them before stepping. Returns the report text. Never raises:
    anomaly mode under torch.func.vmap is not guaranteed supported, so any
    failure of the replay itself is reported rather than propagated.
    """
    lines: list[str] = []
    wlist: list = []
    try:
        model.zero_grad(set_to_none=True)
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter("always")
            with torch.autograd.set_detect_anomaly(True):
                loss = loss_fn()
                loss.backward()
        lines.append(
            "anomaly replay completed WITHOUT raising — the non-finite value "
            "may originate outside this batch's backward (e.g. already-NaN "
            "params), or anomaly mode could not see inside vmap."
        )
    except Exception:
        lines.append(traceback.format_exc())
    # Anomaly mode emits the *forward-call* traceback of the failing op as a
    # warning just before raising — that warning is the payload that names the
    # offending source line, so it must be kept on the exception path too
    # (the 2026-06 post-mortem had to recover it via offline replay).
    for w in wlist:
        lines.append(f"warning: {w.category.__name__}: {w.message}")
    text = "\n".join(lines) + "\n"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text)
    return text


def _jsonable(obj):
    """Best-effort conversion of a debug record to JSON-serialisable types."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, torch.Tensor):
        return _jsonable(obj.detach().cpu().numpy())
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj

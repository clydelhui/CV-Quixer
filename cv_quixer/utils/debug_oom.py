"""OOM-event forensics for training runs.

Always-on, run-dir-local record of a CUDA out-of-memory death. The sibling of
the NaN forensics in ``cv_quixer/utils/debug_nan.py``, but for the *catchable*
failure mode: ``experiments/full_experiment.py`` wraps the per-batch train step
in ``try/except torch.cuda.OutOfMemoryError`` and, on catch, writes
``<run>/debug/oom_event.json`` (and mirrors the same dict into
``history["meta"]["oom_aborted"]``) before **re-raising** the original error.

Why this exists: the 2026-06-12 sweep casualties (runs with no ``history.json``)
were all CUDA OOM on the 40 GB A100-40 at the heavy Fock-sim grid corners
(``num_heads × cutoff^num_modes × poly_degree``). The OOM traceback went to the
SLURM ``.err``, forcing a hunt for the right log. After this, the run dir itself
says "OOM at batch N, peak X MB, corner …".

ADR-0004 (non-finite failures fail loudly): the same philosophy governs here —
the caller logs, then re-raises. It never swallows the OOM, sanitises it, or
continues training. This module only *records*; it never decides to continue.

Scope is the catchable CUDA OOM only. Host-RAM cgroup SIGKILL (untrappable) and
wall-time SIGTERM are deliberately out of scope for now — no observed casualty
died that way on this cluster, and adding a per-batch heartbeat / SIGTERM
handler later is purely additive.

The record is JSON-native by construction (ints / floats / strings), so there is
no tensor coercion and no dependency on ``debug_nan``'s ``_jsonable``.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import torch


# Grid-corner knobs copied from ``history["meta"]`` into the record — the fields
# that multiplicatively drive the Fock-sim working set (see the grid-sweep-oom
# post-mortem: num_modes is the brutal exponent). Already populated in
# full_experiment's history meta, so this is a cheap copy. Missing fields are
# simply omitted (e.g. num_seq2seq_blocks on the non-stacked models).
_CORNER_FIELDS = (
    "num_heads",
    "num_modes",
    "cutoff_dim",
    "poly_degree",
    "num_seq2seq_blocks",
)


def _cuda_mem_mb(device) -> dict:
    """Best-effort CUDA memory stats (MB) + device name.

    Every read is guarded: an allocator OOM leaves the CUDA context intact, so
    these normally succeed — but if the context is unavailable or poisoned the
    field is left ``None`` rather than letting a forensic read raise and mask
    the original OOM. Off-CUDA (CPU/MPS) every read fails the guard and returns
    ``None``, which is exactly what we want in unit tests.
    """
    out: dict[str, object] = {
        "device": None,
        "peak_mem_mb": None,
        "allocated_mb": None,
        "reserved_mb": None,
    }
    try:
        out["device"] = torch.cuda.get_device_name(device)
        out["peak_mem_mb"] = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        out["allocated_mb"] = torch.cuda.memory_allocated(device) / (1024 ** 2)
        out["reserved_mb"] = torch.cuda.memory_reserved(device) / (1024 ** 2)
    except Exception:
        pass
    return out


def build_oom_record(
    exc: BaseException,
    *,
    epoch: int,
    step: int,
    batch_index: int,
    device,
    meta: dict | None = None,
) -> dict:
    """Assemble the JSON-native OOM-event record.

    ``meta`` is the run's ``history["meta"]`` (or any mapping) from which the
    grid corner is read; absent corner fields are omitted. ``str(exc)`` carries
    the one genuinely diagnostic line of a CUDA OOM ("Tried to allocate … GiB"),
    so it is captured verbatim rather than dumping the (generic) traceback.
    """
    meta = meta or {}
    corner = {
        k: int(meta[k])
        for k in _CORNER_FIELDS
        if meta.get(k) is not None
    }
    return {
        "cause": "cuda_oom",
        "aborted_at": datetime.now().isoformat(timespec="seconds"),
        "epoch": int(epoch),
        "step": int(step),
        "batch_index": int(batch_index),
        "error_message": str(exc),
        **_cuda_mem_mb(device),
        "corner": corner,
    }


def dump_oom_event(debug_dir: Path, record: dict) -> Path:
    """Write ``record`` to ``<debug_dir>/oom_event.json``; return the path.

    Mirrors the ``debug/aborted_nan.json`` convention (one small JSON file in
    the run's ``debug/`` dir). The caller is responsible for also mirroring the
    record into ``history["meta"]["oom_aborted"]`` and re-raising.
    """
    debug_dir = Path(debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)
    path = debug_dir / "oom_event.json"
    with open(path, "w") as f:
        json.dump(record, f, indent=2)
    return path

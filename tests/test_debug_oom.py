"""Tests for the OOM-event forensics (cv_quixer.utils.debug_oom).

No GPU required: build_oom_record's CUDA mem reads are guarded to ``None``
off-CUDA, and dump_oom_event is pure IO. Pins the schema that
full_experiment.py writes on a caught ``torch.cuda.OutOfMemoryError`` and
mirrors into ``history["meta"]["oom_aborted"]`` (ADR-0004: log then re-raise).
"""

import json

import torch

from cv_quixer.utils.debug_oom import build_oom_record, dump_oom_event


# A realistic CUDA OOM message (the str carried into the record). On modern
# torch the real exception is torch.cuda.OutOfMemoryError, a RuntimeError
# subclass; the record only ever stores str(exc), so a plain message stands in.
_OOM_MSG = (
    "CUDA out of memory. Tried to allocate 2.50 GiB (GPU 0; 39.50 GiB total "
    "capacity; 38.10 GiB already allocated; 0 bytes free)"
)


def test_oom_error_class_exists():
    # The except clause in full_experiment references this symbol on every
    # platform (CPU/MPS included); it must exist even without a CUDA build.
    assert issubclass(torch.cuda.OutOfMemoryError, RuntimeError)


def test_build_oom_record_schema():
    meta = {
        "num_heads": 15,
        "num_modes": 4,
        "cutoff_dim": 6,
        "poly_degree": 3,
        "num_seq2seq_blocks": 1,
        "model": "quantum",  # not a corner field — must be ignored
    }
    rec = build_oom_record(
        RuntimeError(_OOM_MSG),
        epoch=1, step=47, batch_index=12, device="cpu", meta=meta,
    )

    assert rec["cause"] == "cuda_oom"
    assert (rec["epoch"], rec["step"], rec["batch_index"]) == (1, 47, 12)
    assert "Tried to allocate" in rec["error_message"]
    assert rec["corner"] == {
        "num_heads": 15, "num_modes": 4, "cutoff_dim": 6,
        "poly_degree": 3, "num_seq2seq_blocks": 1,
    }
    # Memory + device fields are always present (None off-CUDA), so the schema
    # is stable whether or not the read succeeded.
    for k in ("peak_mem_mb", "allocated_mb", "reserved_mb", "device"):
        assert k in rec and rec[k] is None
    assert "aborted_at" in rec

    # JSON-native by construction: round-trips with no custom encoder. This is
    # also exactly the dict that becomes history["meta"]["oom_aborted"].
    assert json.loads(json.dumps(rec)) == rec


def test_corner_omits_missing_and_none_fields():
    # Non-stacked models have no num_seq2seq_blocks; a None value is skipped too.
    rec = build_oom_record(
        RuntimeError("oom"),
        epoch=0, step=0, batch_index=0, device="cpu",
        meta={"num_heads": 4, "num_modes": 2, "num_seq2seq_blocks": None},
    )
    assert rec["corner"] == {"num_heads": 4, "num_modes": 2}


def test_build_oom_record_tolerates_empty_meta():
    rec = build_oom_record(
        RuntimeError("oom"), epoch=2, step=3, batch_index=0, device="cpu",
    )
    assert rec["corner"] == {}


def test_dump_oom_event_writes_file(tmp_path):
    debug_dir = tmp_path / "run" / "debug"  # parents do not exist yet
    rec = build_oom_record(
        RuntimeError(_OOM_MSG),
        epoch=2, step=3, batch_index=5, device="cpu", meta={"num_heads": 8},
    )
    path = dump_oom_event(debug_dir, rec)

    assert path == debug_dir / "oom_event.json"
    assert path.exists()
    loaded = json.loads(path.read_text())
    assert loaded == rec
    assert loaded["batch_index"] == 5 and loaded["corner"] == {"num_heads": 8}

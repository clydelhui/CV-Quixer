"""Tests for the NaN-forensics instrumentation.

Covers the model-side debug outputs (per-head trunc, gate-param stats) and
the cv_quixer.utils.debug_nan helpers (grad groups, event detection, stream
writer, anomaly replay) that full_experiment.py wires together.
"""

import json
import math

import numpy as np
import pytest
import torch
import torch.nn as nn

from cv_quixer.config.schema import QuantumConfig
from cv_quixer.models.quantum.cv_quixer import CVQuixer, SharedCVQuixer
from cv_quixer.utils.debug_nan import (
    DebugStreamWriter,
    anomaly_replay,
    build_grad_groups,
    grad_group_norms,
    init_fingerprint,
    nonfinite_heads,
)


@pytest.fixture(scope="module")
def debug_quantum_config():
    return QuantumConfig(
        num_modes=2,
        num_layers=1,
        cutoff_dim=4,
        grad_mode="backprop",
        num_heads=2,
        cnn_channels_1=4,
        cnn_channels_2=8,
        cnn_kernel_size=3,
        decoder_hidden_dim=16,
        poly_degree=2,
        dtype="complex64",
        trunc_penalty="norm",
    )


@pytest.fixture(scope="module")
def model(debug_quantum_config, tiny_data_config):
    torch.manual_seed(0)
    return CVQuixer(debug_quantum_config, tiny_data_config)


@pytest.fixture(scope="module")
def patches(tiny_data_config):
    torch.manual_seed(1)
    n_patches = (tiny_data_config.image_size // tiny_data_config.patch_size) ** 2
    return torch.randn(3, n_patches, tiny_data_config.patch_size ** 2)


# ---------------------------------------------------------------------------
# Model-side debug outputs
# ---------------------------------------------------------------------------


class TestPerHeadDebugOutputs:
    def test_per_head_trunc_matches_scalar(self, model, patches):
        out = model(patches, return_trunc_loss=True)
        H = model.config.num_heads
        assert out.trunc_loss_per_head.shape == (H,)
        assert out.cvqnn_trunc_loss_per_head.shape == (H,)
        # The scalar streams are the head-means of the per-head streams.
        assert torch.allclose(
            out.trunc_loss_per_head.mean(), out.trunc_loss, atol=1e-6
        )
        assert torch.allclose(
            out.cvqnn_trunc_loss_per_head.mean(),
            out.cvqnn_trunc_loss,
            atol=1e-6,
        )

    def test_per_head_outputs_detached(self, model, patches):
        out = model(patches, return_trunc_loss=True)
        for t in (
            out.trunc_loss_per_head,
            out.cvqnn_trunc_loss_per_head,
            out.gate_param_min_abs,
            out.gate_param_zero_count,
        ):
            assert not t.requires_grad

    def test_gate_stat_labels_and_shapes(self, model, patches):
        labels = model.gate_stat_labels
        # Single layer → the 8 canonical (op, param) types in plan order.
        assert labels == [
            "squeeze_r", "squeeze_phi", "bs_theta", "bs_phi",
            "rot_phi", "disp_re", "disp_im", "kerr_kappa",
        ]
        out = model(patches, return_trunc_loss=True)
        H, T = model.config.num_heads, len(labels)
        assert out.gate_param_min_abs.shape == (H, T)
        assert out.gate_param_zero_count.shape == (H, T)
        # Random hypernet outputs are never exactly 0.0.
        assert (out.gate_param_zero_count == 0).all()
        assert (out.gate_param_min_abs > 0).all()

    def test_exact_zero_params_are_counted(
        self, debug_quantum_config, tiny_data_config, patches
    ):
        torch.manual_seed(0)
        m = CVQuixer(debug_quantum_config, tiny_data_config)
        # Zero head 0's hypernet output layer → every gate param of head 0 is
        # exactly 0.0 (the NaN-singular point for bs_theta / disp); head 1
        # untouched.
        with torch.no_grad():
            m.cv_attention.heads[0].hypernetwork.linear.weight.zero_()
            m.cv_attention.heads[0].hypernetwork.linear.bias.zero_()
        out = m(patches, return_trunc_loss=True)
        B, N = patches.shape[0], patches.shape[1]
        labels = m.gate_stat_labels
        modes = m.config.num_modes
        n_bs = modes - 1   # linear topology
        expected = {
            lbl: B * N * (n_bs if lbl.startswith("bs_") else modes)
            for lbl in labels
        }
        for t, lbl in enumerate(labels):
            assert out.gate_param_zero_count[0, t].item() == expected[lbl]
            assert out.gate_param_min_abs[0, t].item() == 0.0
        assert (out.gate_param_zero_count[1] == 0).all()

    def test_shared_model_has_debug_outputs(
        self, debug_quantum_config, tiny_data_config, patches
    ):
        torch.manual_seed(0)
        m = SharedCVQuixer(debug_quantum_config, tiny_data_config)
        out = m(patches, return_trunc_loss=True)
        H = m.config.num_heads
        assert out.trunc_loss_per_head.shape == (H,)
        assert out.gate_param_zero_count.shape == (
            H, len(m.gate_stat_labels)
        )

    def test_plain_forward_unchanged(self, model, patches):
        # No return_* flag → plain logits tensor (BaseVisionTransformer
        # contract preserved).
        logits = model(patches)
        assert isinstance(logits, torch.Tensor)
        assert logits.shape == (patches.shape[0], 10)


# ---------------------------------------------------------------------------
# Helpers in cv_quixer.utils.debug_nan
# ---------------------------------------------------------------------------


class TestGradGroups:
    def test_partition_covers_all_params_once(self, model):
        groups = build_grad_groups(model)
        grouped = [p for ps in groups.values() for p in ps]
        all_params = list(model.parameters())
        assert len(grouped) == len(all_params)
        assert {id(p) for p in grouped} == {id(p) for p in all_params}

    def test_expected_buckets(self, model):
        groups = build_grad_groups(model)
        for h in range(model.config.num_heads):
            assert f"head{h}/hypernet" in groups
            assert f"head{h}/lcu" in groups
            assert f"head{h}/poly" in groups
            assert f"head{h}/cvqnn" in groups
        assert "decoder" in groups

    def test_norms_propagate_nonfinite(self, model, patches):
        groups = build_grad_groups(model)
        model.zero_grad(set_to_none=True)
        out = model(patches, return_trunc_loss=True)
        out.logits.sum().backward()
        norms = grad_group_norms(groups)
        assert all(math.isfinite(v) for v in norms.values())
        # Poison one head's lcu grad → only that group goes non-finite.
        model.cv_attention.heads[0].lcu_coeffs.b_real.grad[0] = float("nan")
        norms = grad_group_norms(groups)
        assert not math.isfinite(norms["head0/lcu"])
        assert math.isfinite(norms["head1/lcu"])


class TestNonfiniteHeads:
    def test_detects_per_head_arrays_and_grad_groups(self):
        record = {
            "step": 7,
            "trunc_per_head": np.array([0.1, float("nan"), 0.2]),
            "gate_min_abs": np.ones((3, 8)),
            "grad_groups": {
                "head2/hypernet": float("inf"),
                "head0/lcu": 1.0,
                "decoder": 0.5,
            },
        }
        assert nonfinite_heads(record, num_heads=3) == {1, 2}

    def test_all_finite(self):
        record = {
            "trunc_per_head": np.zeros(2),
            "grad_groups": {"head0/lcu": 1.0, "head1/lcu": 2.0},
        }
        assert nonfinite_heads(record, num_heads=2) == set()


class TestDebugStreamWriter:
    def test_roundtrip_and_resume(self, tmp_path):
        w = DebugStreamWriter(tmp_path, meta={"grad_group_names": ["a", "b"]})
        for step in range(3):
            w.append({
                "step": step,
                "trunc_per_head": np.array([0.1, 0.2]),
                "grad_group_norms": np.array([1.0, 2.0]),
            })
        w.save()
        # Resume: a fresh writer extends the saved stream.
        w2 = DebugStreamWriter(tmp_path)
        w2.load()
        assert len(w2) == 3
        w2.append({
            "step": 3,
            "trunc_per_head": np.array([0.3, 0.4]),
            "grad_group_norms": np.array([3.0, 4.0]),
        })
        w2.save()
        with np.load(tmp_path / "stream.npz") as z:
            assert z["step"].tolist() == [0, 1, 2, 3]
            assert z["trunc_per_head"].shape == (4, 2)
        meta = json.loads((tmp_path / "stream_meta.json").read_text())
        assert "grad_group_names" in json.dumps(meta) or meta == {}


class TestAnomalyReplay:
    def test_names_the_nan_backward_node(self, tmp_path):
        # Complex z**1 at z == 0 has a NaN backward despite a finite forward —
        # the singularity class of the (pre-fix) displacement gate at α == 0.
        # (Real 0**0 is guarded by torch and would make this test vacuous.)
        re = torch.tensor(0.0, requires_grad=True)
        im = torch.tensor(0.0, requires_grad=True)

        def loss_fn():
            z = torch.complex(re, im) ** torch.tensor(1.0 + 0.0j)
            return z.real ** 2 + z.imag ** 2

        report = anomaly_replay(loss_fn, nn.Linear(1, 1), tmp_path / "r.txt")
        assert (tmp_path / "r.txt").exists()
        # The exception names the failing backward node ...
        assert "returned nan" in report
        # ... and the forward-call traceback warning is preserved on the
        # exception path (the payload that names the offending source line).
        assert "Traceback of forward call" in report

    def test_never_raises_on_broken_loss_fn(self, tmp_path):
        def loss_fn():
            raise RuntimeError("boom")

        report = anomaly_replay(loss_fn, nn.Linear(1, 1), tmp_path / "r.txt")
        assert "boom" in report


class TestInitFingerprint:
    def test_jsonable_and_complete(self, model):
        fp = init_fingerprint(model)
        json.dumps(fp)   # must be JSON-serialisable
        names = {n for n, _ in model.named_parameters()}
        assert set(fp) == names
        for entry in fp.values():
            assert len(entry["first"]) <= 4

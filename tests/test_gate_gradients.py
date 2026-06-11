"""Gate-gradient safety regression suite (ADR-0004).

Pins the fix for the 2026-06 sweep NaN post-mortem: the beamsplitter and
displacement Fock-matrix builders had NaN-singular *gradients* at exactly-zero
gate parameters (finite forward), which killed attention heads in one Adam
step. Four layers of protection:

1. ``TestComplexPowBehaviour`` — exhaustively maps torch's complex-pow
   breakage at zero (the constructive guarantee's empirical leg). If a torch
   upgrade changes this, the suite trips instead of a training run dying.
2. ``TestGradientCorrectnessAtZero`` — autograd at the singular points must
   equal a central finite difference (correct, not merely finite).
3. ``TestGateAuditTable`` — every gate factory at its singular candidates:
   forward and gradients finite.
4. ``TestGoldenValues`` — the fix changed no value the model uses: bitwise
   against ``tests/golden/gate_matrices.npz`` (generated from pre-fix code;
   regenerate only deliberately via tests/golden/regenerate_gate_matrices.py).
5. ``TestEndToEndForcedZero`` — the literal kill scenario from the sweep
   (hypernet emitting exactly-zero bs_theta / displacement) now yields finite
   gradients through the full model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from cv_quixer.quantum.gates.gaussian import (
    beamsplitter_matrix,
    displacement_matrix,
    rotation_phases,
    squeezing_matrix,
    two_mode_squeezing_matrix,
)
from cv_quixer.quantum.gates.non_gaussian import cubic_phase_matrix, kerr_phases

D = 6
SIGNED_ZEROS = [(0.0, 0.0), (-0.0, 0.0), (0.0, -0.0), (-0.0, -0.0)]


def _grad_finite(*params: torch.Tensor) -> bool:
    return all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)


def _scalar(v: float) -> torch.Tensor:
    return torch.tensor(v, dtype=torch.float64, requires_grad=True)


# ---------------------------------------------------------------------------
# 1. torch complex-pow behaviour at zero (version tripwire)
# ---------------------------------------------------------------------------


class TestComplexPowBehaviour:
    def test_broken_set_is_exactly_s0_forward_and_s1_grad(self):
        """z ** tensor(s) at all signed zeros: forward NaN iff s==0, grad NaN
        iff s==1. The displacement fix (exponent floor to 2 + closed forms for
        s in {0,1}) is derived from this map — if torch changes it, revisit
        gaussian.py's displacement comment and ADR-0004."""
        fwd_broken, grad_broken = set(), set()
        for s in range(33):
            for zr, zi in SIGNED_ZEROS:
                re, im = _scalar(zr), _scalar(zi)
                out = torch.complex(re, im) ** torch.tensor(
                    float(s), dtype=torch.complex128
                )
                if not torch.isfinite(out.resolve_conj().abs()).item():
                    fwd_broken.add(s)
                    continue
                (out.real ** 2 + out.imag ** 2).backward()
                if not _grad_finite(re, im):
                    grad_broken.add(s)
        assert fwd_broken == {0}
        assert grad_broken == {1}

    def test_real_pow_zero_exponent_guard(self):
        """Real 0**0: forward 1, grad 0 — the guard the beamsplitter clamp
        relies on (padding entries compute 0^0 after the clamp)."""
        x = _scalar(0.0)
        out = x ** torch.tensor(0.0, dtype=torch.float64)
        assert out.item() == 1.0
        out.backward()
        assert x.grad.item() == 0.0

    def test_real_pow_negative_exponent_is_inf(self):
        """The pre-fix Inf source — documents why the clamp exists."""
        x = torch.tensor(0.0, dtype=torch.float64)
        assert torch.isinf(x ** torch.tensor(-3.0, dtype=torch.float64))


# ---------------------------------------------------------------------------
# 2. gradient correctness (not just finiteness) at the singular points
# ---------------------------------------------------------------------------


def _fd_grad(f, at: float, h: float = 1e-6) -> float:
    hi = f(torch.tensor(at + h, dtype=torch.float64))
    lo = f(torch.tensor(at - h, dtype=torch.float64))
    return float((hi - lo) / (2 * h))


class TestGradientCorrectnessAtZero:
    # A generic (asymmetric) linear functional of the matrix, so the derivative
    # at the symmetric point is non-trivially testable.
    @staticmethod
    def _weights(shape) -> torch.Tensor:
        g = torch.Generator().manual_seed(7)
        return torch.complex(
            torch.randn(*shape, dtype=torch.float64, generator=g),
            torch.randn(*shape, dtype=torch.float64, generator=g),
        )

    def test_beamsplitter_theta_zero_matches_finite_difference(self):
        w = self._weights((D, D, D, D))
        phi = torch.tensor(0.3, dtype=torch.float64)

        def f(theta):
            return (beamsplitter_matrix(theta, phi, D) * w).sum().real

        theta = _scalar(0.0)
        f(theta).backward()
        assert torch.isfinite(theta.grad).item()
        assert theta.grad.item() == pytest.approx(_fd_grad(f, 0.0), abs=1e-5)

    def test_displacement_alpha_zero_matches_finite_difference(self):
        w = self._weights((D, D))

        def loss_re(re):
            a = torch.complex(re, torch.zeros((), dtype=torch.float64))
            return (displacement_matrix(a, D) * w).sum().real

        def loss_im(im):
            a = torch.complex(torch.zeros((), dtype=torch.float64), im)
            return (displacement_matrix(a, D) * w).sum().real

        re, im = _scalar(0.0), _scalar(0.0)
        (displacement_matrix(torch.complex(re, im), D) * w).sum().real.backward()
        assert _grad_finite(re, im)
        assert re.grad.item() == pytest.approx(_fd_grad(loss_re, 0.0), abs=1e-5)
        assert im.grad.item() == pytest.approx(_fd_grad(loss_im, 0.0), abs=1e-5)


# ---------------------------------------------------------------------------
# 3. the full gate-audit table: singular candidates -> finite fwd + grads
# ---------------------------------------------------------------------------


def _check_finite(build, *vals):
    ps = [_scalar(v) for v in vals]
    M = build(*ps)
    assert torch.isfinite(M.resolve_conj().abs()).all(), "non-finite forward"
    (M.abs() ** 2).sum().backward()
    assert _grad_finite(*ps), "non-finite gradient"


class TestGateAuditTable:
    def test_rotation(self):
        _check_finite(lambda p: torch.diag(rotation_phases(p, D)), 0.0)

    def test_kerr(self):
        _check_finite(lambda k: torch.diag(kerr_phases(k, D)), 0.0)

    def test_cubic_phase(self):
        _check_finite(lambda g: cubic_phase_matrix(g, D), 0.0)

    def test_two_mode_squeeze(self):
        _check_finite(lambda r, p: two_mode_squeezing_matrix(r, p, D), 0.0, 0.0)

    @pytest.mark.parametrize("r", [0.0, -0.0, 1e-40])
    def test_squeeze(self, r):
        _check_finite(lambda r_, p: squeezing_matrix(r_, p, D), r, 0.3)

    @pytest.mark.parametrize("theta", [0.0, -0.0, 1e-40,
                                       1.5707963267948966, 3.141592653589793])
    @pytest.mark.parametrize("phi", [0.0, 0.2])
    def test_beamsplitter(self, theta, phi):
        _check_finite(lambda t, p: beamsplitter_matrix(t, p, D), theta, phi)

    @pytest.mark.parametrize("re,im", SIGNED_ZEROS + [(0.0, 1e-8), (1e-8, 0.0)])
    def test_displacement(self, re, im):
        re_t, im_t = _scalar(re), _scalar(im)
        M = displacement_matrix(torch.complex(re_t, im_t), D)
        assert torch.isfinite(M.resolve_conj().abs()).all()
        (M.abs() ** 2).sum().backward()
        assert _grad_finite(re_t, im_t)

    def test_displacement_at_zero_is_identity(self):
        M = displacement_matrix(
            torch.complex(torch.tensor(0.0, dtype=torch.float64),
                          torch.tensor(0.0, dtype=torch.float64)), D)
        assert torch.allclose(M, torch.eye(D, dtype=torch.complex128))


# ---------------------------------------------------------------------------
# 4. golden values: the fix changed nothing the model uses
# ---------------------------------------------------------------------------


class TestGoldenValues:
    @pytest.fixture(scope="class")
    def golden(self):
        path = Path(__file__).parent / "golden" / "gate_matrices.npz"
        with np.load(path) as z:
            return {k: z[k] for k in z.files}

    def test_beamsplitter_bitwise(self, golden):
        assert int(golden["cutoff_dim"]) == D
        for (theta, phi), ref in zip(golden["bs_params"], golden["bs"]):
            live = beamsplitter_matrix(
                torch.tensor(theta, dtype=torch.float64),
                torch.tensor(phi, dtype=torch.float64), D,
            ).resolve_conj().numpy()
            assert (live == ref).all(), f"value drift at theta={theta}, phi={phi}"

    def test_displacement_bitwise_except_s1_within_ulps(self, golden):
        m, n = np.meshgrid(np.arange(D), np.arange(D), indexing="ij")
        s1 = np.abs(n - m) == 1
        for (re, im), ref in zip(golden["disp_params"], golden["disp"]):
            live = displacement_matrix(
                torch.complex(torch.tensor(re, dtype=torch.float64),
                              torch.tensor(im, dtype=torch.float64)), D,
            ).resolve_conj().numpy()
            assert (live[~s1] == ref[~s1]).all(), \
                f"value drift outside s==1 at alpha={re}+{im}j"
            # s == 1 entries are now literal ±α (vs exp(log α) before): allow
            # a few ulps. They must still be essentially identical.
            np.testing.assert_allclose(
                live[s1], ref[s1], rtol=1e-13, atol=0.0,
                err_msg=f"s==1 drift beyond ulps at alpha={re}+{im}j",
            )


# ---------------------------------------------------------------------------
# 5. end-to-end: the literal kill scenario, now finite
# ---------------------------------------------------------------------------


class TestEndToEndForcedZero:
    @pytest.fixture()
    def model_and_batch(self, tiny_data_config):
        from cv_quixer.config.schema import QuantumConfig
        from cv_quixer.models.quantum.cv_quixer import CVQuixer

        qc = QuantumConfig(
            num_modes=2, cutoff_dim=4, num_heads=2, cnn_channels_1=4,
            cnn_channels_2=8, decoder_hidden_dim=16, poly_degree=2,
            dtype="complex64", trunc_penalty="norm",
        )
        torch.manual_seed(0)
        model = CVQuixer(qc, tiny_data_config)
        torch.manual_seed(3)
        n_patches = (tiny_data_config.image_size // tiny_data_config.patch_size) ** 2
        patches = torch.randn(2, n_patches, tiny_data_config.patch_size ** 2)
        labels = torch.tensor([1, 5])
        return model, patches, labels

    @pytest.mark.parametrize("zero_types", [
        ["bs_theta"],                # the sweep-killing trigger
        ["disp_re", "disp_im"],      # the displacement twin
        ["bs_theta", "disp_re", "disp_im"],
    ])
    def test_forced_zero_gate_params_give_finite_grads(
        self, model_and_batch, zero_types
    ):
        model, patches, labels = model_and_batch
        h0 = model.cv_attention.heads[0]
        with torch.no_grad():
            for t in zero_types:
                for col in h0._gate_stat_idx[t]:
                    h0.hypernetwork.linear.weight[col].zero_()
                    h0.hypernetwork.linear.bias[col] = 0.0

        out = model(patches, return_trunc_loss=True)
        loss = (F.cross_entropy(out.logits, labels)
                + 0.1 * out.trunc_loss + 0.01 * out.cvqnn_trunc_loss)
        model.zero_grad()
        loss.backward()

        assert torch.isfinite(loss).item()
        # The exactly-zero params must actually be present (trigger armed) ...
        for t in zero_types:
            assert (out.gate_param_zero_count[0, model.gate_stat_labels.index(t)]
                    > 0)
        # ... and every gradient in the model must be finite (pre-fix: head 0's
        # whole hypernetwork went NaN here).
        bad = [n for n, p in model.named_parameters()
               if p.grad is not None and not torch.isfinite(p.grad).all()]
        assert bad == [], f"non-finite grads: {bad}"

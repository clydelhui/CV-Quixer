"""λ-derivation helper in experiments/report_diagnostics.py (numpy-only)."""

import numpy as np


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

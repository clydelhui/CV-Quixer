"""Regenerate the golden gate-matrix snapshot (tests/golden/gate_matrices.npz).

The snapshot freezes the beamsplitter / displacement Fock matrices at a fixed
grid of normal parameter values. tests/test_gate_gradients.py asserts the live
implementations still reproduce it (bitwise for the beamsplitter; bitwise for
displacement except the |m−n| == 1 entries, which the ADR-0004 fix made
*more* accurate — literal ±α instead of exp(log α) — so they get a ≤ few-ulp
tolerance).

Regenerating this file is a deliberate act: it redefines what "unchanged gate
values" means. Only run it when a gate-value change is intended and reviewed.

    uv run python tests/golden/regenerate_gate_matrices.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from cv_quixer.quantum.gates.gaussian import (
    beamsplitter_matrix,
    displacement_matrix,
)

D = 6

# Normal operating values: mixed signs, near-zero (but non-zero), moderate, large.
BS_PARAMS = [
    (-1.3, 0.0), (-0.4, 0.3), (1e-4, -1.2), (0.7, 0.2),
    (1.1, 2.0), (3.0, -0.7), (-2.2, 1.5), (0.05, 0.0),
]
DISP_PARAMS = [
    (0.3, -0.7), (-1.1, 0.4), (1e-5, 2e-5), (0.9, 0.0),
    (0.0, -1.3), (-2.0, -2.0), (0.02, 0.6), (1.6, -0.1),
]


def main() -> None:
    bs = np.stack([
        beamsplitter_matrix(
            torch.tensor(t, dtype=torch.float64),
            torch.tensor(p, dtype=torch.float64), D,
        ).resolve_conj().numpy()
        for t, p in BS_PARAMS
    ])
    disp = np.stack([
        displacement_matrix(
            torch.complex(torch.tensor(re, dtype=torch.float64),
                          torch.tensor(im, dtype=torch.float64)), D,
        ).resolve_conj().numpy()
        for re, im in DISP_PARAMS
    ])
    out = Path(__file__).parent / "gate_matrices.npz"
    np.savez_compressed(
        out,
        cutoff_dim=np.array(D),
        bs_params=np.array(BS_PARAMS),
        bs=bs,
        disp_params=np.array(DISP_PARAMS),
        disp=disp,
    )
    print(f"wrote {out} (bs {bs.shape}, disp {disp.shape})")


if __name__ == "__main__":
    main()

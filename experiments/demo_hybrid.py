"""Hybrid CV-classical model demo.

Shows how the cv_quixer quantum engine integrates with classical PyTorch layers
to form a fully differentiable hybrid model:

    x (B, input_dim)
    ↓  PreMLP  — classical feature extractor
    ↓  CVBlock — CV quantum circuit (this is the pattern for CVLayer.apply())
    ↓  PostMLP — classical classifier
    → logits (B, num_classes)

The CVBlock implements the quantum circuit inline and is immediately runnable
(no stubs required). The circuit structure — encode → transform → measure —
is identical to what you will write inside CVLayer.apply() and
CVAttention._run_single_patch() when implementing the full model.

Run:
    uv run python experiments/demo_hybrid.py
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from cv_quixer.models.quantum.cv_encoding import DisplacementEncoding
from cv_quixer.quantum import (
    CVCircuit,
    FockState,
    displacement_matrix,
    rotation_matrix,
    squeezing_matrix,
)
from cv_quixer.utils import print_parameter_table


# ============================================================
# CV quantum block
# ============================================================

class CVBlock(nn.Module):
    """A single CV quantum processing block.

    Encodes a classical feature vector into bosonic modes via displacement,
    applies a learnable layer of rotations and squeezing, then reads out the
    quadrature expectation values.

    This is a self-contained, runnable version of the circuit you will write
    inside CVLayer.apply() and CVAttention._run_single_patch().

    Parameter layout:
        encoding.scale  — (num_modes,)  learnable displacement scaling
        phi             — (num_modes,)  per-mode rotation angles
        r               — (num_modes,)  per-mode squeezing magnitudes

    Args:
        num_modes:  Number of bosonic modes.
        cutoff_dim: Fock space truncation D. Memory scales as D^num_modes.
    """

    def __init__(self, num_modes: int, cutoff_dim: int) -> None:
        super().__init__()
        self.num_modes = num_modes
        self.cutoff_dim = cutoff_dim

        # Learnable encoding scale — one per mode
        self.encoding = DisplacementEncoding(patch_dim=num_modes, num_modes=num_modes)

        # Learnable rotation angles — one per mode (quantum parameters)
        self.phi = nn.Parameter(torch.zeros(num_modes))

        # Learnable squeezing magnitudes — one per mode (quantum parameters)
        self.r = nn.Parameter(torch.zeros(num_modes))

        # Stateless circuit executor (no nn.Parameters here)
        self.circuit = CVCircuit(num_modes=num_modes, cutoff_dim=cutoff_dim)

    def _run_single_input(self, x: torch.Tensor) -> torch.Tensor:
        """Run the CV circuit on one feature vector and return quadrature readings.

        This is the core circuit implementation. The structure is:
          1. Initialise vacuum state
          2. Encode classical data via displacement gates   ← data encoding
          3. Apply learnable rotation + squeezing gates    ← trainable transform
          4. Measure position quadrature <x̂> per mode     ← readout

        Gradient flows back through every einsum gate application automatically
        (GradMode.BACKPROP).

        When implementing CVLayer.apply() and CVAttention._run_single_patch()
        you will follow exactly this pattern, but with more gates (interferometers,
        Kerr interactions, etc.) and the parameters coming from CVLayer's
        nn.Parameter tensors.

        Args:
            x: 1-D real tensor of shape (num_modes,).

        Returns:
            Real tensor of shape (num_modes,) with <x̂ᵢ> for each mode.
        """
        D = self.cutoff_dim
        device = x.device

        # --- Step 1: vacuum state |0, 0, ..., 0⟩ ---
        state = FockState.vacuum(
            num_modes=self.num_modes,
            cutoff_dim=D,
            device=device,
            dtype=torch.complex128,
        )

        # --- Step 2: displacement encoding ---
        # encoding_alphas returns complex amplitudes αᵢ = scale_i * x_i
        # Applying D(αᵢ) to mode i maps |0⟩ → coherent state |αᵢ⟩,
        # so ⟨x̂ᵢ⟩ = √2 Re(αᵢ) encodes the classical feature value.
        alphas = self.encoding.encoding_alphas(x)
        for i, alpha in enumerate(alphas):
            mat = displacement_matrix(alpha, D).to(device)
            state = self.circuit.apply_single_mode_gate(mat, i, state)

        # --- Step 3: learnable transformation ---
        # Rotation R(φ) rotates in phase space, changing the quadrature basis.
        # Squeezing S(r, 0) reduces variance in x̂ at the cost of p̂ variance.
        # Together they implement a trainable Gaussian transformation per mode.
        # (In the full model this expands to interferometers + full Killoran layer.)
        for i in range(self.num_modes):
            r_mat = rotation_matrix(self.phi[i], D).to(device)
            state = self.circuit.apply_single_mode_gate(r_mat, i, state)

            s_mat = squeezing_matrix(self.r[i], torch.tensor(0.0), D).to(device)
            state = self.circuit.apply_single_mode_gate(s_mat, i, state)

        # --- Step 4: quadrature readout ---
        return torch.stack([
            self.circuit.measure_quadrature_x(i, state)
            for i in range(self.num_modes)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the CV block to a batch of feature vectors.

        Args:
            x: Tensor of shape (batch_size, num_modes).

        Returns:
            Tensor of shape (batch_size, num_modes) with quadrature readings,
            cast back to float32 to match the surrounding classical layers.
        """
        out = torch.stack([self._run_single_input(x[b]) for b in range(x.shape[0])])
        # Quadrature measurements are float64 (real part of complex128 traces).
        # Cast to match the float32 dtype of the surrounding classical MLPs.
        return out.to(x.dtype)


# ============================================================
# Hybrid model
# ============================================================

class HybridCV(nn.Module):
    """Hybrid classical-CV-classical model.

    Architecture:
        PreMLP  : Linear(input_dim → 64) → ReLU → Linear(64 → num_modes)
        CVBlock : displacement encoding + rotations + squeezing + quadrature readout
        PostMLP : Linear(num_modes → 32) → ReLU → Linear(32 → num_classes)

    Args:
        input_dim:  Dimensionality of the raw input features.
        num_modes:  Number of CV bosonic modes (= CVBlock width).
        cutoff_dim: Fock space truncation. Memory scales as cutoff_dim^num_modes.
        num_classes: Number of output classes.
    """

    def __init__(
        self,
        input_dim: int,
        num_modes: int,
        cutoff_dim: int,
        num_classes: int,
    ) -> None:
        super().__init__()

        # Classical pre-processing: reduce input to num_modes features
        self.pre_mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_modes),
        )

        # CV quantum block
        self.cv_block = CVBlock(num_modes=num_modes, cutoff_dim=cutoff_dim)

        # Classical post-processing: map quadrature readings to class logits
        self.post_mlp = nn.Sequential(
            nn.Linear(num_modes, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            Logits tensor of shape (batch_size, num_classes).
        """
        x = self.pre_mlp(x)      # (B, num_modes) — classical features
        x = self.cv_block(x)     # (B, num_modes) — quantum readout
        return self.post_mlp(x)  # (B, num_classes) — logits


# ============================================================
# Demo
# ============================================================

def main() -> None:
    torch.manual_seed(0)

    # Small parameters so the demo runs in < 1 second
    INPUT_DIM = 8
    NUM_MODES = 2
    CUTOFF_DIM = 6
    NUM_CLASSES = 4
    BATCH_SIZE = 4

    print("=" * 60)
    print("  Hybrid CV-classical model demo")
    print("=" * 60)

    # Build the model
    model = HybridCV(
        input_dim=INPUT_DIM,
        num_modes=NUM_MODES,
        cutoff_dim=CUTOFF_DIM,
        num_classes=NUM_CLASSES,
    )

    # --- Parameter breakdown ---
    print_parameter_table(model)

    quantum_params = sum(
        p.numel() for p in model.cv_block.parameters() if p.requires_grad
    )
    classical_params = sum(
        p.numel() for name, p in model.named_parameters()
        if p.requires_grad and not name.startswith("cv_block")
    )
    print(f"\n  Quantum params (CVBlock):   {quantum_params:>8,}")
    print(f"  Classical params (MLPs):    {classical_params:>8,}")

    # --- Forward pass ---
    print("\n" + "─" * 60)
    print("  Forward pass")
    print("─" * 60)
    x = torch.randn(BATCH_SIZE, INPUT_DIM)
    logits = model(x)
    print(f"  Input shape:   {tuple(x.shape)}")
    print(f"  Output shape:  {tuple(logits.shape)}")
    print(f"  Logits (first sample): {logits[0].detach().tolist()}")

    # --- Backward pass: verify gradients flow end-to-end ---
    print("\n" + "─" * 60)
    print("  Backward pass (gradient check)")
    print("─" * 60)
    targets = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,))
    loss = F.cross_entropy(logits, targets)
    loss.backward()

    params_with_grad = [
        (name, p.grad is not None)
        for name, p in model.named_parameters()
        if p.requires_grad
    ]
    all_have_grad = all(has_grad for _, has_grad in params_with_grad)

    for name, has_grad in params_with_grad:
        status = "OK" if has_grad else "MISSING"
        print(f"  {name:<40} grad: {status}")

    print()
    if all_have_grad:
        print("  Gradients flow through the full classical → quantum → classical graph.")
    else:
        missing = [n for n, g in params_with_grad if not g]
        print(f"  WARNING: missing gradients for: {missing}")

    print("=" * 60)


if __name__ == "__main__":
    main()

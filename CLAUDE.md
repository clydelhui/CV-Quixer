# CV-Quixer Codebase Guide

A hypernetwork-driven Continuous Variable (CV) Quantum Vision Transformer benchmarked
against a classical ViT on FashionMNIST. Masters thesis project by Clyde Lhui.

## Project goal

Implement a hypernetwork-driven CV Quantum Vision Transformer (CV-Quixer) and compare
its classification performance on FashionMNIST against an equivalent classical Vision
Transformer. The quantum model uses a pure PyTorch Fock-basis simulation engine (no
PennyLane at inference time) so gradients flow through standard `torch.autograd`.

## Running experiments

```bash
# Train the quantum model
uv run python experiments/train_quantum.py --config configs/cv_quixer.yaml

# Train the classical baseline
uv run python experiments/train_classical.py --config configs/classical_vit.yaml

# Quick ablation without editing YAML
uv run python experiments/train_quantum.py --config configs/cv_quixer.yaml \
    --overrides training.lr=5e-4 quantum.num_modes=4

# Compare results after both runs
uv run python experiments/compare_models.py \
    --classical results/checkpoints/classical_vit_baseline/history.json \
    --quantum   results/checkpoints/cv_quixer_baseline/history.json

# Hybrid model demo (no MNIST, runs in < 1 second)
uv run python experiments/demo_hybrid.py

# Run tests
uv run pytest tests/
```

## Package structure

```
cv_quixer/
├── config/         ExperimentConfig dataclasses + YAML loader
├── data/           MNIST/FashionMNIST download, patch extraction, DataLoader
├── models/
│   ├── base.py         BaseVisionTransformer (shared interface)
│   ├── classical/      Classical ViT wrapper
│   └── quantum/        CV quantum model (core thesis contribution)
│       ├── cv_encoding.py   Patch → bosonic mode displacement encoding (legacy)
│       ├── cv_layers.py     CVLayer stub + interferometer_param_count re-export
│       ├── cv_attention.py  PatchHypernetwork, HyperCVAttentionHead,
│       │                    HyperCVAttention + truncation loss helpers
│       └── cv_quixer.py     PatchEmbed, CVDecoder, CVQuixer (main model)
├── quantum/        Pure PyTorch CV simulation engine (no PennyLane)
│   ├── state.py        FockState — N-mode Fock statevector container
│   ├── circuit.py      CVCircuit — einsum-based gate application
│   ├── gates/          Gate matrix factories (Gaussian + non-Gaussian)
│   ├── interferometer.py  Clements rectangular beamsplitter mesh
│   ├── grad.py         ParameterShiftFunction (torch.autograd.Function)
│   └── ops.py          Observable matrices (QuadX, QuadP, number operator)
├── training/       Model-agnostic Trainer
├── evaluation/     Metrics + classical vs quantum comparison utilities
└── utils/          Parameter counting, logging (wandb), seeding, matplotlib
```

## CV-Quixer model architecture

```
Input: patches (B, N, patch_dim)
  ↓
PatchEmbed: Linear(patch_dim → embed_dim) + learnable pos_embed
  ↓
HyperCVAttention  (num_heads independent heads):
  Per head, per batch element:
    state ← FockState.vacuum(num_modes, cutoff_dim)
    M|ψ⟩ = Σ_i b_i (U_i|ψ⟩)          # LCU with learned complex b_i
    P(M)|ψ⟩ = Σ_j c_j M^j|ψ⟩          # polynomial with learned real c_j
      For each patch n:
        params ← PatchHypernetwork(patch_n)
                  # shape: (_gate_param_count(num_modes, bs_topology),)
                  # = 8m−2 (linear) or 8m (ring), m = num_modes
        Apply Killoran gate sequence: S(r,φ) → BS(θ,φ) → R(φ) → D(α) → K(κ)
    readout ← [⟨x̂_i⟩ for each mode of post-selected state]  # (num_modes,)
  Concatenate heads → (B, num_heads × num_modes)
  ↓
CVDecoder: Linear(num_heads×num_modes → H_d) → ReLU → Linear(H_d → num_classes)
  ↓
Logits: (B, num_classes)
```

## Key design decisions

- **Shared interface**: Both models inherit `BaseVisionTransformer` — the Trainer and
  evaluation code never import from `models/quantum` or `models/classical` directly.
- **Model factory**: `cv_quixer.models.build_model(config)` is the only place the string
  `"quantum"` / `"classical"` is resolved to a class.
- **Config system**: YAML files override Python dataclass defaults; loaded via `dacite`.
  Full resolved config is saved as JSON alongside every checkpoint.
- **Pure PyTorch simulation engine**: `cv_quixer/quantum/` is a standalone Fock-basis
  circuit simulator. Gate matrices are differentiable PyTorch ops via
  `torch.linalg.matrix_exp`; no PennyLane required for training. Gradient mode is
  `backprop` (autograd through einsum gate chain) or `parameter_shift` (PSR, deferred).
- **Hypernetwork-driven circuits**: each patch's quantum gate parameters are generated
  by a small classical MLP (`PatchHypernetwork`), not fixed learned weights. The circuit
  is input-dependent — each token sees a different unitary.
- **Identical data pipeline**: Both models receive the same `DataLoader` output — same
  patches, same normalisation.

## Simulation notes

The Fock backend memory scales as `cutoff_dim ^ num_modes`. Keep `num_modes ≤ 8` and
`cutoff_dim ≤ 8` for tractable single-GPU simulation.

Default config: `num_modes=4`, `cutoff_dim=6`, `num_heads=4`. Setting `target_params` in
`QuantumConfig` auto-scales `hyper_hidden_dim` to reach the target parameter count (binary
search at model init time, within ~5%).

**Fock truncation penalty**: add a loss term during training to penalise states near the
cutoff boundary. Two options, both weighted by `trunc_lambda`:
- `trunc_penalty = "norm"` — penalises `1 - ‖ψ‖²` (probability leakage outside cutoff)
- `trunc_penalty = "photon_number"` — penalises mean `⟨n̂⟩` normalised by `cutoff_dim - 1`

Use `model.forward(patches, return_trunc_loss=True)` to get `(logits, trunc_loss)`.

## Results

- Checkpoints: `results/checkpoints/<experiment_name>/` (gitignored)
- Training logs: `results/logs/` (gitignored)
- Thesis figures: `results/figures/` (tracked in git)

# CV-Quixer Codebase Guide

A Continuous Variable (CV) Quantum Vision Transformer benchmarked against a classical ViT on MNIST. Masters thesis project by Clyde Lhui.

## Project goal

Implement the Quixer transformer architecture on PennyLane's CV quantum computing framework and compare its classification performance on MNIST against an equivalent classical Vision Transformer.

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

# Run tests
uv run pytest tests/
```

## Package structure

```
cv_quixer/
├── config/         ExperimentConfig dataclasses + YAML loader
├── data/           MNIST download, patch extraction, DataLoader
├── models/
│   ├── base.py         BaseVisionTransformer (shared interface)
│   ├── classical/      Classical ViT wrapper
│   └── quantum/        CV quantum model (core thesis contribution)
│       ├── cv_encoding.py   Patch → bosonic mode displacement encoding
│       ├── cv_layers.py     Interferometer + squeezing + Kerr primitives
│       ├── cv_attention.py  CV quantum attention block
│       └── cv_quixer.py     Full CV-Quixer model
├── training/       Model-agnostic Trainer
├── evaluation/     Metrics + classical vs quantum comparison utilities
└── utils/          Logging (wandb), seeding, matplotlib helpers
```

## Key design decisions

- **Shared interface**: Both models inherit `BaseVisionTransformer` — the Trainer and evaluation code never import from `models/quantum` or `models/classical` directly.
- **Model factory**: `cv_quixer.models.build_model(config)` is the only place the string `"quantum"` / `"classical"` is resolved to a class.
- **Config system**: YAML files override Python dataclass defaults; loaded via `dacite`. Full resolved config is saved as JSON alongside every checkpoint.
- **Swappable PennyLane device**: Pass `config.quantum.backend` to `pennylane.device()`. Use `strawberryfields.fock` for exact Fock simulation (slow, supports Kerr) or `strawberryfields.gaussian` for fast Gaussian simulation (no Kerr).
- **Identical data pipeline**: Both models receive the same `DataLoader` output — same patches, same normalization.

## Simulation notes

The Fock backend memory scales as `cutoff_dim ^ num_modes`. Keep `num_modes ≤ 8` and `cutoff_dim ≤ 10` for tractable single-GPU simulation. The Gaussian backend is faster but restricts the circuit to linear optics (no Kerr), which limits expressibility.

## Results

- Checkpoints: `results/checkpoints/<experiment_name>/` (gitignored)
- Training logs: `results/logs/` (gitignored)
- Thesis figures: `results/figures/` (tracked in git)

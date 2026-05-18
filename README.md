# CV-Quixer

A hypernetwork-driven **Continuous-Variable (CV) Quantum Vision Transformer**,
benchmarked against a classical ViT on FashionMNIST. Masters thesis project
by Clyde Lhui.

The CV quantum head encodes each image patch into a parametric Gaussian +
Kerr unitary (Killoran et al. 2019 ansatz: `S → BS → R → D → K`) emitted by
a CNN hypernetwork. Across the patch sequence the head implements a linear
combination of unitaries (LCU) and a real-coefficient matrix polynomial
`P(M) = Σ_j c_j Mʲ`, producing per-mode quadrature `⟨x̂⟩` readouts that feed
a small classical MLP decoder. All Fock-basis simulation is **pure PyTorch**
(no PennyLane at training time), so gradients flow through standard
`torch.autograd`.

## Quick start

Requires Python ≥ 3.12 and [`uv`](https://github.com/astral-sh/uv).

```bash
# Forward + backward smoke test (~ 1 minute, no MNIST download)
uv run python experiments/smoke_test.py

# Mini experiment: 200 train / 50 test, 100 epochs, periodic checkpoints
uv run python experiments/mini_experiment.py

# Override the epoch count for short regression runs
uv run python experiments/mini_experiment.py --epochs 10

# Resume from a checkpoint
uv run python experiments/mini_experiment.py \
    --resume results/checkpoints/mini_experiment/checkpoint_epoch_0050.pt

# Full YAML-driven training (quantum and classical baselines)
uv run python experiments/train_quantum.py --config configs/cv_quixer.yaml
uv run python experiments/train_classical.py --config configs/classical_vit.yaml

# Compare results after both runs
uv run python experiments/compare_models.py \
    --classical results/checkpoints/classical_vit_baseline/history.json \
    --quantum   results/checkpoints/cv_quixer_baseline/history.json
```

## Repo layout

```
cv_quixer/
├── quantum/        Pure-PyTorch Fock-basis simulator
│   ├── state.py        FockState container
│   ├── circuit.py      einsum-based gate application
│   ├── gates/          Gaussian (S, BS, R, D) + non-Gaussian (Kerr, cubic) factories
│   ├── interferometer.py   Clements rectangular mesh
│   ├── ops.py          Observable matrices (x̂, p̂, n̂)
│   └── grad.py         Parameter-shift rule (deferred)
├── models/
│   ├── base.py         BaseVisionTransformer (shared interface)
│   ├── classical/      Classical ViT wrapper
│   └── quantum/        CV-Quixer (CNN hypernet, LCU, polynomial, multi-head)
├── config/         ExperimentConfig dataclasses + YAML loader
├── data/           MNIST / FashionMNIST + patch extraction
├── training/       Model-agnostic Trainer
├── evaluation/     Metrics + classical vs quantum comparison utilities
└── utils/          Parameter counting, logging (wandb), seeding, matplotlib

experiments/
├── smoke_test.py            Fast forward+gradient check
├── mini_experiment.py       200/50 subset, 100 epochs (override with --epochs)
├── train_quantum.py         Full YAML-driven training
├── train_classical.py       Classical-ViT baseline
└── compare_models.py        Training-curve overlays

configs/                  Defaults + per-experiment YAML overrides
tests/                    Project unit tests
scripts/                  SLURM batch jobs + CUDA diagnostics
```

## Testing

```bash
uv run pytest tests/
```

## Key design points

* **Pure PyTorch simulation** — Gaussian gates use exact closed-form
  Fock-basis matrix elements (column-norm sub-isometric, deficit equals the
  Fock-truncation leakage probability). Non-Gaussian gates: Kerr is diagonal,
  cubic phase via `matrix_exp`.
* **vmap batching** — `HyperCVAttention.forward` uses `torch.func.vmap` +
  `functional_call` to batch attention heads across the batch dimension; all
  inner ops are out-of-place for vmap compatibility.
* **Identical data pipeline** for classical and quantum models — same
  patches, same normalisation, same DataLoader output.
* **Auto-scaling** — `QuantumConfig.target_params > 0` binary-searches
  `cnn_channels_2` to land within 5 % of the target parameter count.

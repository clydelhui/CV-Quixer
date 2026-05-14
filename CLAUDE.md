# CV-Quixer Codebase Guide

A hypernetwork-driven Continuous Variable (CV) Quantum Vision Transformer benchmarked
against a classical ViT on FashionMNIST. Masters thesis project by Clyde Lhui.

## Project goal

Implement a hypernetwork-driven CV Quantum Vision Transformer (CV-Quixer) and compare
its classification performance on FashionMNIST against an equivalent classical Vision
Transformer. The quantum model uses a pure PyTorch Fock-basis simulation engine (no
PennyLane at inference time) so gradients flow through standard `torch.autograd`.

## Running experiments

### Local (macOS, MPS)

```bash
# Quick smoke test (< 1 min, no MNIST download needed)
uv run python experiments/smoke_test.py

# Mini experiment — 200 train / 50 test, 100 epochs, checkpoints every 10
uv run python experiments/mini_experiment.py

# Resume mini experiment from a checkpoint
uv run python experiments/mini_experiment.py \
    --resume results/checkpoints/mini_experiment/checkpoint_epoch_0050.pt

# Full training runs (YAML-driven)
uv run python experiments/train_quantum.py --config configs/cv_quixer.yaml
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

### SLURM cluster (SoC Compute Cluster)

Login via the login node, `cd ~/CV-Quixer`, then submit:

```bash
sbatch scripts/run_mini_experiment.sh      # submit job
squeue --me                                # monitor
cat slurm-cv_quixer_mini-<JOBID>.out      # view output
```

To resume from a checkpoint, edit `run_mini_experiment.sh` and add `--resume <path>`
to the `mini_experiment.py` invocation at the bottom.

#### SLURM job config (`run_mini_experiment.sh`)

| Directive | Value | Note |
|---|---|---|
| `--job-name` | `cv_quixer_mini` | shown in `squeue` |
| `--time` | `02:00:00` | 2-hour wall clock |
| `--gres` | `gpu:nv:1` | any NVIDIA `nv` GPU (currently runs on V100) |
| `--cpus-per-task` | `4` | CPU cores |
| `--mem` | `16G` | RAM |

#### GPU types on SoC Compute Cluster

| GPU | SLURM gres | Feature flag | CUDA arch |
|---|---|---|---|
| Tesla V100 (×58) | `gpu:nv:1` | `cuda70,v100` | sm_70 |
| Titan RTX | `gpu:nv:1` | `cuda75,titanrtx` | sm_75 |
| Tesla T4 | `gpu:nv:1` | `cuda75,t4` | sm_75 |
| A100 40 GB (×30) | `gpu:a100-40:1` | `cuda80,a100` | sm_80 |
| A100 80 GB (×10) | `gpu:a100-80:1` | `cuda80,a100` | sm_80 |
| H100 47 GB (×50) | `gpu:h100-47:1` | `cuda90,h100` | sm_90 |
| H100 96 GB (×20) | `gpu:h100-96:1` | `cuda90,h100` | sm_90 |
| H200 141 GB (×4) | `gpu:h200-141:1` | `cuda90,h200` | sm_90 |

To target a specific GPU: `sbatch --gres=gpu:a100-40:1 scripts/run_mini_experiment.sh`

To accept whichever GPU runs first (submit multiple, cancel the rest when one starts):
```bash
sbatch -J gpujob --gres=gpu:nv:1      scripts/run_mini_experiment.sh
sbatch -J gpujob --gres=gpu:a100-40:1 scripts/run_mini_experiment.sh
# first job to start should run: scancel -J gpujob --state=PENDING
```

#### PyTorch / CUDA setup on the cluster

- Cluster runs CUDA 12.5 and 12.9 on GPU nodes; no `module load` needed.
- `pyproject.toml` pins torch/torchvision to the `cu124` index on Linux via
  `[tool.uv.sources]`. CUDA 12.4 runtime wheels are forward-compatible with 12.5 / 12.9.
- The script builds a dedicated venv at `$HOME/.venvs/cv-quixer-cuda` (deleted and
  rebuilt fresh each run to avoid stale CUDA state). `UV_PROJECT_ENVIRONMENT` overrides
  uv's default `.venv` location.
- The local macOS `.venv` uses the MPS/CPU torch build. Never cross-activate the two venvs.

#### Storage on the cluster

| Location | Type | Notes |
|---|---|---|
| `$HOME/CV-Quixer/` | Network storage | Shared across all login and compute nodes. Code, checkpoints, logs go here. |
| `/tmp` on compute node | Local flash | Fastest I/O but **transient** — wiped immediately when the job ends. Do not write outputs here. |

#### Debugging CUDA / GPU issues

```bash
sbatch scripts/triage_cuda.sh        # runs nvidia-smi, nvcc, torch CUDA sanity check
squeue --me                          # check job status
cat slurm-cuda_triage-<JOBID>.out   # view triage output
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
│       ├── cv_encoding.py   DisplacementEncoding (legacy, unused in main model)
│       ├── cv_layers.py     CVLayer stub + interferometer_param_count re-export
│       ├── cv_attention.py  CNNHypernetwork, LCUSumCoefficients,
│       │                    PolynomialCoefficients, HyperCVAttentionHead,
│       │                    HyperCVAttention + truncation loss helpers
│       └── cv_quixer.py     CVDecoder, CVQuixer (main model),
│                            _param_count_formula, _resolve_cnn_channels
├── quantum/        Pure PyTorch CV simulation engine (no PennyLane)
│   ├── state.py        FockState — N-mode Fock statevector container
│   ├── circuit.py      CVCircuit — einsum-based gate application
│   ├── gates/
│   │   ├── gaussian.py      Squeeze, beamsplitter, rotation, displacement
│   │   └── non_gaussian.py  Kerr (kerr_phases diagonal phases), cubic phase (matrix_exp)
│   ├── interferometer.py  Clements rectangular beamsplitter mesh
│   ├── grad.py         ParameterShiftFunction (torch.autograd.Function, deferred)
│   └── ops.py          Observable matrices (QuadX, QuadP, number operator)
├── training/       Model-agnostic Trainer
├── evaluation/     Metrics + classical vs quantum comparison utilities
└── utils/          Parameter counting, logging (wandb), seeding, matplotlib

experiments/
├── mini_experiment.py   200 train / 50 test, 100 epochs, periodic checkpointing
├── smoke_test.py        Fast forward-pass + gradient check (no MNIST, < 1 min)
├── train_quantum.py     Full YAML-driven training (quantum)
├── train_classical.py   Full YAML-driven training (classical ViT)
├── compare_models.py    Plot classical vs quantum training curves
└── demo_hybrid.py       Hybrid model demo (no MNIST)

scripts/
├── run_mini_experiment.sh   SLURM batch job for mini_experiment
├── triage_cuda.sh           SLURM GPU/CUDA diagnostic job
└── debug_imports.py         Sequential import diagnostics with tracebacks

configs/
├── defaults.yaml            Base defaults for all config dataclasses
├── cv_quixer.yaml           CV-Quixer baseline (30 epochs, backprop)
├── classical_vit.yaml       Classical ViT baseline
└── ablations/
    ├── quixer_fewer_modes.yaml        num_modes=4
    └── quixer_gaussian_backend.yaml   cutoff_dim=6 near-Gaussian regime
```

## CV-Quixer model architecture

```
Input: patches (B, N, patch_dim)     patch_dim = patch_size²
  ↓
HyperCVAttention  (num_heads independent heads, batched with vmap):
  Per head (B elements processed in parallel):
    state ← FockState.vacuum(num_modes, cutoff_dim)
    LCU:        M|ψ⟩ = Σ_{i=0}^{N-1} b_i (U_i|ψ⟩)      # learned complex b_i per patch
    Polynomial: P(M)|ψ⟩ = Σ_{j=0}^{d} c_j M^j|ψ⟩         # learned real c_j, degree d
      For each patch i:
        gate_params ← CNNHypernetwork(patch_i, patch_idx=i)
          # Conv2d(1,C1,k)→Tanh→Conv2d(C1,C2,k)→Tanh→flatten+2D_PE→Linear
          # output dim: _gate_param_count(num_modes, bs_topology)
          # = 8m−2 (linear) or 8m (ring), where m = num_modes
        Apply Killoran gate sequence (per mode):
          Squeeze(r, φ) → Beamsplitter mesh(θ, φ) → Rotate(φ) → Displace(α) → Kerr(κ)
    readout ← [⟨x̂_i⟩ for each mode]                        # (num_modes,)
  Concatenate heads → (B, num_heads × num_modes)
  ↓
CVDecoder: Linear(num_heads×num_modes → H_d) → ReLU → Linear(H_d → num_classes)
  ↓
Logits: (B, num_classes)
```

## Key design decisions

- **Shared interface**: Both models inherit `BaseVisionTransformer` — Trainer and
  evaluation code never import from `models/quantum` or `models/classical` directly.
- **Model factory**: `cv_quixer.models.build_model(config)` is the only place `"quantum"`
  / `"classical"` is resolved to a class.
- **Config system**: YAML files override Python dataclass defaults via `dacite`.
  Full resolved config is saved as JSON alongside every checkpoint.
- **Pure PyTorch simulation**: `cv_quixer/quantum/` is a standalone Fock-basis circuit
  simulator. Gaussian gates (rotation, displacement, squeezing, beamsplitter) use exact
  analytic Fock-basis formulas (true sub-isometries: column norms ≤ 1); only cubic phase
  uses `matrix_exp`. Diagonal gates (rotation via `rotation_phases`, Kerr via
  `kerr_phases`) return `(D,)` phase vectors applied via `apply_single_mode_phases`
  (O(D) vs O(D³) matrix-exp). No PennyLane at training time. Gradient mode is `backprop`
  (autograd through einsum chain) or `parameter_shift` (PSR, deferred).
- **vmap batch parallelism**: `HyperCVAttention.forward` uses `torch.func.vmap` +
  `functional_call` to batch across B elements per head, replacing a sequential Python
  `for b in range(B)` loop. All inner ops are out-of-place. `FockState.vacuum` uses
  `index_put` (not in-place setitem); diagonal gates use `apply_single_mode_phases` with
  broadcasting (not `torch.diag` + einsum) for vmap compatibility.
- **CNN hypernetwork**: Each patch's gate parameters come from a 2-layer CNN
  (`CNNHypernetwork`) with 2D sinusoidal positional encodings injected before the linear
  projection. The circuit is input-dependent — every token sees a different unitary.
- **Identical data pipeline**: Both models receive the same `DataLoader` output — same
  patches, same normalisation.

## Config reference

### QuantumConfig

| Field | Default | Description |
|---|---|---|
| `num_modes` | 8 | Number of bosonic modes |
| `num_layers` | 4 | Reserved (not yet implemented) |
| `cutoff_dim` | 10 | Fock space truncation D |
| `grad_mode` | `"backprop"` | `"backprop"` or `"parameter_shift"` (PSR deferred) |
| `bs_topology` | `"linear"` | Beamsplitter mesh: `"linear"` or `"ring"` |
| `dtype` | `"complex128"` | `"complex64"` or `"complex128"` |
| `num_heads` | 4 | Parallel CV attention heads |
| `decoder_hidden_dim` | 64 | CVDecoder hidden layer width |
| `cnn_channels_1` | 8 | CNNHypernetwork first conv output channels |
| `cnn_channels_2` | 16 | CNNHypernetwork second conv output channels (auto-scaled if `target_params > 0`) |
| `cnn_kernel_size` | 3 | Conv kernel size |
| `poly_degree` | 2 | Matrix polynomial degree (keep ≤ 4) |
| `target_params` | -1 | If > 0, binary-search `cnn_channels_2` to hit this count (±5%) |
| `trunc_penalty` | `"none"` | `"none"`, `"norm"`, or `"photon_number"` |
| `trunc_lambda` | 0.01 | Truncation penalty loss weight |

### DataConfig

| Field | Default | Description |
|---|---|---|
| `dataset` | `"fashionmnist"` | `"fashionmnist"` or `"mnist"` |
| `image_size` | 28 | Input image side length |
| `patch_size` | 4 | Patch side length (must divide `image_size`) |
| `num_classes` | 10 | Output classes |
| `batch_size` | 64 | Training batch size |
| `num_workers` | 2 | DataLoader workers |
| `data_root` | `"data/"` | Dataset download/cache root |

### TrainingConfig

| Field | Default | Description |
|---|---|---|
| `lr` | 1e-3 | Learning rate |
| `epochs` | 30 | Training epochs |
| `optimizer` | `"adam"` | `"adam"` or `"sgd"` |
| `weight_decay` | 1e-4 | L2 regularisation |
| `seed` | 42 | Random seed |

### mini_experiment.py constants

| Constant | Value | Description |
|---|---|---|
| `EPOCHS` | 100 | Total training epochs |
| `TRAIN_SIZE` | 200 | Training subset size |
| `TEST_SIZE` | 50 | Test subset size |
| `BATCH_SIZE` | 32 | Batch size |
| `TARGET_PARAMS` | 13,760 | Parameter budget (auto-scales `cnn_channels_2`) |
| `CHECKPOINT_INTERVAL` | 10 | Save checkpoint every N epochs |

## Simulation notes

Fock backend memory scales as `cutoff_dim ^ num_modes`. Keep `num_modes ≤ 8` and
`cutoff_dim ≤ 10` for tractable single-GPU simulation.

**mini_experiment config**: `num_modes=2`, `cutoff_dim=6`, `num_heads=4`,
`patch_size=7` (16 patches), `dtype=complex64`. `target_params=13760` auto-scales
`cnn_channels_2` via binary search at init time.

**Fock truncation penalty** (weighted by `trunc_lambda`):
- `"norm"` — penalises `1 - ‖ψ‖²` (probability leakage outside truncated space)
- `"photon_number"` — penalises mean `⟨n̂⟩ / (cutoff_dim − 1)` (deferred)

Use `model.forward(patches, return_trunc_loss=True)` to get `(logits, trunc_loss)`.

## Results

| Path | Git-tracked | Contents |
|---|---|---|
| `results/checkpoints/<name>/` | No | `final_model.pt`, `checkpoint_epoch_NNNN.pt` |
| `results/logs/<name>/` | No | `history.json` per-epoch metrics |
| `results/figures/` | Yes | Thesis-quality training curve PNGs |

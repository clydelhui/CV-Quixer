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

# Full FashionMNIST experiment — same quantum config as mini_experiment but
# 60k train / 10k test. Default 3 epochs. Self-contained run directory under
# results/runs/. Use fractions for a local smoke run.
uv run python experiments/full_experiment.py \
    --epochs 1 --train-fraction 0.1 --test-fraction 0.1

# Resume a full run (continues writing into the same run directory)
uv run python experiments/full_experiment.py \
    --resume results/runs/full_fashionmnist_<ts>/checkpoints/latest.pt

# Cutoff-dim sweep — re-evaluate a trained checkpoint at larger Fock cutoffs
uv run python experiments/eval_cutoff_sweep.py \
    --checkpoint results/runs/full_fashionmnist_<ts>/checkpoints/final_model.pt \
    --cutoffs 6 8 10 12

# Post-hoc thesis figures from a completed full_experiment run
uv run python experiments/report_diagnostics.py \
    --run-dir results/runs/full_fashionmnist_<ts>/ --epoch best

# Run tests
uv run pytest tests/
```

`report_diagnostics.py` is the primary thesis-figure tool. It runs *after* a
`full_experiment.py` run, reading that run's saved artefacts (`config.json`,
`history.json`, `checkpoints/`, `predictions/`, `diagnostics/`) and writing the
qualitative + quantum-specific figures the training loop deliberately skips.
Heavy torch/model imports are deferred, so the default path is fast and runs
without a configured PyTorch backend.

| Flag | Default | Effect |
|---|---|---|
| `--run-dir` | (required unless `--multi-run`) | the `results/runs/full_fashionmnist_<ts>/` dir to report on |
| `--epoch` | `best` | `best` \| `final` \| integer epoch checkpoint to load |
| `--multi-run` | off | scan sibling `results/runs/full_fashionmnist_*` dirs, emit cross-run `sample_efficiency.png` (no `--run-dir` needed) |
| `--full-inference` | off | rebuild model from checkpoint + run full test-set inference instead of reading saved npz (slow; for old runs lacking artefacts, or as a cross-check) |
| `--check-parity` | off | compare saved-file predictions vs a fresh inference pass, print max/mean \|Δ y_probs\| + agreement, then exit |

Figures written into the run's figure dir (each wrapped in try/except — a
failed figure warns but does not abort the rest; `sanity_checks` runs first):

- *Fast* (history.json / npz only): `confusion_matrix_evolution`,
  `per_class_metrics_table`, `top_k_accuracy`, `calibration_reliability`,
  `hypernet_gate_param_histograms`, `photon_number_per_mode`,
  `state_norm_histogram`, `lcu_coefficients_heatmap`,
  `polynomial_coefficients_trajectory`.
- *Slow* (need readouts/test images — from saved npz, or `--full-inference`):
  `misclassification_gallery`, `embedding_tsne`.

This is complementary to, not a superset of, the figures `full_experiment.py`
itself re-renders every epoch (loss/accuracy/trunc/per-batch curves, per-class
accuracy curve, latest-epoch confusion matrix). The only overlap is the
confusion matrix (latest-epoch in the training loop vs full epoch-by-epoch
evolution here).

### SLURM cluster (SoC Compute Cluster)

Login via the login node, `cd ~/CV-Quixer`, then submit:

```bash
sbatch scripts/run_mini_experiment.sh      # submit job
squeue --me                                # monitor
cat slurm-cv_quixer_mini-<JOBID>.out      # view output
```

To resume from a checkpoint, edit `run_mini_experiment.sh` and add `--resume <path>`
to the `mini_experiment.py` invocation at the bottom.

#### Batch jobs

| Script | Job name | Wall | GPU / mem / cpus | Runs |
|---|---|---|---|---|
| `run_mini_experiment.sh` | `cv_quixer_mini` | `02:00:00` | `gpu:nv:1` / 16G / 4 | `mini_experiment.py` (200/50, 100 epochs) |
| `run_full_experiment.sh` | `cv_quixer_full` | `03:00:00` | `gpu:a100-40:1` / 32G / 4 | `full_experiment.py --train-fraction 0.1 --test-fraction 0.1` (3 epochs on a 10% subset; edit the script to resume or change fractions) |
| `run_eval_cutoff_sweep.sh` | `cv_quixer_eval` | `12:00:00` | `gpu:a100-40:1` / 32G / 4 | `eval_cutoff_sweep.py` — takes a checkpoint as `$1`, extra flags passed through |
| `triage_cuda.sh` | `cuda_triage` | — | `gpu:nv:1` | CUDA/GPU sanity diagnostics |

```bash
# Full experiment (edit the script to resume / change fractions)
sbatch scripts/run_full_experiment.sh

# Cutoff sweep — checkpoint is positional $1, remaining args forwarded
sbatch scripts/run_eval_cutoff_sweep.sh \
    results/runs/full_fashionmnist_<ts>/checkpoints/final_model.pt \
    --test-fraction 0.5 --cutoffs 8 10 12
```

All cluster scripts reuse the dedicated `$HOME/.venvs/cv-quixer-cuda` venv
(`run_full_experiment.sh` rebuilds it fresh; `run_eval_cutoff_sweep.sh` reuses
it) and `cd "$HOME/CV-Quixer"` before running.

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
├── config/         schema.py: ExperimentConfig/Quantum/Data/Training dataclasses
│                   + ObservableSpec. utils.py: load_config() YAML→config (LEGACY,
│                   see below) and save_config() config→JSON
├── data/           MNIST/FashionMNIST download, patch extraction, DataLoader
├── models/
│   ├── base.py         BaseVisionTransformer (shared interface)
│   ├── classical/      Classical ViT wrapper
│   └── quantum/        CV quantum model (core thesis contribution)
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
│   ├── interferometer.py  Clements rectangular beamsplitter mesh +
│   │                      interferometer_param_count
│   ├── grad.py         ParameterShiftFunction (torch.autograd.Function, deferred)
│   └── ops.py          Observable matrices: number, QuadX, QuadP,
│                       quadrature_x_squared, quadrature_p_squared
├── evaluation/     metrics.py only (classical-vs-quantum compare.py removed)
└── utils/          params (parameter counting), logging (wandb), reproducibility (seeding)

experiments/
├── smoke_test.py          Fast forward-pass + gradient check (no MNIST, < 1 min)
├── mini_experiment.py     200 train / 50 test, 100 epochs, periodic checkpointing
├── full_experiment.py     60k/10k FashionMNIST, self-contained results/runs/<ts>/ dir
├── eval_cutoff_sweep.py   Re-evaluate a trained checkpoint at larger Fock cutoffs
└── report_diagnostics.py  Post-hoc thesis figures from a full_experiment run

scripts/
├── run_mini_experiment.sh     SLURM batch job for mini_experiment
├── run_full_experiment.sh     SLURM batch job for full_experiment (A100, 3 h)
├── run_eval_cutoff_sweep.sh   SLURM batch job for eval_cutoff_sweep (A100, 12 h)
├── triage_cuda.sh             SLURM GPU/CUDA diagnostic job
└── debug_imports.py           Sequential import diagnostics with tracebacks

configs/                 LEGACY — not loaded by any current experiment script.
├── defaults.yaml         Kept deliberately (intend to revive YAML-driven runs).
├── cv_quixer.yaml        Experiment scripts now build ExperimentConfig directly
├── classical_vit.yaml    in Python; load_config() in config/utils.py is unused.
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
    readout ← configurable observables per `readout_observables`     # (R,)
             # each spec → x | p | x² | p² | n | prob_n on mode(s)
             # default (no config): ⟨x̂⟩ per mode → R = num_modes
  Concatenate heads → (B, num_heads × R)
  ↓
CVDecoder: Linear(num_heads×R → H_d) → ReLU → Linear(H_d → num_classes)
  ↓
Logits: (B, num_classes)
```

`R` = length of the expanded observable plan (`schema._expand_observable_specs`):
spec list order → mode order → n order (for `prob_n`). It is `num_modes` only in
the default ⟨x̂⟩-per-mode case; e.g. a `prob_n` PNR spec over all modes gives
`R = num_modes × len(n)`. The decoder input dim is derived from this plan, not
hard-wired to `num_modes`. See `QuantumConfig.readout_observables` /
`readout_observable` in the config reference below.

## Key design decisions

- **Shared interface**: Both models inherit `BaseVisionTransformer`. There is no
  longer a model-agnostic `Trainer` class — each experiment script
  (`mini_experiment.py`, `full_experiment.py`) owns its own training loop and
  drives the model only through the `BaseVisionTransformer` interface, never
  importing from `models/quantum` or `models/classical` directly.
- **Model factory**: `cv_quixer.models.build_model(config)` is the only place `"quantum"`
  / `"classical"` is resolved to a class.
- **Config system**: Experiment scripts construct the `ExperimentConfig`
  dataclasses directly in Python (no YAML at runtime). `full_experiment.py`
  writes the fully resolved config to `config.json` in its run directory;
  `eval_cutoff_sweep.py` and `report_diagnostics.py` reconstruct
  `ExperimentConfig` from that saved `config.json` via `dacite`. The
  `load_config()` YAML loader and `configs/*.yaml` are kept but currently
  unused (intended for a future revival of YAML-driven runs).
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
| `num_modes` | 4 | Number of bosonic modes |
| `num_layers` | 4 | Reserved — not read until multi-layer stacking is implemented |
| `cutoff_dim` | 6 | Fock space truncation D |
| `grad_mode` | `"backprop"` | `"backprop"` or `"parameter_shift"` (PSR deferred) |
| `param_shift_shift` | 1.5708 | PSR shift `s` (π/2); only used when `grad_mode="parameter_shift"` |
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
| `readout_observable` | `None` | Legacy single-string selector: `"quadrature_x"`, `"photon_number"`, or `"pnr_distribution"`. Mutually exclusive with `readout_observables` |
| `readout_observables` | `None` | Canonical `list[ObservableSpec]`. Mutually exclusive with `readout_observable`. Both `None` → default ⟨x̂⟩ per mode |

`ObservableSpec(type, mode="all", n=None)`: `type` ∈ `{"x","p","x_squared",
"p_squared","n","prob_n"}`; `mode` is an int, list of ints, or `"all"`; `n` is
required iff `type=="prob_n"` (int or list of ints in `[0, cutoff_dim)`) and
forbidden otherwise. The expanded plan order (spec → mode → n) fixes the readout
vector layout and the decoder input dim. Validated/expanded in
`QuantumConfig.__post_init__` (raises on invalid combos, e.g. both readout
fields set, unknown type, out-of-range mode/n).

### DataConfig

| Field | Default | Description |
|---|---|---|
| `dataset` | `"fashionmnist"` | `"fashionmnist"` or `"mnist"` |
| `normalize` | `True` | Compute & cache dataset mean/std on first load; `False` → `ToTensor` only |
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
| `checkpoint_dir` | `"results/checkpoints/"` | Default checkpoint root |
| `log_dir` | `"results/logs/"` | Default per-run log root |
| `log_interval` | 10 | Log every N batches |

Note: `mini_experiment.py` / `full_experiment.py` set their own constants and
output paths (see below) rather than reading every `TrainingConfig` field.

### mini_experiment.py constants

| Constant | Value | Description |
|---|---|---|
| `EPOCHS` | 100 | Total training epochs |
| `TRAIN_SIZE` | 200 | Training subset size |
| `TEST_SIZE` | 50 | Test subset size |
| `BATCH_SIZE` | 32 | Batch size |
| `TARGET_PARAMS` | 13,760 | Parameter budget (auto-scales `cnn_channels_2`) |
| `CHECKPOINT_INTERVAL` | 10 | Save checkpoint every N epochs |

### full_experiment.py constants

Same quantum config as `mini_experiment.py` (num_modes=2, cutoff_dim=6,
num_heads=4, poly_degree=2, ~13.7k params) but trained on the full 60k/10k
split. CLI-overridable: `--epochs`, `--train-fraction`, `--test-fraction`,
`--resume`.

| Constant | Value | Description |
|---|---|---|
| `EPOCHS` | 3 | Default epochs (~75-90 min/epoch V100, ~30-45 min/epoch A100) |
| `BATCH_SIZE` | 64 | Batch size |
| `TARGET_PARAMS` | 13,760 | Parameter budget (auto-scales `cnn_channels_2`) |
| `CHECKPOINT_INTERVAL` | 1 | Versioned `epoch_NNNN.pt` every N epochs |
| `MA_WINDOW` | 50 | Moving-average window for per-batch plots |

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

Output layout differs per script (none of `results/` is git-tracked):

**`mini_experiment.py`** (split across fixed roots):

| Path | Contents |
|---|---|
| `results/checkpoints/mini_experiment/` | `final_model.pt`, `checkpoint_epoch_NNNN.pt` |
| `results/logs/mini_experiment/history.json` | Per-epoch metrics |
| `results/figures/mini_experiment_curve.png` | Two-panel training curve |

**`full_experiment.py`** (one self-contained directory per run,
`results/runs/full_fashionmnist_<YYYY-MM-DD_HH-MM-SS>/`):

| Entry | Contents |
|---|---|
| `config.json` | Full resolved `ExperimentConfig` (read back by eval/diagnostics) |
| `history.json` | Per-epoch + per-batch + meta metrics — plot source of truth |
| `parameter_table.txt` | Snapshot of `print_parameter_table()` |
| `checkpoints/` | `latest.pt` (every epoch), `best.pt` (best test acc), `final_model.pt`, `epoch_NNNN.pt` |
| `figures/` | Training-loop PNGs re-rendered every epoch; `report_diagnostics.py` writes its figures here too |
| `predictions/`, `diagnostics/`, `logs/` | Saved test images / readouts / quantum diagnostics consumed by `report_diagnostics.py` |

**`eval_cutoff_sweep.py`**: writes `<run_dir>/eval/cutoff_sweep_<timestamp>/`
containing `results.json`, `results.csv`, and per-metric plots.

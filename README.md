# CV-Quixer

A hypernetwork-driven **Continuous-Variable (CV) Quantum Vision Transformer**,
benchmarked against a classical ViT on FashionMNIST. Masters thesis project
by Clyde Lhui.

The CV quantum head encodes each image patch into a parametric Gaussian +
Kerr unitary (Killoran et al. 2019 ansatz: `S → BS → R → D → K`) emitted by
a CNN hypernetwork. Across the patch sequence the head implements a linear
combination of unitaries (LCU) and a real-coefficient matrix polynomial
`P(M) = Σ_j c_j Mʲ`, producing configurable per-mode observable readouts
(`x`, `p`, `x²`, `p²`, `n`, or photon-number-resolving `prob_n`; default ⟨x̂⟩
per mode) that feed a small classical MLP decoder. All Fock-basis simulation is
**pure PyTorch** (no PennyLane at training time), so gradients flow through
standard `torch.autograd`.

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

# Full FashionMNIST (60k/10k), self-contained results/runs/<ts>/ dir.
# Default 3 epochs; use fractions for a quick local run.
uv run python experiments/full_experiment.py \
    --epochs 1 --train-fraction 0.1 --test-fraction 0.1

# Re-evaluate a trained checkpoint at larger Fock cutoffs
uv run python experiments/eval_cutoff_sweep.py \
    --checkpoint results/runs/<ts>/checkpoints/final_model.pt --cutoffs 6 8 10 12

# Post-hoc thesis figures from a completed full_experiment run
uv run python experiments/report_diagnostics.py --run-dir results/runs/<ts>/

# Single run at a chosen parameter budget + observable preset
uv run python experiments/full_experiment.py \
    --epochs 1 --train-fraction 0.1 --test-fraction 0.1 \
    --target-params 8000 --observables xp

# Hyperparameter sweep over parameter budget × observable preset (× seeds).
# --dry-run writes the manifest only; --launch local|slurm runs/submits it.
uv run python experiments/sweep.py \
    --target-params 8000 13760 20000 --observables x xpxsps pnr \
    --epochs 3 --train-fraction 0.1 --test-fraction 0.1 --dry-run

# Aggregate a sweep into a thesis table (summary.csv/md) + comparison plots
uv run python experiments/report_sweep.py --sweep-dir results/sweeps/<sweep>_<ts>/
```

## Repo layout

```
cv_quixer/
├── quantum/        Pure-PyTorch Fock-basis simulator
│   ├── state.py        FockState container
│   ├── circuit.py      einsum-based gate application
│   ├── gates/          Gaussian (S, BS, R, D) + non-Gaussian (Kerr, cubic) factories
│   ├── interferometer.py   Clements rectangular mesh
│   ├── ops.py          Observable matrices (x̂, p̂, n̂, x̂², p̂²)
│   └── grad.py         Parameter-shift rule (deferred)
├── models/
│   ├── base.py         BaseVisionTransformer (shared interface)
│   ├── classical/      Classical ViT wrapper
│   └── quantum/        CV-Quixer (CNN hypernet, LCU, polynomial, multi-head)
├── config/         schema.py: ExperimentConfig dataclasses + ObservableSpec;
│                   observable_presets.py: named readout presets
│                   (utils.py load_config() YAML loader — legacy, unused)
├── data/           MNIST / FashionMNIST + patch extraction
├── evaluation/     metrics.py (compare.py removed)
└── utils/          Parameter counting, logging (wandb), seeding

experiments/
├── smoke_test.py            Fast forward+gradient check
├── mini_experiment.py       200/50 subset, 100 epochs (override with --epochs)
├── full_experiment.py       60k/10k FashionMNIST, self-contained results/runs/<ts>/
├── eval_cutoff_sweep.py     Re-evaluate a checkpoint at larger Fock cutoffs
├── sweep.py                 Fan a (param × observable × seed) grid into runs
├── report_sweep.py          Cross-run sweep table + comparison plots
└── report_diagnostics.py    Post-hoc thesis figures from a full_experiment run

configs/                  LEGACY — YAML overrides, not loaded by any current
                          script; kept for a planned YAML-workflow revival
tests/                    Project unit tests
scripts/                  SLURM batch jobs (mini/full/eval/sweep) + CUDA diagnostics
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
* **vmap batching** — `HyperCVAttention.forward` uses nested `torch.func.vmap` +
  `functional_call` to run the attention heads and batch elements as a single
  vectorized pass (head → batch → patch nesting); all inner ops are out-of-place
  for vmap compatibility.
* **Identical data pipeline** for classical and quantum models — same
  patches, same normalisation, same DataLoader output.
* **Auto-scaling** — `QuantumConfig.target_params > 0` binary-searches the configured `scaling_knob`
  (default `num_heads`, but e.g. `cnn_channels_2` / `num_modes` also work) to land
  within ~10 % of the target parameter count.
* **Configurable readout** — `QuantumConfig.readout_observables` selects any
  mix of `x`, `p`, `x²`, `p²`, `n`, and photon-number-resolving `prob_n`
  observables per mode (defaults to ⟨x̂⟩ per mode).
* **No Trainer class** — each experiment script owns its training loop and
  drives models only through `BaseVisionTransformer`; `build_model(config)`
  is the only model factory.
* **Configs built in Python** — scripts construct `ExperimentConfig`
  directly; `full_experiment.py` writes `config.json` per run, which
  `eval_cutoff_sweep.py` / `report_diagnostics.py` reload via `dacite`.
* **Hyperparameter sweeps** — `full_experiment.py` exposes the two sweep axes
  (`--target-params`, `--observables <preset>`) plus `--seed`/`--run-name`/
  `--runs-root`. `experiments/sweep.py` fans a grid into one process per point
  (local or SLURM array via `scripts/run_sweep.sh`) under
  `results/sweeps/<sweep>_<ts>/`; `experiments/report_sweep.py` aggregates them
  into `summary.csv`/`summary.md` + comparison plots. Optional W&B logging via
  `--wandb` (grouped per sweep; `WANDB_MODE=offline` on offline nodes).

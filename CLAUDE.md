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

# Single run with a chosen parameter budget + observable preset
uv run python experiments/full_experiment.py \
    --epochs 1 --train-fraction 0.1 --test-fraction 0.1 \
    --target-params 8000 --observables xp

# Hyperparameter sweep — two modes. --dry-run writes the manifest only;
# --launch local runs sequentially here; --launch slurm submits a job array.
# BUDGET mode: grid over param budgets × observables (× seeds). --scaling-knob is
# required here (no default — name the budget knob explicitly).
uv run python experiments/sweep.py \
    --target-params 8000 13760 20000 --observables x xpxsps pnr \
    --scaling-knob num_heads \
    --epochs 3 --train-fraction 0.1 --test-fraction 0.1 --dry-run

# MANUAL mode: set architecture knobs directly as grid axes (no auto-scaling).
# At least one budget or manual axis is required; the two modes can coexist.
uv run python experiments/sweep.py \
    --num-heads 4 6 --num-modes 2 3 --decoder-num-layers 2 3 --observables xpxsps \
    --epochs 3 --train-fraction 0.1 --test-fraction 0.1 --dry-run

# Aggregate a finished/partial sweep into a thesis table + comparison plots
# (also renders the full report_diagnostics suite per run; --skip-per-run-figures to skip)
uv run python experiments/report_sweep.py \
    --sweep-dir results/sweeps/<sweep>_<ts>/

# Compare ≥2 sweeps side by side (e.g. one quantum sweep vs one quantum_shared
# sweep) into one combined table + cross-sweep figures (JSON only, no torch)
uv run python experiments/report_sweep_compare.py \
    --sweep-dir results/sweeps/<sweepA>_<ts>/ \
    --sweep-dir results/sweeps/<sweepB>_<ts>/ \
    --label quantum --label quantum_shared

# Cutoff-dim sweep across EVERY run in a sweep dir (one eval_cutoff_sweep per run).
# --dry-run writes the manifest only; --launch local|slurm as with sweep.py.
uv run python experiments/eval_cutoff_sweep_all.py \
    --sweep-dir results/sweeps/<sweep>_<ts>/ --launch slurm

# Aggregate the whole-sweep cutoff eval into cross-run figures + per-cutoff suites
uv run python experiments/report_cutoff_sweep.py \
    --sweep-dir results/sweeps/<sweep>_<ts>/

# Run tests
uv run pytest tests/
```

`report_diagnostics.py` is the single figure-generation tool for the project —
`report_sweep.py` and `report_cutoff_sweep.py` *invoke* it (one subprocess per
run / per cutoff) rather than reimplementing any figures.
`full_experiment.py` writes only **raw** data artefacts (`config.json`,
`history.json`, `predictions/`, `diagnostics/`, `checkpoints/`,
`subset_indices.npz`); `report_diagnostics.py` consumes those and *derives*
every metric and figure on demand. `history["epoch"]` carries a training-time
log of loss/acc values but is **never** treated as canonical for figures —
`plot_training_curves` derives accuracy/loss from each epoch's predictions
npz and emits a `RuntimeWarning` if the logged value disagrees with the
derived one. Run `report_diagnostics.py` ad-hoc to refresh figures from a
partial or in-progress run (`history.json` is updated every epoch). Heavy
torch/model imports are deferred, so the default path is fast and runs
without a configured PyTorch backend.

If you have an older run that lacks the new artefacts (per-epoch
`epoch_NNNN_train.npz`, or diagnostics npz without `lcu_coeffs` /
`poly_coeffs` keys), `report_diagnostics.py` fails loudly with a hint to
run `experiments/backfill_artefacts.py --run-dir <run>` first. Backfill
replays the post-epoch eval pass from each checkpoint and writes the
missing artefacts; it also overwrites `history["epoch"]["train_acc"]`
etc. with the clean post-epoch values (older runs stored a running
per-batch average for `train_acc` that is *not* comparable to test_acc).

| Flag | Default | Effect |
|---|---|---|
| `--run-dir` | (required unless `--multi-run`) | the `results/runs/full_fashionmnist_<ts>/` dir to report on |
| `--epoch` | `best` | `best` \| `final` \| integer epoch checkpoint to load |
| `--multi-run` | off | scan sibling `results/runs/full_fashionmnist_*` dirs, emit cross-run `sample_efficiency.png` (no `--run-dir` needed) |
| `--full-inference` | off | rebuild model from checkpoint + run full test-set inference instead of reading saved npz (slow; for old runs lacking artefacts, or as a cross-check) |
| `--check-parity` | off | compare saved-file predictions vs a fresh inference pass, print max/mean \|Δ y_probs\| + agreement, then exit |

Figures written into the run's figure dir (each wrapped in try/except — a
failed figure warns but does not abort the rest; `sanity_checks` runs first):

- *Fast* (history.json / npz only): training curves —
  `loss_curve`, `accuracy_curve`, `trunc_loss_curve`,
  `cvqnn_trunc_loss_curve`, `query_trunc_loss_curve`,
  `per_class_accuracy_curve`, `confusion_matrix_test`,
  `per_batch_train_loss`, `per_batch_trunc_loss`,
  `per_batch_train_accuracy`, `per_batch_grad_norm`,
  `per_batch_cvqnn_trunc_loss`, `per_batch_query_trunc_loss` — plus
  `confusion_matrix_evolution`, `per_class_metrics_table`,
  `top_k_accuracy`, `calibration_reliability`,
  `hypernet_gate_param_histograms`, `cvqnn_param_values`,
  `photon_number_per_mode`, `state_norm_histogram`,
  `lcu_coefficients_heatmap`, `polynomial_coefficients_trajectory`.
  The query / W trunc figures are skipped when the stream is absent or
  identically zero (old runs, canonical runs without the stream, L_W=0).
- *Slow* (need readouts/test images — from saved npz, or `--full-inference`):
  `misclassification_gallery`, `embedding_tsne`.
- *Stacked runs* (`model="quantum_stacked"`, detected data-driven from the
  block-prefixed diagnostics keys): the four per-stage figures —
  `hypernet_gate_param_histograms`, `cvqnn_param_values`,
  `lcu_coefficients_heatmap`, `polynomial_coefficients_trajectory` — render
  one file per stage (`…_block{b}.png`, `…_agg.png`), with a separate
  `…_block{b}_query.png` histogram for each block's query-unitary slice.
  Canonical runs keep the historic unsuffixed filenames; state-norm / photon
  figures keep canonical names (they describe the decoder-input stage).

#### Hyperparameter sweeps

Sweep axes are exposed on `full_experiment.py` directly — the budget axes
(`--target-params`/`--scaling-knob`) and every architecture knob as a direct flag
(manual mode). Defaults preserve the canonical single-run behaviour exactly:

| Flag | Default | Effect |
|---|---|---|
| `--model NAME` | `quantum` | `quantum` \| `quantum_shared` \| `quantum_stacked` \| `classical`. `quantum_shared` defaults `--scaling-knob` to `num_heads`; `quantum_stacked` is the seq-to-seq stacked model (ADR-0002). |
| `--num-seq2seq-blocks N` | 1 | (`quantum_stacked`) seq-to-seq blocks; ≥ 1; excludes the optional aggregator block. A valid `--scaling-knob`, but coarse |
| `--pooling MODE` | `mean` | (`quantum_stacked`) `mean` pools final tokens over positions; `quixer` appends a canonical seq-to-one aggregator block |
| `--block-residual on\|off` | `on` | (`quantum_stacked`) identity residual `x + block(x)` from block 2 onward; `off` = pure-pipeline ablation |
| `--query-trunc-lambda F` | 0.01 | (`quantum_stacked`) weight of the separate query-unitary truncation penalty |
| `--target-params N` | -1 (off; frozen 13,530-param model) | parameter budget; when >0 auto-scales `--scaling-knob` (default `num_heads`) |
| `--num-layers L` | 1 | per-patch circuit depth (L stacked gate sequences + L−1 `BS→Rot` interferometers); can also be a `--scaling-knob` |
| `--observables NAME` | `xpxsps` | named observable preset (see below) |
| `--observables-json STR` | — | ad-hoc `ObservableSpec` JSON; requires `--run-name` |
| `--seed N` | 42 | training seed (vary for repeats / error bars) |
| direct arch flags | frozen-model values | `--num-modes`, `--cutoff-dim`, `--num-heads`, `--cnn-channels-1/-2`, `--cnn-kernel-size`, `--decoder-hidden-dim`, `--poly-degree`, `--cnn-num-conv-layers`, `--hypernet-num-linear-layers`, `--decoder-num-layers` — set any QuantumConfig knob directly (manual mode, no auto-scaling). Honoured even with `--target-params` set, except the chosen `--scaling-knob`. `--cutoff-dim` also resizes the `pnr`/`xpxsps_pnr` plans + the `auto` gate bound |
| `--decoder-hidden-mult C` | off | size the decoder hidden width as `round(C × in_dim)` instead of a fixed value (see `decoder_hidden_mult`). Mutually exclusive with `--decoder-hidden-dim` |
| `--run-name STR` | `full_fashionmnist_<ts>` | run-dir name |
| `--runs-root PATH` | `results/runs` | parent dir (sweeps pass `results/sweeps/<sweep>`) |
| `--wandb` / `--wandb-group` / `--wandb-tags` | off | W&B logging (see below) |

**Observable presets** (`cv_quixer/config/observable_presets.py`,
`resolve_observables(name, cutoff_dim)`): `x` (⟨x̂⟩/mode), `xp`, `xpxsps`
(x,p,x²,p² per mode — the `full_experiment.py` default), `n` (⟨n̂⟩/mode),
`xpn`, `pnr` (P(n=k) for k=0..min(cutoff-1, 5) per mode — a fixed PNR detector
resolving limit, `PNR_MAX_PHOTON`), `xpxsps_pnr` (xpxsps + pnr combined →
(4 + min(cutoff, 6))·num_modes scalars/head). Because `target_params`
targets the *total* count, the observable choice barely shifts the budget — the
two axes are cleanly separable.

`experiments/sweep.py` fans a Cartesian grid into independent `full_experiment.py`
runs, one process per grid point, in **two combinable modes**:

- **Budget mode** — `--target-params … --scaling-knob …` (+ `--observables`,
  `--seeds`, `--num-layers`): each run auto-scales the named knob to a param
  budget. `--scaling-knob` is **required when `--target-params` is given** (no
  default — name the budget knob explicitly). Run dirs are named
  `p{params}__obs-{preset}__seed{seed}` (`__knob-{knob}` marker unless the historic
  `cnn_channels_2`; `__L{n}` when `num_layers > 1`).
- **Manual mode** — set architecture knobs **directly** as grid axes (no
  auto-scaling): `--num-heads`, `--num-modes`, `--cutoff-dim`, `--poly-degree`,
  `--cnn-channels-1`, `--cnn-channels-2`, `--cnn-kernel-size`,
  `--decoder-hidden-dim`, `--cnn-num-conv-layers`, `--hypernet-num-linear-layers`,
  `--decoder-num-layers`, `--decoder-hidden-mult`, plus the stacked-model axes
  `--num-seq2seq-blocks`, `--pooling`, `--block-residual`,
  `--query-trunc-lambda` (each `nargs='+'`; the same flags
  exist on `full_experiment.py` for single runs; `--decoder-hidden-mult` is
  mutually exclusive with `--decoder-hidden-dim`). Run dirs are named
  `manual__obs-{preset}__seed{seed}` plus a short marker per active axis
  (`__nh6`, `__nm3`, `__pd2`, `__ncl3`, `__hll2`, `__dnl3`, `__dhm1.0`, `__sb2`,
  `__pool-quixer`, `__nores`, `__qtl0.02`, …).

At least one of `--target-params` or a manual axis is required; they can coexist
(fix some fields manually while auto-scaling one knob). It writes
`results/sweeps/<sweep_name>_<ts>/sweep_manifest.json` and per-run sub-dirs (each a
full run dir). All runs share `--subset-seed` + fractions so they see the identical
data subset.
`--launch local` runs them sequentially; `--launch slurm` submits
`scripts/run_sweep.sh` as a job array (`#SBATCH --array`, one task per grid
point; the per-arch CUDA venv is auto-built once via `scripts/setup_cuda_env.sh`'s
flock — no manual pre-build); `--dry-run` writes the manifest only.

`experiments/report_sweep.py --sweep-dir <dir>` scans the sweep (JSON only — no
torch), emitting `summary.csv` + `summary.md` (one row/run: target/achieved
params, observable preset, the resolved architecture knobs, best/final acc,
runtime) and `figures/acc_vs_params.png` + `figures/acc_by_observable.png`
(cross-run figures plot the *achieved* param count), plus one
`figures/acc_vs_<field>.png` per architecture field that *varies* across the runs
(the manual-sweep figure — e.g. `acc_vs_num_heads.png`; absent fields / constant
fields are skipped). It then **also renders the full `report_diagnostics.py` figure suite for
every run by default** (one subprocess per run; `report_diagnostics`'s default
path is npz/JSON-based, so heavy torch imports stay deferred); pass
`--skip-per-run-figures` for the fast cross-run-only pass.

**Whole-sweep cutoff sweep.** `experiments/eval_cutoff_sweep_all.py --sweep-dir
<dir>` fans `eval_cutoff_sweep.py` over **every** run in a sweep (discovers runs
that have `checkpoints/<name>` + `config.json` + `subset_indices.npz`), writes
`cutoff_sweep_manifest.json` and drives every per-run eval with one shared
`--output-name` (`cutoff_sweep_<ts>`) so all outputs land at
`<run>/eval/<name>/`; `--launch local|slurm|none` / `--dry-run` mirror `sweep.py`
(slurm submits `scripts/run_eval_cutoff_sweep_array.sh` as a job array, one task
per run). `experiments/report_cutoff_sweep.py --sweep-dir <dir>` (JSON only)
aggregates the per-run `eval/<name>/results.json` into `cutoff_summary.csv/md` +
`figures/cutoff/{acc_vs_cutoff_all,trunc_loss_vs_cutoff_all,acc_recovery_vs_params,
acc_vs_params_by_cutoff}.png`, then auto-renders the full per-cutoff `D{NN}/`
diagnostics suite (`--skip-per-run-figures` to skip). Cost caveat: cutoff cost
scales with `num_heads` (not the param budget), so high-head runs need
`--test-fraction` to stay under the array wall — default `--cutoffs 6 8 10`.

**Weights & Biases** (optional, opt-in via `--wandb`; off by default). Wired
through `cv_quixer/utils/logging.py` (`init_logging`/`log_metrics`/
`finish_logging`, lazy `wandb` import). Per-epoch metrics are logged on an
`epoch` x-axis; the full run config — including the swept axes (`target_params`,
the observable specs, `seed`) — is captured for cross-run views (the *achieved*
param count lands in `history["meta"].achieved_params` / `config.json`, not the
wandb config); sweep runs are grouped under the sweep name with
`params=`/`obs=`/`seed=` tags. On
cluster nodes without outbound network use `WANDB_MODE=offline` and `wandb sync`
later. Not wandb-Sweeps: the grid/orchestration stays in `sweep.py`.

### SLURM cluster (SoC Compute Cluster)

Login via the login node, `cd ~/CV-Quixer`, then submit:

```bash
sbatch scripts/run_mini_experiment.sh      # submit job
squeue --me                                # monitor
cat slurm_logs/slurm-cv_quixer_mini-<JOBID>.out   # view output (logs → slurm_logs/)
```

To resume from a checkpoint, edit `run_mini_experiment.sh` and add `--resume <path>`
to the `mini_experiment.py` invocation at the bottom.

#### Batch jobs

| Script | Job name | Wall | GPU / mem / cpus | Runs |
|---|---|---|---|---|
| `run_mini_experiment.sh` | `cv_quixer_mini` | `02:00:00` | `gpu:nv:1` / 16G / 4 | `mini_experiment.py` (200/50, 100 epochs) |
| `run_full_experiment.sh` | `cv_quixer_full` | `03:00:00` | `gpu:a100-40:1` / 32G / 4 | `full_experiment.py --train-fraction 0.1 --test-fraction 0.1` (3 epochs on a 10% subset; edit the script to resume or change fractions) |
| `run_eval_cutoff_sweep.sh` | `cv_quixer_eval` | `12:00:00` | `gpu:a100-40:1` / 32G / 4 | `eval_cutoff_sweep.py` — takes a checkpoint as `$1`, extra flags passed through |
| `run_sweep.sh` | `cv_quixer_sweep` | `03:00:00` | `gpu:a100-40:1` / 32G / 4 | job ARRAY — takes a `sweep_manifest.json` as `$1`, runs the `full_experiment.py` entry for `$SLURM_ARRAY_TASK_ID` |
| `run_eval_cutoff_sweep_array.sh` | `cv_quixer_eval_all` | `12:00:00` | `gpu:a100-40:1` / 32G / 4 | job ARRAY — takes a `cutoff_sweep_manifest.json` as `$1`, runs the `eval_cutoff_sweep.py` entry for `$SLURM_ARRAY_TASK_ID` |
| `triage_cuda.sh` | `cuda_triage` | — | `gpu:nv:1` | CUDA/GPU sanity diagnostics |

```bash
# Full experiment (edit the script to resume / change fractions)
sbatch scripts/run_full_experiment.sh

# Cutoff sweep — checkpoint is positional $1, remaining args forwarded
sbatch scripts/run_eval_cutoff_sweep.sh \
    results/runs/full_fashionmnist_<ts>/checkpoints/final_model.pt \
    --test-fraction 0.5 --cutoffs 8 10 12

# Hyperparameter sweep — sweep.py writes the manifest and prints the exact
# `sbatch --array=0-<N-1> scripts/run_sweep.sh <manifest>` command (or pass
# --launch slurm to submit it directly). The CUDA venv is auto-built by the job.
uv run python experiments/sweep.py \
    --target-params 8000 13760 20000 --observables x xp xpxsps pnr \
    --scaling-knob num_heads \
    --epochs 3 --train-fraction 0.1 --test-fraction 0.1 --launch slurm

# Whole-sweep cutoff eval — eval_cutoff_sweep_all.py writes the manifest and
# prints/submits `sbatch --array=0-<N-1> scripts/run_eval_cutoff_sweep_array.sh
# <manifest>` (one array task per run). The CUDA venv is auto-built by the job.
uv run python experiments/eval_cutoff_sweep_all.py \
    --sweep-dir results/sweeps/<sweep>_<ts>/ --launch slurm
```

All cluster scripts `source scripts/setup_cuda_env.sh` (after `cd
"$HOME/CV-Quixer"`), which installs/repairs `uv` and **auto-builds the CUDA venv —
no manual pre-build**. Both are **architecture-keyed** by `$(uname -m)`
(`~/.local/uv/<arch>/uv`, `~/.venvs/cv-quixer-cuda-<arch>`), so a job array
spanning mixed hardware (x86_64 / aarch64) never shares an incompatible binary or
venv. The venv build is serialised with an `flock` (the first array task builds,
the rest wait, then reuse). The helper also **repairs a broken/wrong-arch `uv`**
(it checks `uv --version`, not just presence — the fix for
`cannot execute binary file: Exec format error`). Force a clean venv rebuild with
`REBUILD_VENV=1` (e.g. `sbatch --export=ALL,REBUILD_VENV=1 …`). Experiment calls
use `uv run --no-sync` since the helper already synced.

All SLURM `.out`/`.err` logs and the per-task `gpu_util-*.csv` files are written
to `slurm_logs/` (kept in git via `slurm_logs/.gitkeep`; the logs themselves are
gitignored) instead of cluttering the repo root. Submit jobs from the repo root
so the relative `--output`/`--error` paths resolve — SLURM does **not** create the
directory, so it must already exist (the `.gitkeep` ensures it does after a pull).

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
- `scripts/setup_cuda_env.sh` builds a per-arch venv at
  `$HOME/.venvs/cv-quixer-cuda-$(uname -m)` (built once, reused; `REBUILD_VENV=1`
  forces a clean rebuild). `UV_PROJECT_ENVIRONMENT` (set by the helper) overrides
  uv's default `.venv` location; uv itself installs per-arch to
  `~/.local/uv/$(uname -m)` via `UV_INSTALL_DIR`.
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
cat slurm_logs/slurm-cuda_triage-<JOBID>.out   # view triage output (logs → slurm_logs/)
```

## Package structure

```
cv_quixer/
├── config/         schema.py: ExperimentConfig/Quantum/Data/Training dataclasses
│                   + ObservableSpec. observable_presets.py: named ObservableSpec
│                   sets + resolve_observables(). utils.py: load_config() YAML→config
│                   (LEGACY, see below) and save_config() config→JSON
├── data/           MNIST/FashionMNIST download, patch extraction, DataLoader
├── models/
│   ├── base.py         BaseVisionTransformer (shared interface)
│   ├── classical/      Classical ViT wrapper
│   └── quantum/        CV quantum model (core thesis contribution)
│       ├── cv_attention.py  CNNHypernetwork, LCUSumCoefficients,
│       │                    PolynomialCoefficients, HyperCVAttentionHead,
│       │                    HyperCVAttention + truncation loss helpers
│       ├── cv_quixer.py     CVDecoder, CVQuixer (main model),
│       │                    param-count auto-scaling → utils.params.autoscale_to_target
│       └── cv_seq2seq.py    Seq-to-seq stacked model (model="quantum_stacked",
│                            ADR-0002): Seq2SeqCNNHead/Seq2SeqLinearHead,
│                            Seq2SeqCVAttention blocks, StackedCVQuixer
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
├── eval_cutoff_sweep_all.py  Fan eval_cutoff_sweep over EVERY run in a sweep (manifest + local/slurm)
├── sweep.py               Fan a (param × observable × seed) grid into full_experiment runs
├── report_sweep.py        Cross-run sweep table (summary.csv/md) + plots + per-run diagnostics
├── report_sweep_compare.py  Overlay ≥2 sweeps (e.g. quantum vs quantum_shared) → combined table + cross-sweep figures
├── report_cutoff_sweep.py Cross-run cutoff table + figures/cutoff/ + per-cutoff diagnostics
├── report_diagnostics.py  All figures (training curves + diagnostics) from a full_experiment run
└── migrate_add_cvqnn_field.py  One-shot: bake cvqnn_num_layers=0 into pre-W run config.json (so old checkpoints reload as pre-W models)

scripts/
├── run_mini_experiment.sh        SLURM batch job for mini_experiment
├── run_full_experiment.sh        SLURM batch job for full_experiment (A100, 3 h)
├── run_eval_cutoff_sweep.sh      SLURM batch job for eval_cutoff_sweep (A100, 12 h)
├── run_eval_cutoff_sweep_array.sh SLURM job ARRAY for eval_cutoff_sweep_all manifests (A100, one task/run)
├── run_sweep.sh                  SLURM job ARRAY for sweep.py manifests (A100, one task/run)
├── triage_cuda.sh                SLURM GPU/CUDA diagnostic job
└── debug_imports.py              Sequential import diagnostics with tracebacks

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
          # output dim: _op_plan_param_count(_build_op_plan(num_layers), m, topology)
          # = L·(8m−2) + (L−1)·(3m−2) linear  [= 8m−2 when num_layers L=1]
        Apply depth-L unitary U_i (L stacked layers, L−1 interferometers between):
          [ Squeeze(r,φ) → Beamsplitter mesh(θ,φ) → Rotate(φ) → Displace(α) → Kerr(κ) ]   ×L
          interleaved with  [ Beamsplitter mesh(θ,φ) → Rotate(φ) ]   ×(L−1)
    post-select renorm: |ψ⟩ ← P(M)|ψ⟩ / ‖P(M)|ψ⟩‖       # heralded QSVT output
    CVQNN block W:  |ψ⟩ ← W|ψ⟩ / ‖W|ψ⟩‖                   # fixed per-head Killoran circuit
      # W = [ Beamsplitter(θ,φ) → Rotate(φ) → Squeeze(r,φ) → Beamsplitter(θ,φ)
      #       → Rotate(φ) → Displace(α) → Kerr(κ) ] × L_W   (canonical 2-interferometer form)
      # owned nn.Parameters (input-independent), small-random init (W≈I); L_W=0 ⇒ no W
      # leakage 1−‖W|ψ⟩‖² tracked separately and penalised by cvqnn_trunc_lambda
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
- **Model factory**: `cv_quixer.models.build_model(config)` is the only place the model
  string is resolved to a class: `"quantum"` → `CVQuixer` (per-head CNN hypernetworks),
  `"quantum_shared"` → `SharedCVQuixer` (one shared patch CNN + per-head linears),
  `"quantum_stacked"` → `StackedCVQuixer` (seq-to-seq blocks, ADR-0002),
  `"classical"` → `ClassicalViT`.
- **Seq-to-seq stacked variant** (`model="quantum_stacked"`, `StackedCVQuixer`,
  ADR-0002): `num_seq2seq_blocks` uniform **seq-to-seq blocks** — per head, all N
  positions share one LCU `M`, polynomial `P`, and CVQNN block `W`; position i's
  output token is the readout of `W·P(M)·U_{q,i}|0⟩`, where the **query unitary**
  `U_{q,i}` is a second slice of the same hypernet output that emits `U_i`
  (key/query projections of one shared patch embedding; the hypernet's final
  linear emits `2×` the op-plan width). The query state is renormalised before
  `P(M)`; its leakage is the separate `query_trunc_loss` stream (weighted by
  `query_trunc_lambda`), and the per-patch trunc penalty measures LCU-term
  leakage on the *actual input states* (degenerates to the vacuum form
  elsewhere). Block 1 uses per-head CNN hypernets on raw patches; blocks ≥ 2 use
  per-head `Linear(H×R → 2×gate_params)` over tokens, with identity residuals
  from block 2 on (`block_residual`). The stack ends in mean-pooling
  (`pooling="mean"`) or a canonical seq-to-one **aggregator block**
  (`pooling="quixer"`, `LinearCVHead`s — not counted by `num_seq2seq_blocks`);
  either way the decoder input is `H×R`, identical to the canonical models.
  Engine stays iterative (no LCU matrix): the vmap nesting gains a query axis
  (head → batch → query → patch) at ~`d·N²` gate-plan applications per head per
  block (~N× the canonical model, × blocks). Diagnostics npz keys are
  block-prefixed (`block{b}_head{h}_{gate}`, `…_q_{gate}` for the query slice,
  `agg_…`, `block{b}_lcu_coeffs`, …); state norms / photon numbers keep the
  canonical keys and describe the decoder-input stage. A new, never-trained
  model — no checkpoint-compat constraints with `quantum`/`quantum_shared`.
- **Shared-CNN variant** (`model="quantum_shared"`, `SharedCVQuixer`): a single
  `SharedPatchCNN` embeds each patch once (`Conv→Conv→flatten→+2D-PE`, shared across heads);
  it emits the flattened conv features directly (width `cnn_channels_2 × h_out²`, **no
  projection layer**). Each head is then just `Linear(feature_dim → gate_params)` feeding its
  quantum circuit (`LinearCVHead`). The conv stack runs once per forward, not once per head.
  LCU/poly coefficients stay per-head. The circuit core (`_apply_lcu_to_vector`,
  `_apply_polynomial_iterative`, readout, etc.) is shared with the canonical head via
  `_CVHeadBase`, and the model `forward` via `_CVQuixerBase`, so `CVQuixer` is unchanged.
  Selected via `full_experiment.py --model quantum_shared` (which also defaults
  `scaling_knob` to `num_heads`; `num_heads` then auto-scales to ~5 heads at the 13,760
  budget). **Not checkpoint-compatible with `quantum`** — different parameter structure.
- **Config system**: Experiment scripts construct the `ExperimentConfig`
  dataclasses directly in Python (no YAML at runtime). `full_experiment.py`
  writes the fully resolved config to `config.json` in its run directory;
  `eval_cutoff_sweep.py`, `report_diagnostics.py`, and `backfill_artefacts.py`
  reconstruct `ExperimentConfig` from that saved `config.json` via the shared
  `config.utils.experiment_config_from_dict()` helper (dacite + the CVQNN guard
  below). The `load_config()` YAML loader and `configs/*.yaml` are kept but
  currently unused (intended for a future revival of YAML-driven runs).
- **CVQNN block W + pre-W migration**: the canonical model now applies a fixed
  per-head Killoran circuit `W` to the post-polynomial state before readout
  (`cvqnn_num_layers=1` default; see `QuantumConfig` reference and ADR-0001).
  This is **checkpoint-incompatible with pre-W runs** when `W` is on; the frozen
  13,530-param baseline is retired (re-run to re-establish numbers with `W`).
  `cvqnn_num_layers=0` reproduces the pre-W model exactly (byte-identical
  state_dict). Old runs load via a one-shot **migration** that bakes
  `cvqnn_num_layers: 0` / `cvqnn_trunc_lambda: 0.0` into their `config.json`:
  `uv run python experiments/migrate_add_cvqnn_field.py --runs-root results/`.
  `experiment_config_from_dict()` carries a **loud guard** that *raises* (with a
  hint to run the migration) on an un-migrated config rather than silently
  defaulting `W` on — deliberately no silent "absence means 0" shim.
- **Pure PyTorch simulation**: `cv_quixer/quantum/` is a standalone Fock-basis circuit
  simulator. Gaussian gates (rotation, displacement, squeezing, beamsplitter) use exact
  analytic Fock-basis formulas (true sub-isometries: column norms ≤ 1); only cubic phase
  uses `matrix_exp`. Diagonal gates (rotation via `rotation_phases`, Kerr via
  `kerr_phases`) return `(D,)` phase vectors applied via `apply_single_mode_phases`
  (O(D) vs O(D³) matrix-exp). No PennyLane at training time. Gradient mode is `backprop`
  (autograd through einsum chain) or `parameter_shift` (PSR, deferred).
- **vmap head/batch parallelism**: `HyperCVAttention.forward` uses nested
  `torch.func.vmap` + `functional_call` — an outer vmap over the head axis nests
  over the batch vmap, which nests over the patch vmap (head → batch → patch),
  replacing the former sequential Python `for head in self.heads` and `for b in
  range(B)` loops. Heads are stacked with plain differentiable `torch.stack`
  (not `stack_module_state`, which detaches and would strand head gradients), so
  the `nn.ModuleList` and checkpoint keys are unchanged. All inner ops are
  out-of-place. `FockState.vacuum` uses `index_put` (not in-place setitem);
  diagonal gates use `apply_single_mode_phases` with broadcasting (not
  `torch.diag` + einsum) for vmap compatibility.
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
| `num_layers` | 1 | Per-patch circuit depth L: each token's unitary U_i is L stacked Killoran gate sequences with L−1 `BS→Rot` interferometers interleaved between them (layer 1 has no leading interferometer — it acts trivially on the vacuum). L=1 reproduces the single-layer model exactly (checkpoint-compatible). Params are hypernetwork-emitted, so larger L widens the hypernet output linear → more params |
| `cutoff_dim` | 6 | Fock space truncation D |
| `grad_mode` | `"backprop"` | `"backprop"` or `"parameter_shift"` (PSR deferred) |
| `param_shift_shift` | 1.5708 | PSR shift `s` (π/2); only used when `grad_mode="parameter_shift"` |
| `bs_topology` | `"linear"` | Beamsplitter mesh: `"linear"` or `"ring"` |
| `dtype` | `"complex128"` | `"complex64"` or `"complex128"` |
| `num_heads` | 4 | Parallel CV attention heads |
| `decoder_hidden_dim` | 64 | CVDecoder hidden layer width |
| `decoder_hidden_mult` | `None` | If set (float > 0), size the decoder hidden width relative to its input: `decoder_hidden_dim = max(1, round(mult × in_dim))`, `in_dim = num_heads × readout_width`. Resolved in the model `__init__` after any `target_params` num_heads scaling (idempotent on reload); overrides `decoder_hidden_dim`. `None` = off |
| `cnn_channels_1` | 8 | CNNHypernetwork first conv output channels |
| `cnn_channels_2` | 16 | CNNHypernetwork second conv output channels (a valid `scaling_knob`, but `num_heads` is the default) |
| `cnn_kernel_size` | 3 | Conv kernel size |
| `cnn_num_conv_layers` | 2 | Total conv layers in the CNN stack. 2 = historic `conv1→conv2`; >2 appends `(n−2)` same-padding `C2→C2` convs (preserve `h_out`/`feature_dim`). Additive (empty `ModuleList` at 2 ⇒ unchanged state-dict keys). A valid `scaling_knob` |
| `hypernet_num_linear_layers` | 1 | Total Linear layers in the hypernet DNN (after the 2D-PE add). 1 = historic single `feature_dim→gate_params`; >1 prepends `(n−1)` `feature_dim→feature_dim` Tanh blocks. Additive (empty at 1 ⇒ unchanged keys). A valid `scaling_knob` |
| `decoder_num_layers` | 2 | Total Linear layers in the CVDecoder MLP. 2 = historic `Linear→ReLU→Linear` (keys `net.0`/`net.2`); >2 inserts `(n−2)` `h→h` ReLU blocks. A valid `scaling_knob` |
| `poly_degree` | 2 | Matrix polynomial degree (keep ≤ 4) |
| `cvqnn_num_layers` | 1 | CVQNN block W depth L_W: a fixed, per-head, trainable Killoran circuit applied to the post-polynomial (post-selected) state before readout. Each layer is the canonical two-interferometer form `(BS→R)→S→(BS→R)→D→K` (the per-patch `U_i` drops the leading interferometer, trivial on its vacuum input). Owned `nn.Parameter`s, small-random init (W≈I). **`0` disables W** → state_dict byte-identical to a pre-W model (ablation / checkpoint-compat baseline). A valid `scaling_knob` but too coarse for budget targeting |
| `cvqnn_trunc_lambda` | 0.01 | Weight of the **separate** W truncation penalty `1 − ‖W|ψ⟩‖²` added to the training loss (independent of `trunc_lambda`/`trunc_penalty` — W's single-block leakage compounds far less than the per-patch penalty does through the polynomial powers, hence a lighter default). Free to compute (it is the norm used for the post-W renorm). `0` → tracked but not penalised |
| `target_params` | -1 | If > 0, binary-search the configured `scaling_knob` (build-and-count) to hit this count (warns if the closest achievable is >10% off — `tol=0.10` in `utils/params.py`) |
| `scaling_knob` | `"num_heads"` | Integer QuantumConfig field auto-scaled toward `target_params` (e.g. `num_heads`, `cnn_channels_2`, `num_modes`, `num_layers`, the three depth fields above — all monotonic in param count). `num_heads` is the default (robust across budgets; `cnn_channels_2` accuracy degrades with scale). The build-and-count search in `utils/params.autoscale_to_target` tolerates knobs with a minimum > 1 (e.g. `decoder_num_layers ≥ 2`): out-of-range trials are skipped, never crash |
| `trunc_penalty` | `"none"` | `"none"`, `"norm"`, or `"photon_number"` |
| `trunc_lambda` | 0.01 | Truncation penalty loss weight |
| `num_seq2seq_blocks` | 1 | (`quantum_stacked` only, ADR-0002) Number of uniform seq-to-seq blocks; ≥ 1 enforced (a 0-block model would duplicate the existing CVQuixer). Does **not** count the optional aggregator block. A valid (monotonic) `scaling_knob`, but coarse |
| `pooling` | `"mean"` | (`quantum_stacked` only) `"mean"` pools the final tokens over positions; `"quixer"` appends a canonical seq-to-one aggregator block. Both end at the same `H×R` decoder width |
| `block_residual` | `True` | (`quantum_stacked` only) Identity residual `x + block(x)` from block 2 onward (block 1's widths differ). `False` = pure-pipeline ablation |
| `query_trunc_lambda` | 0.01 | (`quantum_stacked` only) Weight of the **separate** query-unitary truncation penalty `mean_i(1 − ‖U_{q,i}\|0⟩‖²)`. Single-application leakage like W's (fires once per position, before the polynomial), hence the lighter default vs the compounding per-patch `trunc_lambda`. `0` → tracked but not penalised |
| `gate_param_bound` | `None` | Soft-clip magnitude gate params (squeeze r, displacement re/im) to `(-b, b)` via `b·tanh(x/b)` — stops gates driving the state far past the cutoff (the NaN-head cause at high `num_heads`). `None` = off (not checkpoint-compatible with a bounded model). `full_experiment.py --gate-param-bound auto` → `auto_gate_bound(cutoff)=asinh(√(cutoff-1))` (the representable photon budget; ≈1.54 at cutoff 6). Avoid large b (~4 → ⟨n⟩≈745, the degenerate regime). |
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
| `TARGET_PARAMS` | 13,760 | Parameter budget (auto-scales `num_heads`, the default knob) |
| `CHECKPOINT_INTERVAL` | 10 | Save checkpoint every N epochs |

### full_experiment.py constants

Default is the frozen best p12800 L2 sweep point **plus the CVQNN block W**
(`cvqnn_num_layers=1`): num_modes=2, cutoff_dim=6, num_heads=4, num_layers=2,
poly_degree=3, cnn_channels_2=8, trunc norm λ=0.1, `xpxsps` observables
(`TARGET_PARAMS=-1` so the architecture is built verbatim, no auto-scaling).
Adding `W` makes this **checkpoint-incompatible with pre-W runs** and shifts the
param count off the historic 13,530 (set `--cvqnn-num-layers 0` to reproduce the
exact pre-W model). Trained on the full 60k/10k split. CLI-overridable:
`--epochs`, `--train-fraction`, `--test-fraction`, `--resume`, `--target-params`
(re-enables auto-scaling on `--scaling-knob`, default `num_heads`),
`--cvqnn-num-layers`, `--cvqnn-trunc-lambda`.

| Constant | Value | Description |
|---|---|---|
| `EPOCHS` | 3 | Default epochs (~75-90 min/epoch V100, ~30-45 min/epoch A100) |
| `BATCH_SIZE` | 64 | Batch size |
| `TARGET_PARAMS` | -1 | Auto-scaling off → frozen architecture. Set >0 to binary-search `SCALING_KNOB` |
| `SCALING_KNOB` | `"num_heads"` | Knob auto-scaled toward `--target-params` when enabled |
| `NUM_LAYERS` | 2 | Per-patch circuit depth L |
| `TRUNC_LAMBDA` | 0.1 | Fock truncation penalty weight (per-patch) |
| `CVQNN_NUM_LAYERS` | 1 | CVQNN block W depth L_W (0 = disabled / pre-W ablation) |
| `CVQNN_TRUNC_LAMBDA` | 0.01 | Weight of the separate W truncation penalty |
| `CHECKPOINT_INTERVAL` | 1 | Versioned `epoch_NNNN.pt` every N epochs |
| `MA_WINDOW` | 50 | Moving-average window for per-batch plots |

## Simulation notes

Fock backend memory scales as `cutoff_dim ^ num_modes`. Keep `num_modes ≤ 8` and
`cutoff_dim ≤ 10` for tractable single-GPU simulation.

**mini_experiment config**: `num_modes=2`, `cutoff_dim=6`, `num_heads=4`,
`patch_size=7` (16 patches), `dtype=complex64`. `target_params=13760` auto-scales
`num_heads` (the default knob) via binary search at init time.

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
| `history.json` | Training-time log only — per-epoch `train_loss`, `train_acc`, `test_loss`, `test_acc`, `trunc_loss`, `test_trunc_loss`, `elapsed_sec`, plus `history["batch"]` (per-minibatch arrays) and `history["meta"]` (best_epoch, runtime, plus sweep coordinates `target_params`/`achieved_params`/`observables_name`/`scaling_knob`/`num_layers`/`trunc_lambda`/`seed`/`model`). Never the canonical source for figures. |
| `parameter_table.txt` | Snapshot of `print_parameter_table()` |
| `checkpoints/` | `latest.pt` (every epoch), `best.pt` (best test acc), `final_model.pt`, `epoch_NNNN.pt` |
| `figures/` | Populated by `report_diagnostics.py` (run post-hoc or partway through). Not written by `full_experiment.py` itself. |
| `predictions/epoch_NNNN.npz` | Test side per epoch: `y_true`, `y_pred`, `y_probs`, `readouts`. Canonical source for accuracy/loss/per-class/confusion/calibration figures. |
| `predictions/epoch_NNNN_train.npz` | Train side per epoch: same four keys, from the clean post-epoch train eval. Enables train-side per-class/confusion/embedding figures. |
| `predictions/test_images.npz` | One-time, reassembled `(N, H, W)` test images for the misclassification gallery. |
| `diagnostics/epoch_NNNN.npz` | Per-epoch raw quantum diagnostics: gate-param samples (`head{h}_{gate}`), state norms (`head{h}_state_norms`), `mean_photon_number`, `lcu_coeffs`, `poly_coeffs`. |
| `subset_indices.npz` | Absolute train/test/diag indices into the full PatchedDataset, written once. |
| `logs/train.log` | Tee'd stdout from the training process. |

**`eval_cutoff_sweep.py`**: writes `<run_dir>/eval/cutoff_sweep_<timestamp>/`
containing `results.json`, `results.csv`, `meta.json`, per-metric plots, and one
`D{NN}/` report_diagnostics-runnable sub-run dir per cutoff.

**`eval_cutoff_sweep_all.py` / `report_cutoff_sweep.py`** (whole-sweep cutoff
eval, written into the sweep dir):

| Entry | Contents |
|---|---|
| `cutoff_sweep_manifest.json` | Per-run `eval_cutoff_sweep.py` argv (one shared `--output-name` across runs), consumed by `run_eval_cutoff_sweep_array.sh`. |
| `<run>/eval/<shared-name>/` | The per-run `eval_cutoff_sweep.py` output (layout above), one per run, all under the same `cutoff_sweep_<ts>` name. |
| `cutoff_summary.csv` / `.md` | Written by `report_cutoff_sweep.py` — one row per (run, split, cutoff). |
| `figures/cutoff/*.png` | Written by `report_cutoff_sweep.py` — cross-run acc/trunc-vs-cutoff, acc-recovery-vs-params, acc-vs-params-by-cutoff. |

**`sweep.py`** (one directory per sweep,
`results/sweeps/<sweep_name>_<YYYY-MM-DD_HH-MM-SS>/`):

| Entry | Contents |
|---|---|
| `sweep_manifest.json` | Grid axes, per-run `run_name`, and the exact `full_experiment.py` argv for each grid point (consumed by `run_sweep.sh`). |
| `p{params}__obs-{preset}__seed{seed}/` | One full `full_experiment.py` run dir per grid point (same layout as above). |
| `summary.csv` / `summary.md` | Written by `report_sweep.py` — one row per run (model, target/achieved params, observable preset, scaling knob, best/final acc, runtime). |
| `figures/acc_vs_params.png`, `figures/acc_by_observable.png` | Written by `report_sweep.py` — cross-run comparison plots. |

**`report_sweep_compare.py`** (cross-sweep overlay; default out-dir
`results/sweeps/compare_<YYYY-MM-DD_HH-MM-SS>/`, override with `--out-dir`).
Reuses `report_sweep`'s loaders; reads each run's `model` from `history["meta"]`
or, for older runs, `config.json` (default `"quantum"`). Series key on
`(model, observables, scaling_knob)` so different knobs/models are never averaged
together:

| Entry | Contents |
|---|---|
| `comparison.csv` / `comparison.md` | One row per run across all `--sweep-dir`s, tagged with `sweep_label` + `model`. |
| `figures/acc_vs_params_compare.png` | Best test acc vs achieved params (colour = model, marker = scaling knob). |
| `figures/acc_by_params_compare.png` | Grouped bars: model/knob at each target budget (bar labels = achieved params). |

## Agent skills

### Issue tracker

Issues and PRDs are tracked as GitHub issues via the `gh` CLI (repo `clydelhui/CV-Quixer`). See `docs/agents/issue-tracker.md`.

### Triage labels

Canonical default labels — `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one `CONTEXT.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.

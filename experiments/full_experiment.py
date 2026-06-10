"""Full FashionMNIST experiment for CV-Quixer.

Default quantum config is the frozen best p12800 L2 sweep point (num_modes=2,
cutoff_dim=6, num_heads=4, num_layers=2, poly_degree=3, cnn_channels_2=8 →
13,530 params; target_params=-1 so the architecture is built verbatim, no
auto-scaling), trained on the **full** 60k train / 10k test FashionMNIST split.
Pass --target-params N to re-enable auto-scaling (scaling knob defaults to
num_heads). Default 3 epochs; ~75-90 min/epoch on V100, ~30-45 min/epoch on A100.

Per-batch metrics (CE loss, trunc loss, total loss, batch accuracy, gradient L2 norm)
are logged alongside per-epoch metrics so the loss/trunc/accuracy/gradient curves
have ~1800-2800 data points instead of 2-3.

Run directory layout (one self-contained folder per run):

    results/runs/full_fashionmnist_YYYY-MM-DD_HH-MM-SS/
    ├── config.json                  # full resolved ExperimentConfig
    ├── history.json                 # epoch + batch + meta metrics (plot source of truth)
    ├── parameter_table.txt          # snapshot of print_parameter_table()
    ├── checkpoints/
    │   ├── latest.pt                # overwritten every epoch (resume safety)
    │   ├── best.pt                  # best test-acc snapshot
    │   ├── final_model.pt           # written once at end of training
    │   └── epoch_NNNN.pt            # versioned per-epoch checkpoint
    └── logs/

Figures are produced separately by `experiments/report_diagnostics.py` —
run it against the run directory (post-hoc or partway through) to populate
`figures/` from `history.json` + the saved npz files.

Run (fresh):
    uv run python experiments/full_experiment.py

Resume from a checkpoint (continues writing into the same run directory):
    uv run python experiments/full_experiment.py \\
        --resume results/runs/full_fashionmnist_2026-05-15_14-30-00/checkpoints/latest.pt

Local smoke test (reduced scale):
    uv run python experiments/full_experiment.py \\
        --epochs 1 --train-fraction 0.1 --test-fraction 0.1
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from cv_quixer.config.observable_presets import PRESET_NAMES, resolve_observables
from cv_quixer.config.schema import (
    DataConfig,
    ExperimentConfig,
    ObservableSpec,
    QuantumConfig,
    TrainingConfig,
    auto_gate_bound,
)
from cv_quixer.data.mnist import PatchedDataset
from cv_quixer.evaluation.diagnostics import (
    ensure_history_schema,
    evaluate,
    new_history,
    quantum_diagnostics,
    save_test_images_once,
    snapshot_coefficients,
    snapshot_cvqnn_params,
    snapshot_stacked_coefficients,
)
from cv_quixer.models import build_model
from cv_quixer.utils import print_parameter_table
from cv_quixer.utils.logging import finish_logging, init_logging, log_metrics

# ---------------------------------------------------------------------------
# Defaults (CLI-overridable below)
# ---------------------------------------------------------------------------

EPOCHS = 3
BATCH_SIZE = 64
# Default = the frozen 13,530-param model (the best p12800 L2 sweep point:
# num_heads=4, cnn_channels_2=8). TARGET_PARAMS=-1 disables auto-scaling so the
# explicit architecture below is built verbatim. Pass --target-params N to
# re-enable the binary search, which then scales SCALING_KNOB (num_heads).
TARGET_PARAMS = -1
SCALING_KNOB = "num_heads"  # knob auto-scaled to hit --target-params (when enabled)
NUM_LAYERS = 2  # per-patch circuit depth L (L gate sequences + L-1 BS→Rot interferometers)
CUTOFF_DIM = 6  # Fock truncation (also used to expand the `pnr` observable preset)
SEED = 42
CHECKPOINT_INTERVAL = 1  # versioned epoch_NNNN.pt every N epochs
DEFAULT_OBSERVABLES = "xpxsps"  # x, p, x², p² per mode
TRUNC_LAMBDA = 0.1  # Fock truncation penalty weight (added to CE loss)
CVQNN_NUM_LAYERS = 1   # CVQNN block W depth L_W (0 = disabled / pre-W ablation)
CVQNN_TRUNC_LAMBDA = 0.01  # weight of the separate W truncation penalty (added to CE loss)
# Seq-to-seq stacked model (--model quantum_stacked only; ADR-0002).
NUM_SEQ2SEQ_BLOCKS = 1     # seq-to-seq blocks (excludes the optional aggregator)
POOLING = "mean"           # "mean" | "quixer" (append an aggregator block)
BLOCK_RESIDUAL = True      # identity residual x + block(x) from block 2 onward
QUERY_TRUNC_LAMBDA = 0.01  # weight of the separate query-unitary truncation penalty
# Architecture knobs of the frozen 13,530-param model. Each is overridable by the
# matching CLI flag below for manual (no-auto-scale) runs / sweeps; defaults here
# reproduce the frozen architecture verbatim.
NUM_MODES = 2
NUM_HEADS = 4              # resolved value of the frozen model (auto-scaled only with --target-params)
CNN_CHANNELS_1 = 8
CNN_CHANNELS_2 = 8         # resolved value of the frozen model
CNN_KERNEL_SIZE = 3
DECODER_HIDDEN_DIM = 32
DECODER_HIDDEN_MULT = None  # if set (float >0), decoder_hidden_dim = round(mult * decoder_in_dim)
POLY_DEGREE = 3
CNN_NUM_CONV_LAYERS = 2          # total conv layers in the CNN stack
HYPERNET_NUM_LINEAR_LAYERS = 1   # total Linear layers in the hypernet DNN
DECODER_NUM_LAYERS = 2           # total Linear layers in the decoder MLP


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument(
    "--resume",
    type=str,
    default=None,
    help="path to checkpoint .pt file to resume from. The "
    "parent run directory is reused so the run continues "
    "writing into the same folder.",
)
parser.add_argument(
    "--epochs", type=int, default=None, help=f"override the default EPOCHS={EPOCHS}"
)
parser.add_argument(
    "--train-limit",
    type=int,
    default=None,
    help="cap training set to first N samples (deterministic, "
    "smoke-test only). Mutually exclusive with --train-fraction.",
)
parser.add_argument(
    "--test-limit",
    type=int,
    default=None,
    help="cap test set to first N samples (deterministic, "
    "smoke-test only). Mutually exclusive with --test-fraction.",
)
parser.add_argument(
    "--train-fraction",
    type=float,
    default=None,
    help="random subset of the train set (0 < x <= 1). "
    "Mutually exclusive with --train-limit.",
)
parser.add_argument(
    "--test-fraction",
    type=float,
    default=None,
    help="random subset of the test set (0 < x <= 1). "
    "Mutually exclusive with --test-limit.",
)
parser.add_argument(
    "--subset-seed",
    type=int,
    default=42,
    help="seed shared by --train-fraction and --test-fraction random subsets",
)
# --- sweep axes -----------------------------------------------------------
parser.add_argument(
    "--target-params",
    type=int,
    default=None,
    help=f"parameter budget; auto-scales the configured --scaling-knob "
    f"(default TARGET_PARAMS={TARGET_PARAMS} → auto-scaling off, the frozen "
    f"13,530-param architecture is built as-is)",
)
parser.add_argument(
    "--scaling-knob",
    type=str,
    default=None,
    help=f"integer QuantumConfig field to auto-scale toward --target-params "
    f"(default {SCALING_KNOB!r}; only used when --target-params is set); "
    f"e.g. num_heads, cnn_channels_2, num_modes",
)
parser.add_argument(
    "--num-layers",
    type=int,
    default=None,
    help="per-patch circuit depth L (default 1): L stacked gate sequences with "
    "L-1 BS→Rot interferometers between them. Deepening L raises the param "
    "count, so it may also be used as a --scaling-knob.",
)
# --- direct architecture knobs (manual mode; override the frozen-model values) ---
# Each defaults to None → inherit the frozen-model constant above. Set any of
# these to build an explicit architecture without --target-params auto-scaling
# (they are honoured even when --target-params is set, except the chosen
# --scaling-knob, which the binary search overrides).
parser.add_argument("--num-modes", type=int, default=None,
                    help=f"bosonic modes (default {NUM_MODES})")
parser.add_argument("--cutoff-dim", type=int, default=None,
                    help=f"Fock cutoff D (default {CUTOFF_DIM}); also sizes the "
                    "pnr/xpxsps_pnr observable plans and 'auto' gate bound")
parser.add_argument("--num-heads", type=int, default=None,
                    help=f"parallel CV attention heads (default {NUM_HEADS})")
parser.add_argument("--cnn-channels-1", type=int, default=None,
                    help=f"first conv output channels (default {CNN_CHANNELS_1})")
parser.add_argument("--cnn-channels-2", type=int, default=None,
                    help=f"second conv output channels (default {CNN_CHANNELS_2})")
parser.add_argument("--cnn-kernel-size", type=int, default=None,
                    help=f"conv kernel size (default {CNN_KERNEL_SIZE})")
parser.add_argument("--decoder-hidden-dim", type=int, default=None,
                    help=f"decoder MLP hidden width (default {DECODER_HIDDEN_DIM}); "
                    "mutually exclusive with --decoder-hidden-mult")
parser.add_argument("--decoder-hidden-mult", type=float, default=None,
                    help="size the decoder hidden width relative to its input as "
                    "round(mult * decoder_in_dim), where decoder_in_dim = num_heads * "
                    "readout_width (resolved after any --target-params scaling). "
                    "Mutually exclusive with --decoder-hidden-dim")
parser.add_argument("--poly-degree", type=int, default=None,
                    help=f"matrix polynomial degree d (default {POLY_DEGREE})")
parser.add_argument("--cnn-num-conv-layers", type=int, default=None,
                    help=f"total conv layers in the CNN stack (default "
                    f"{CNN_NUM_CONV_LAYERS}); >2 appends same-padding C2→C2 convs")
parser.add_argument("--hypernet-num-linear-layers", type=int, default=None,
                    help=f"total Linear layers in the hypernet DNN (default "
                    f"{HYPERNET_NUM_LINEAR_LAYERS}); >1 prepends feature→feature blocks")
parser.add_argument("--decoder-num-layers", type=int, default=None,
                    help=f"total Linear layers in the decoder MLP (default "
                    f"{DECODER_NUM_LAYERS}); >2 inserts extra hidden blocks")
parser.add_argument("--cvqnn-num-layers", type=int, default=None,
                    help=f"CVQNN block W depth L_W (default {CVQNN_NUM_LAYERS}); "
                    f"0 disables W (pre-W ablation, checkpoint-compatible). A "
                    f"valid --scaling-knob, but too coarse for budget targeting.")
# --- seq-to-seq stacked model knobs (--model quantum_stacked only) ---------
parser.add_argument("--num-seq2seq-blocks", type=int, default=None,
                    help=f"seq-to-seq blocks in the stacked model (default "
                    f"{NUM_SEQ2SEQ_BLOCKS}; >= 1; excludes the optional "
                    f"aggregator block). A valid --scaling-knob, but coarse.")
parser.add_argument("--pooling", type=str, default=None,
                    choices=["mean", "quixer"],
                    help=f"how the stacked model's final tokens reach the "
                    f"decoder (default {POOLING!r}): 'mean' pools over "
                    f"positions; 'quixer' appends a canonical seq-to-one "
                    f"aggregator block.")
parser.add_argument("--block-residual", type=str, default=None,
                    choices=["on", "off"],
                    help=f"identity residual x + block(x) from block 2 onward "
                    f"(default {'on' if BLOCK_RESIDUAL else 'off'}; 'off' = "
                    f"pure-pipeline ablation).")
parser.add_argument(
    "--query-trunc-lambda",
    type=float,
    default=None,
    help=f"weight of the separate query-unitary truncation penalty added to "
    f"the CE loss (default QUERY_TRUNC_LAMBDA={QUERY_TRUNC_LAMBDA}; stacked "
    f"model only)",
)
parser.add_argument(
    "--observables",
    type=str,
    default=None,
    choices=PRESET_NAMES,
    help="named observable readout preset (default 'xpxsps'). "
    "Mutually exclusive with --observables-json.",
)
parser.add_argument(
    "--observables-json",
    type=str,
    default=None,
    help="ad-hoc observable spec as a JSON list of "
    '{"type","mode","n"} objects, e.g. \'[{"type":"x","mode":"all"}]\'. '
    "Requires --run-name. Mutually exclusive with --observables.",
)
parser.add_argument(
    "--seed",
    type=int,
    default=None,
    help="training seed (default 42); vary for repeat runs / error bars",
)
# --- model selection ------------------------------------------------------
parser.add_argument(
    "--model",
    type=str,
    default="quantum",
    choices=["quantum", "quantum_shared", "quantum_stacked", "classical"],
    help="model variant: 'quantum' (per-head CNN hypernetworks, default), "
    "'quantum_shared' (shared patch CNN + per-head linears; defaults "
    "--scaling-knob to num_heads), 'quantum_stacked' (seq-to-seq blocks, "
    "ADR-0002), or 'classical'.",
)
# --- run organisation -----------------------------------------------------
parser.add_argument(
    "--run-name",
    type=str,
    default=None,
    help="explicit run-directory name (default full_fashionmnist_<timestamp>). "
    "Sweeps pass an encoded name like p13760__obs-xpxsps__seed42.",
)
parser.add_argument(
    "--runs-root",
    type=str,
    default="results/runs",
    help="parent directory for the run (default results/runs; "
    "sweeps pass results/sweeps/<sweep_name>)",
)
# --- experiment tracking --------------------------------------------------
parser.add_argument(
    "--wandb",
    action="store_true",
    help="log metrics to Weights & Biases (set WANDB_MODE=offline on "
    "nodes without network)",
)
parser.add_argument(
    "--wandb-group",
    type=str,
    default=None,
    help="wandb run group (pass the sweep name to group sweep runs)",
)
parser.add_argument(
    "--wandb-tags",
    type=str,
    nargs="*",
    default=None,
    help="extra wandb tags (params/obs/seed tags are added automatically)",
)
parser.add_argument(
    "--trunc-lambda",
    type=float,
    default=None,
    help=f"Fock truncation penalty weight added to the CE loss "
    f"(default TRUNC_LAMBDA={TRUNC_LAMBDA})",
)
parser.add_argument(
    "--cvqnn-trunc-lambda",
    type=float,
    default=None,
    help=f"weight of the separate CVQNN block (W) truncation penalty added to "
    f"the CE loss (default CVQNN_TRUNC_LAMBDA={CVQNN_TRUNC_LAMBDA})",
)
parser.add_argument(
    "--gate-param-bound",
    type=str,
    default=None,
    help="soft-clip magnitude gate params (squeeze r, displacement re/im) to "
    "(-b, b) via b*tanh(x/b), preventing the degenerate over-leakage that NaNs "
    "heads at high num_heads. 'auto' = cutoff-aware photon budget "
    "asinh(sqrt(cutoff-1)) (recommended); or pass an explicit float; default off. "
    "Not checkpoint-compatible with unbounded.",
)
args = parser.parse_args()

if args.decoder_hidden_dim is not None and args.decoder_hidden_mult is not None:
    parser.error("--decoder-hidden-dim and --decoder-hidden-mult are mutually exclusive")
if args.train_fraction is not None and args.train_limit is not None:
    parser.error("--train-fraction and --train-limit are mutually exclusive")
if args.test_fraction is not None and args.test_limit is not None:
    parser.error("--test-fraction and --test-limit are mutually exclusive")
if args.train_fraction is not None and not (0.0 < args.train_fraction <= 1.0):
    parser.error("--train-fraction must be in (0, 1]")
if args.test_fraction is not None and not (0.0 < args.test_fraction <= 1.0):
    parser.error("--test-fraction must be in (0, 1]")

if args.epochs is not None:
    EPOCHS = args.epochs
if args.target_params is not None:
    TARGET_PARAMS = args.target_params
# The shared-CNN model auto-scales on num_heads by default (per-head linears are
# cheap, so cnn_channels_2 is a poor budget knob here). An explicit --scaling-knob
# still wins.
if args.model == "quantum_shared" and args.scaling_knob is None:
    SCALING_KNOB = "num_heads"
if args.scaling_knob is not None:
    SCALING_KNOB = args.scaling_knob
if args.num_layers is not None:
    NUM_LAYERS = args.num_layers
if args.trunc_lambda is not None:
    TRUNC_LAMBDA = args.trunc_lambda
if args.cvqnn_trunc_lambda is not None:
    CVQNN_TRUNC_LAMBDA = args.cvqnn_trunc_lambda
if args.seed is not None:
    SEED = args.seed
# Direct architecture overrides (manual mode). Applied before the gate-bound /
# observable resolution below so --cutoff-dim feeds both.
if args.num_modes is not None:
    NUM_MODES = args.num_modes
if args.cutoff_dim is not None:
    CUTOFF_DIM = args.cutoff_dim
if args.num_heads is not None:
    NUM_HEADS = args.num_heads
if args.cnn_channels_1 is not None:
    CNN_CHANNELS_1 = args.cnn_channels_1
if args.cnn_channels_2 is not None:
    CNN_CHANNELS_2 = args.cnn_channels_2
if args.cnn_kernel_size is not None:
    CNN_KERNEL_SIZE = args.cnn_kernel_size
if args.decoder_hidden_dim is not None:
    DECODER_HIDDEN_DIM = args.decoder_hidden_dim
if args.decoder_hidden_mult is not None:
    DECODER_HIDDEN_MULT = args.decoder_hidden_mult
if args.poly_degree is not None:
    POLY_DEGREE = args.poly_degree
if args.cnn_num_conv_layers is not None:
    CNN_NUM_CONV_LAYERS = args.cnn_num_conv_layers
if args.hypernet_num_linear_layers is not None:
    HYPERNET_NUM_LINEAR_LAYERS = args.hypernet_num_linear_layers
if args.decoder_num_layers is not None:
    DECODER_NUM_LAYERS = args.decoder_num_layers
if args.cvqnn_num_layers is not None:
    CVQNN_NUM_LAYERS = args.cvqnn_num_layers
if args.num_seq2seq_blocks is not None:
    NUM_SEQ2SEQ_BLOCKS = args.num_seq2seq_blocks
if args.pooling is not None:
    POOLING = args.pooling
if args.block_residual is not None:
    BLOCK_RESIDUAL = args.block_residual == "on"
if args.query_trunc_lambda is not None:
    QUERY_TRUNC_LAMBDA = args.query_trunc_lambda

# Resolve the gate-param bound: 'auto' → cutoff-aware photon budget, else a float,
# else None (off). Resolved to a concrete value here so config.json is reproducible.
if args.gate_param_bound is None:
    GATE_PARAM_BOUND = None
elif args.gate_param_bound == "auto":
    GATE_PARAM_BOUND = auto_gate_bound(CUTOFF_DIM)
    print(f"gate_param_bound: {GATE_PARAM_BOUND:.4f}  (auto for cutoff_dim={CUTOFF_DIM})\n")
else:
    GATE_PARAM_BOUND = float(args.gate_param_bound)

# Resolve the observable readout (sweep axis 2). A named preset gives a clean
# `observables_name` for run naming / history meta; ad-hoc JSON is labelled
# "custom" and requires an explicit --run-name since there is no short name.
if args.observables is not None and args.observables_json is not None:
    parser.error("--observables and --observables-json are mutually exclusive")
if args.observables_json is not None:
    if args.run_name is None:
        parser.error("--observables-json requires --run-name")
    readout_observables = [
        ObservableSpec(**spec) for spec in json.loads(args.observables_json)
    ]
    observables_name = "custom"
else:
    observables_name = args.observables or DEFAULT_OBSERVABLES
    readout_observables = resolve_observables(observables_name, CUTOFF_DIM)


# ---------------------------------------------------------------------------
# Config — the frozen best p12800 L2 architecture (num_heads=4, cnn_channels_2=8,
# num_layers=2, poly_degree=3, trunc norm λ=0.1, xpxsps → 13,530 params).
# target_params=-1 by default so it is built verbatim; --target-params re-enables
# auto-scaling on --scaling-knob (default num_heads). Observables / seed are swept.
# ---------------------------------------------------------------------------

data_cfg = DataConfig(
    dataset="fashionmnist",
    normalize=True,
    patch_size=7,
    batch_size=BATCH_SIZE,
    num_workers=0,
    data_root="data/",
)
quantum_cfg = QuantumConfig(
    num_modes=NUM_MODES,
    num_layers=NUM_LAYERS,
    cutoff_dim=CUTOFF_DIM,
    num_heads=NUM_HEADS,   # frozen-model value unless --num-heads / --target-params
    cnn_channels_1=CNN_CHANNELS_1,
    cnn_channels_2=CNN_CHANNELS_2,
    cnn_kernel_size=CNN_KERNEL_SIZE,
    decoder_hidden_dim=DECODER_HIDDEN_DIM,
    decoder_hidden_mult=DECODER_HIDDEN_MULT,
    poly_degree=POLY_DEGREE,
    cnn_num_conv_layers=CNN_NUM_CONV_LAYERS,
    hypernet_num_linear_layers=HYPERNET_NUM_LINEAR_LAYERS,
    decoder_num_layers=DECODER_NUM_LAYERS,
    cvqnn_num_layers=CVQNN_NUM_LAYERS,
    cvqnn_trunc_lambda=CVQNN_TRUNC_LAMBDA,
    num_seq2seq_blocks=NUM_SEQ2SEQ_BLOCKS,
    pooling=POOLING,
    block_residual=BLOCK_RESIDUAL,
    query_trunc_lambda=QUERY_TRUNC_LAMBDA,
    dtype="complex64",
    trunc_penalty="norm",
    trunc_lambda=TRUNC_LAMBDA,
    gate_param_bound=GATE_PARAM_BOUND,
    target_params=TARGET_PARAMS,
    scaling_knob=SCALING_KNOB,
    readout_observables=readout_observables,
)
config = ExperimentConfig(
    name="full_fashionmnist",
    model=args.model,
    data=data_cfg,
    quantum=quantum_cfg,
    training=TrainingConfig(lr=1e-3, epochs=EPOCHS, seed=SEED),
    use_wandb=args.wandb,
)


# ---------------------------------------------------------------------------
# Run directory: fresh datetime or recovered from --resume path
# ---------------------------------------------------------------------------

if args.resume:
    resume_path = Path(args.resume).resolve()
    if not resume_path.is_file():
        raise FileNotFoundError(f"--resume path does not exist: {resume_path}")
    # checkpoints/<name>.pt → parent is <run_dir>/checkpoints → grandparent is run_dir
    run_dir = resume_path.parent.parent
    print(f"Resuming into existing run directory: {run_dir}")
else:
    if args.run_name is not None:
        run_name = args.run_name
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_name = f"full_fashionmnist_{timestamp}"
    run_dir = Path(args.runs_root) / run_name
    print(f"Fresh run directory: {run_dir}")

ckpt_dir = run_dir / "checkpoints"
log_dir = run_dir / "logs"
preds_dir = run_dir / "predictions"
diag_dir = run_dir / "diagnostics"
for d in (run_dir, ckpt_dir, log_dir, preds_dir, diag_dir):
    d.mkdir(parents=True, exist_ok=True)


class _Tee:
    """Duplicates writes to multiple streams (used to tee stdout to a log file)."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, s):
        for st in self._streams:
            st.write(s)
        return len(s)

    def flush(self):
        for st in self._streams:
            st.flush()


# Tee stdout to logs/train.log (append on resume, fresh otherwise). tqdm writes to
# stderr by default, so progress bars do not pollute the log file.
_log_fh = open(log_dir / "train.log", "a" if args.resume else "w", buffering=1)
sys.stdout = _Tee(sys.__stdout__, _log_fh)


# ---------------------------------------------------------------------------
# Device + seed
# ---------------------------------------------------------------------------

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}\n")

torch.manual_seed(config.training.seed)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

train_ds_full = PatchedDataset(data_cfg, train=True)
test_ds_full = PatchedDataset(data_cfg, train=False)

if args.train_limit is not None:
    n = min(args.train_limit, len(train_ds_full))
    train_ds = Subset(train_ds_full, indices=list(range(n)))
    print(f"Train subset:  first {len(train_ds):,} samples (--train-limit)")
elif args.train_fraction is not None and args.train_fraction < 1.0:
    n = int(args.train_fraction * len(train_ds_full))
    g = torch.Generator().manual_seed(args.subset_seed)
    perm = torch.randperm(len(train_ds_full), generator=g)[:n].tolist()
    train_ds = Subset(train_ds_full, indices=perm)
    print(
        f"Train subset:  random {len(train_ds):,} / {len(train_ds_full):,} "
        f"samples (--train-fraction {args.train_fraction}, seed {args.subset_seed})"
    )
else:
    train_ds = train_ds_full
    print(f"Train set:     full {len(train_ds):,} samples")

if args.test_limit is not None:
    n = min(args.test_limit, len(test_ds_full))
    test_ds = Subset(test_ds_full, indices=list(range(n)))
    print(f"Test subset:   first {len(test_ds):,} samples (--test-limit)")
elif args.test_fraction is not None and args.test_fraction < 1.0:
    n = int(args.test_fraction * len(test_ds_full))
    g = torch.Generator().manual_seed(args.subset_seed)
    perm = torch.randperm(len(test_ds_full), generator=g)[:n].tolist()
    test_ds = Subset(test_ds_full, indices=perm)
    print(
        f"Test subset:   random {len(test_ds):,} / {len(test_ds_full):,} "
        f"samples (--test-fraction {args.test_fraction}, seed {args.subset_seed})"
    )
else:
    test_ds = test_ds_full
    print(f"Test set:      full {len(test_ds):,} samples")

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
train_eval_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)


save_test_images_once(
    test_loader,
    data_cfg.image_size,
    data_cfg.patch_size,
    preds_dir / "test_images.npz",
    progress="reassembling test images",
)

# Fixed diagnostic subset (sampled once at startup, stable across epochs and
# across resumes thanks to the fixed seed).
DIAG_SIZE = min(512, len(test_ds))
_diag_g = torch.Generator().manual_seed(args.subset_seed + 1)
diag_indices_in_test = torch.randperm(len(test_ds), generator=_diag_g)[
    :DIAG_SIZE
].tolist()
diag_ds = Subset(test_ds, indices=diag_indices_in_test)
diag_loader = DataLoader(diag_ds, batch_size=BATCH_SIZE, shuffle=False)

print(
    f"Train samples: {len(train_ds):,} | test samples: {len(test_ds):,} "
    f"| diagnostic subset: {len(diag_ds):,}"
)
print(
    f"Train batches/epoch: {len(train_loader):,} | test batches: {len(test_loader):,}\n"
)


def _absolute_indices(ds, full_len: int) -> np.ndarray:
    """Indices into the underlying full PatchedDataset. `Subset.indices` are
    already absolute. For an un-subset dataset we return arange(full_len)."""
    if isinstance(ds, Subset):
        return np.asarray(ds.indices, dtype=np.int64)
    return np.arange(full_len, dtype=np.int64)


def _save_subset_indices_once(out_path: Path) -> None:
    """Record train/test/diag subset indices (absolute, into the full
    PatchedDataset) so consumers like experiments/report_diagnostics.py's
    --full-inference path can reconstruct the exact same Subset.

    Idempotent: skip if the file already exists with matching sizes.
    """
    train_abs = _absolute_indices(train_ds, len(train_ds_full))
    test_abs = _absolute_indices(test_ds, len(test_ds_full))
    diag_abs = test_abs[diag_indices_in_test]
    if out_path.is_file():
        with np.load(out_path) as existing:
            if (
                existing["train_indices"].size == train_abs.size
                and existing["test_indices"].size == test_abs.size
                and existing["diag_indices"].size == diag_abs.size
            ):
                return
    np.savez_compressed(
        out_path,
        train_indices=train_abs,
        test_indices=test_abs,
        diag_indices=diag_abs,
    )


_save_subset_indices_once(run_dir / "subset_indices.npz")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

model = build_model(config).to(device)

# Capture parameter table to both stdout and a text file
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    print_parameter_table(model)
param_table_str = buf.getvalue()
print(param_table_str)
(run_dir / "parameter_table.txt").write_text(param_table_str)

n_params = model.get_num_parameters()
_target_str = f"{TARGET_PARAMS:,}" if TARGET_PARAMS > 0 else "off (fixed architecture)"
print(f"Trainable parameters: {n_params:,}  (target: {_target_str})\n")

# Shape check on one batch
_patches, _ = next(iter(train_loader))
_logits = model(_patches.to(device))
assert _logits.shape[-1] == 10, f"Expected logits dim 10, got {_logits.shape}"
print(f"Forward shape OK — logits: {tuple(_logits.shape)}\n")


# ---------------------------------------------------------------------------
# Optimizer + state
# ---------------------------------------------------------------------------

optimizer = torch.optim.Adam(model.parameters(), lr=config.training.lr)

history: dict = new_history(int(n_params), str(device))

start_epoch = 1
global_step = 0
if args.resume:
    ckpt = torch.load(args.resume, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    history = ckpt["history"]
    ensure_history_schema(history)
    start_epoch = ckpt["epoch"] + 1
    global_step = len(history["batch"]["step"])
    print(f"Resumed from {args.resume}")
    print(f"  Continuing from epoch {start_epoch} (global_step={global_step})\n")


# ---------------------------------------------------------------------------
# Save resolved config (after auto-scaling has chosen cnn_channels_2)
# ---------------------------------------------------------------------------

# Reflect the resolved quantum config (the auto-scaler may change any single
# scaling_knob — cnn_channels_2, num_heads, … — inside CVQuixer.__init__) so the
# saved JSON matches the model actually built and eval_cutoff_sweep.py /
# report_diagnostics.py rebuild the same architecture from config.json.
config_to_save = asdict(config)
if hasattr(model, "config"):
    config_to_save["quantum"] = asdict(model.config)

with open(run_dir / "config.json", "w") as f:
    json.dump(config_to_save, f, indent=2)

# Record sweep coordinates in history meta so report_sweep.py can group/plot
# runs without re-parsing config.json or logs. Idempotent across resumes.
history["meta"]["target_params"] = int(TARGET_PARAMS)
history["meta"]["scaling_knob"] = str(SCALING_KNOB)
history["meta"]["achieved_params"] = int(n_params)
history["meta"]["observables_name"] = observables_name
# model.config carries the *resolved* quantum config (post auto-scaling), so
# this reflects the achieved num_layers when it is the scaling_knob.
history["meta"]["num_layers"] = int(
    getattr(getattr(model, "config", None), "num_layers", config.quantum.num_layers)
)
history["meta"]["trunc_lambda"] = float(TRUNC_LAMBDA)
history["meta"]["cvqnn_trunc_lambda"] = float(CVQNN_TRUNC_LAMBDA)
history["meta"]["seed"] = int(SEED)
history["meta"]["model"] = str(args.model)
# Resolved architecture knobs (post auto-scaling), so report_sweep.py can table /
# plot manual-sweep axes without re-reading config.json. Read from model.config
# (the resolved config) with config.quantum as the fallback.
_resolved_q = getattr(model, "config", None) or config.quantum
for _field in (
    "num_modes", "num_heads", "cutoff_dim", "poly_degree",
    "cnn_channels_1", "cnn_channels_2", "cnn_kernel_size", "decoder_hidden_dim",
    "cnn_num_conv_layers", "hypernet_num_linear_layers", "decoder_num_layers",
    "cvqnn_num_layers", "num_seq2seq_blocks",
):
    history["meta"][_field] = int(getattr(_resolved_q, _field))
# Stacked-model coordinates (inert for the other models; ADR-0002).
history["meta"]["pooling"] = str(getattr(_resolved_q, "pooling", "mean"))
history["meta"]["block_residual"] = bool(
    getattr(_resolved_q, "block_residual", True)
)
history["meta"]["query_trunc_lambda"] = float(QUERY_TRUNC_LAMBDA)
# decoder_hidden_mult is a float|None (the input that produced the resolved
# decoder_hidden_dim above), kept separately from the int-cast loop.
_dhm = getattr(_resolved_q, "decoder_hidden_mult", None)
history["meta"]["decoder_hidden_mult"] = None if _dhm is None else float(_dhm)


# ---------------------------------------------------------------------------
# Weights & Biases (opt-in via --wandb; no-op otherwise)
# ---------------------------------------------------------------------------

wandb_tags = [
    f"params={TARGET_PARAMS}",
    f"obs={observables_name}",
    f"seed={SEED}",
    *(args.wandb_tags or []),
]
init_logging(
    config,
    group=args.wandb_group,
    tags=wandb_tags,
    name=run_dir.name,
    dir=run_dir,
)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------


def _grad_l2_norm(parameters) -> float:
    """Global L2 norm of gradients (diagnostic only, no clipping)."""
    norms = [p.grad.detach().norm(2) for p in parameters if p.grad is not None]
    if not norms:
        return 0.0
    return float(torch.norm(torch.stack(norms), 2).item())


def train_epoch(epoch: int) -> tuple[float, float, float]:
    """One epoch of training. Returns the mean per-sample truncation losses
    (per-patch, CVQNN-block W, query-unitary — the last is 0 for models
    without query unitaries) measured across training batches (cheap in-epoch
    summary, no extra pass).

    The epoch-level CE loss and accuracy are NOT computed here — they are
    produced after this function returns by running the post-epoch model on
    the full training set in eval mode, which avoids the running-average
    bias of mixing early-batch (under-trained) and late-batch predictions.

    Per-batch CE / trunc / cvqnn-trunc / total loss / batch acc / grad norm are
    still appended to history["batch"] so the per-batch diagnostic plots are
    unchanged.
    """
    global global_step
    model.train()
    total_trunc, total_cvqnn_trunc, total_query_trunc, total = 0.0, 0.0, 0.0, 0
    _logged_first_batch_mem = False
    for patches, labels in tqdm(
        train_loader, desc=f"Epoch {epoch:>3}/{EPOCHS}", leave=False, unit="batch"
    ):
        patches, labels = patches.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(patches, return_trunc_loss=True)
        logits, trunc_loss = out.logits, out.trunc_loss
        cvqnn_trunc_loss = out.cvqnn_trunc_loss
        # None for models without query unitaries (only the stacked model has
        # the stream); a zero stand-in keeps the logging path uniform.
        query_trunc_loss = (
            out.query_trunc_loss
            if out.query_trunc_loss is not None
            else torch.zeros((), device=logits.device)
        )
        ce_loss = F.cross_entropy(logits, labels)
        loss = (
            ce_loss
            + quantum_cfg.trunc_lambda * trunc_loss
            + quantum_cfg.cvqnn_trunc_lambda * cvqnn_trunc_loss
            + quantum_cfg.query_trunc_lambda * query_trunc_loss
        )
        loss.backward()

        grad_norm = _grad_l2_norm(model.parameters())
        optimizer.step()

        # Early OOM-headroom signal: the steady-state footprint (fwd+bwd+step)
        # is reached on the first batch. Print it once to stdout (the tee'd
        # train.log) so memory headroom is visible even if the epoch never
        # finishes. (peak measured since the per-epoch reset in the main loop.)
        if device.type == "cuda" and not _logged_first_batch_mem:
            _peak = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            print(f"  [epoch {epoch}] peak GPU mem after first batch: {_peak:,.0f} MB")
            _logged_first_batch_mem = True

        n = labels.size(0)
        preds = logits.argmax(dim=-1)
        batch_acc = (preds == labels).sum().item() / n

        total_trunc += trunc_loss.item() * n
        total_cvqnn_trunc += cvqnn_trunc_loss.item() * n
        total_query_trunc += query_trunc_loss.item() * n
        total += n

        global_step += 1
        history["batch"]["step"].append(global_step)
        history["batch"]["epoch"].append(epoch)
        history["batch"]["train_loss"].append(float(ce_loss.item()))
        history["batch"]["trunc_loss"].append(float(trunc_loss.item()))
        history["batch"]["cvqnn_trunc_loss"].append(float(cvqnn_trunc_loss.item()))
        history["batch"]["query_trunc_loss"].append(float(query_trunc_loss.item()))
        history["batch"]["total_loss"].append(float(loss.item()))
        history["batch"]["batch_acc"].append(float(batch_acc))
        history["batch"]["grad_norm"].append(float(grad_norm))

    return total_trunc / total, total_cvqnn_trunc / total, total_query_trunc / total


# Figure generation lives in `experiments/report_diagnostics.py` — run it
# post-hoc (or partway through a still-running job) against this run dir to
# regenerate the full figure set from `history.json` + the saved npz files.


def save_history() -> None:
    with open(run_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

print(
    f"{'Epoch':<7} {'Train loss':<12} {'Train acc':<11} "
    f"{'Test loss':<11} {'Test acc':<11} {'Trunc loss':<12} {'CVQNN trunc':<13} "
    f"{'Peak mem':<11} {'Time'}"
)
print("─" * 101)

# Per-epoch peak GPU memory is recorded under this key (CUDA only).
history["epoch"].setdefault("peak_mem_mb", [])

run_start = time.time()
best_test_acc = (
    max(history["epoch"]["test_acc"]) if history["epoch"]["test_acc"] else -1.0
)
best_epoch = (
    history["epoch"]["test_acc"].index(best_test_acc) + 1
    if history["epoch"]["test_acc"]
    else None
)

for epoch in range(start_epoch, EPOCHS + 1):
    t0 = time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    trunc_loss, cvqnn_trunc_loss, query_trunc_loss = train_epoch(epoch)
    train_eval = evaluate(model, train_eval_loader, device,
                          progress="train eval")
    test_eval = evaluate(model, test_loader, device,
                         progress="test eval")
    elapsed_train_eval = time.time() - t0

    # Peak GPU memory across this epoch (train + eval), recorded for headroom
    # tracking. None on non-CUDA devices.
    peak_mem_mb = (
        torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        if device.type == "cuda" else None
    )
    history["epoch"]["peak_mem_mb"].append(peak_mem_mb)

    train_loss, train_acc = train_eval["loss"], train_eval["acc"]
    test_loss, test_acc = test_eval["loss"], test_eval["acc"]

    history["epoch"]["train_loss"].append(train_loss)
    history["epoch"]["train_acc"].append(train_acc)
    history["epoch"]["test_loss"].append(test_loss)
    history["epoch"]["test_acc"].append(test_acc)
    history["epoch"]["trunc_loss"].append(trunc_loss)
    history["epoch"]["test_trunc_loss"].append(float(test_eval["trunc_loss"]))
    history["epoch"]["cvqnn_trunc_loss"].append(cvqnn_trunc_loss)
    history["epoch"]["test_cvqnn_trunc_loss"].append(float(test_eval["cvqnn_trunc_loss"]))
    history["epoch"]["query_trunc_loss"].append(query_trunc_loss)
    history["epoch"]["test_query_trunc_loss"].append(float(test_eval["query_trunc_loss"]))

    log_metrics(
        {
            "epoch": epoch,
            "epoch/train_loss": train_loss,
            "epoch/train_acc": train_acc,
            "epoch/test_loss": test_loss,
            "epoch/test_acc": test_acc,
            "epoch/trunc_loss": trunc_loss,
            "epoch/test_trunc_loss": float(test_eval["trunc_loss"]),
            "epoch/cvqnn_trunc_loss": cvqnn_trunc_loss,
            "epoch/test_cvqnn_trunc_loss": float(test_eval["cvqnn_trunc_loss"]),
            "epoch/query_trunc_loss": query_trunc_loss,
            "epoch/test_query_trunc_loss": float(test_eval["query_trunc_loss"]),
        },
        use_wandb=config.use_wandb,
    )

    np.savez_compressed(
        preds_dir / f"epoch_{epoch:04d}.npz",
        y_true=test_eval["y_true"],
        y_pred=test_eval["y_pred"],
        y_probs=test_eval["y_probs"],
        readouts=test_eval["readouts"],
    )
    np.savez_compressed(
        preds_dir / f"epoch_{epoch:04d}_train.npz",
        y_true=train_eval["y_true"],
        y_pred=train_eval["y_pred"],
        y_probs=train_eval["y_probs"],
        readouts=train_eval["readouts"],
    )

    # Coefficient snapshots: canonical models keep the historic flat keys;
    # the stacked model writes block-prefixed keys (block{b}_*/agg_*, ADR-0002).
    if args.model == "quantum_stacked":
        coeff_snaps = snapshot_stacked_coefficients(model)
    else:
        lcu_snap, poly_snap = snapshot_coefficients(model)
        cvqnn_snap = snapshot_cvqnn_params(model)   # None when cvqnn_num_layers == 0
        coeff_snaps = {"lcu_coeffs": lcu_snap, "poly_coeffs": poly_snap}
        if cvqnn_snap is not None:
            coeff_snaps["cvqnn_params"] = cvqnn_snap

    t_diag = time.time()
    try:
        stats_summary, mean_photon, diag_raw = quantum_diagnostics(
            model, diag_loader, device,
            progress="diagnostics",
        )
        diag_raw.update(coeff_snaps)
        np.savez_compressed(diag_dir / f"epoch_{epoch:04d}.npz", **diag_raw)
        diag_status = f"  (diag {time.time() - t_diag:.1f}s)"
    except Exception as e:
        # Don't let a diagnostic-pass failure kill a multi-hour training run.
        # Still persist lcu/poly snapshots so cross-epoch coefficient figures
        # have data for this epoch.
        import warnings

        warnings.warn(
            f"quantum_diagnostics failed at epoch {epoch}: {type(e).__name__}: {e}. "
            "Saving only lcu/poly snapshots for this epoch.",
            RuntimeWarning,
        )
        np.savez_compressed(diag_dir / f"epoch_{epoch:04d}.npz", **coeff_snaps)
        diag_status = "  (diag skipped)"

    elapsed = time.time() - t0
    mem_str = f"{peak_mem_mb:,.0f} MB" if peak_mem_mb is not None else "n/a"
    print(
        f"{epoch:<7} {train_loss:<12.4f} {train_acc:<11.3f} "
        f"{test_loss:<11.4f} {test_acc:<11.3f} {trunc_loss:<12.4f} "
        f"{cvqnn_trunc_loss:<13.6f} {mem_str:<11} {elapsed:.1f}s{diag_status}"
    )

    # Checkpoints
    ckpt_payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": history,
    }
    torch.save(ckpt_payload, ckpt_dir / "latest.pt")
    if epoch % CHECKPOINT_INTERVAL == 0:
        torch.save(ckpt_payload, ckpt_dir / f"epoch_{epoch:04d}.pt")
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        best_epoch = epoch
        torch.save(ckpt_payload, ckpt_dir / "best.pt")

    # Persist metadata every epoch so a SLURM kill is non-destructive.
    # (Figures are rendered separately by experiments/report_diagnostics.py.)
    history["meta"]["best_test_acc"] = float(best_test_acc)
    history["meta"]["best_epoch"] = int(best_epoch) if best_epoch is not None else None
    history["meta"]["total_runtime_sec"] = float(time.time() - run_start)
    save_history()

total_runtime = time.time() - run_start

# ---------------------------------------------------------------------------
# Final saves
# ---------------------------------------------------------------------------

history["meta"]["completed_at"] = datetime.now().isoformat(timespec="seconds")
history["meta"]["total_runtime_sec"] = float(total_runtime)
save_history()

torch.save(
    {"model_state_dict": model.state_dict(), "history": history},
    ckpt_dir / "final_model.pt",
)

log_metrics(
    {"best_test_acc": float(best_test_acc)},
    use_wandb=config.use_wandb,
)
finish_logging(config.use_wandb)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print(f"\nTotal runtime:    {total_runtime / 60:.1f} min")
print(f"Best test acc:    {best_test_acc:.4f} (epoch {best_epoch})")
print(f"Run directory:    {run_dir}/")
print(f"  config         → config.json")
print(f"  history        → history.json")
print(f"  predictions    → predictions/  (per-epoch npz + test_images.npz)")
print(f"  diagnostics    → diagnostics/  (per-epoch npz)")
print(
    f"  checkpoints    → checkpoints/  (latest.pt, best.pt, final_model.pt, epoch_NNNN.pt)"
)
print(
    f"\nFigures: run `uv run python experiments/report_diagnostics.py "
    f"--run-dir {run_dir}/` to populate {run_dir}/figures/"
)

#!/bin/bash
# -----------------------------------------------------------------------
# SLURM directives — A100 (40 GB), 12 h wall time.
#
# Prerequisite: the checkpoint must come from a recent
# experiments/full_experiment.py run that wrote subset_indices.npz to its
# run directory. The sweep mandatorily reuses subset_indices.npz's
# test_indices and diag_indices so per-cutoff numbers are apples-to-apples
# with the trained run; older runs without that file will fail at startup.
#
# Wall time is dominated by the per-cutoff test-set evaluation and scales
# linearly with the parent run's saved test-subset size (NOT the eval
# flags). For reference: sweep of [6,8,10,12] on a 10k-sample reused test
# set ≈ 4.5 h on A100-40 (D=12 dominates at ~2.5 h); on a 1k subset (e.g.
# parent run trained with --test-fraction 0.1) it's ~25-30 min. The 12 h
# wall covers extended cutoff lists like [6,8,10,12,14] and complex128
# reruns at full scale. The sweep also runs a fixed 512-sample quantum-
# diagnostic pass per cutoff (~tens of seconds each), independent of the
# test-set size.
#
# V100 fallback: change to `--gres=gpu:nv:1` if A100 queues are congested.
# Keep the 12 h time — V100 is slower.
#
# Usage:
#   sbatch scripts/run_eval_cutoff_sweep.sh <checkpoint.pt> [extra args]
#
# Examples:
#   # Default sweep [6,8,10,12], reusing the trained run's test subset
#   sbatch scripts/run_eval_cutoff_sweep.sh \
#       results/runs/full_fashionmnist_2026-05-15_.../checkpoints/final_model.pt
#
#   # Custom cutoff list
#   sbatch scripts/run_eval_cutoff_sweep.sh \
#       results/runs/<run>/checkpoints/final_model.pt \
#       --cutoffs 8 10
#
#   # Higher-precision sweep at extended cutoffs
#   sbatch scripts/run_eval_cutoff_sweep.sh \
#       results/runs/<run>/checkpoints/final_model.pt \
#       --cutoffs 6 8 10 12 14 --dtype complex128
#
# To shrink the eval pass below the parent run's saved subset, point at a
# checkpoint from a smaller-fraction parent run (preferred), or pass
# --test-fraction X / --test-limit N as an override — those override flags
# print a loud warning that the resulting numbers will NOT match the
# trained run.
#
# After the sweep completes, each cutoff has a self-contained sub-run dir
# at <run>/eval/cutoff_sweep_<ts>/D{NN}/ that report_diagnostics.py
# consumes directly to render the full thesis figure suite at that cutoff:
#   uv run python experiments/report_diagnostics.py --run-dir <...>/D{NN}
# -----------------------------------------------------------------------
#SBATCH --job-name=cv_quixer_eval
#SBATCH --output=slurm_logs/slurm-%x-%j.out
#SBATCH --error=slurm_logs/slurm-%x-%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:a100-40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail

CHECKPOINT="${1:?Usage: sbatch $0 <checkpoint.pt> [extra args]}"
shift   # drop $1; "$@" now holds any extra eval_cutoff_sweep.py flags

echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $SLURMD_NODENAME"
echo "Started:    $(date)"
echo "Checkpoint: $CHECKPOINT"
echo "Extra args: $*"

# -----------------------------------------------------------------------
# Navigate to project
# -----------------------------------------------------------------------
cd "$HOME/CV-Quixer"

# -----------------------------------------------------------------------
# uv — install if not in PATH
# -----------------------------------------------------------------------
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

# -----------------------------------------------------------------------
# Python environment — same dedicated CUDA venv as the training script.
# Do NOT rebuild the venv here; eval is a short-lived job and the venv
# from a recent training run is reusable. If it's missing, uv sync will
# create it.
# -----------------------------------------------------------------------
export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/cv-quixer-cuda"
echo "Syncing dependencies..."
uv sync

# -----------------------------------------------------------------------
# Sanity check — verify CUDA is visible to PyTorch
# -----------------------------------------------------------------------
uv run python - <<'EOF'
import torch
assert torch.cuda.is_available(), "CUDA not available — check GPU allocation"
print(f"Device:         {torch.cuda.get_device_name(0)}")
print(f"CUDA version:   {torch.version.cuda}")
print(f"PyTorch:        {torch.__version__}")
EOF

# -----------------------------------------------------------------------
# Run evaluation
# -----------------------------------------------------------------------
echo ""
echo "=== Import diagnostics ==="
PYTHONPATH="$HOME/CV-Quixer${PYTHONPATH:+:$PYTHONPATH}" \
    uv run python scripts/debug_imports.py

echo ""
echo "Starting cutoff-dim sweep evaluation..."
PYTHONPATH="$HOME/CV-Quixer${PYTHONPATH:+:$PYTHONPATH}" \
    uv run python experiments/eval_cutoff_sweep.py \
        --checkpoint "$CHECKPOINT" \
        "$@"

echo ""
echo "Finished: $(date)"

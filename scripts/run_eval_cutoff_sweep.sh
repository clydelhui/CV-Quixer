#!/bin/bash
# -----------------------------------------------------------------------
# SLURM directives — A100 (40 GB), 12 h wall time.
#
# Default sweep [6,8,10,12] on the full 10k test set is ~4.5 h on A100-40
# (dominated by D=12 at ~2.5 h). 12 h covers extended cutoff lists like
# [6,8,10,12,14] and complex128 reruns. With --test-fraction 0.5 or 0.25
# expect ~2.25 h / ~1.1 h respectively.
#
# V100 fallback: change to `--gres=gpu:nv:1` if A100 queues are congested.
# Keep the 12 h time — V100 is slower.
#
# Usage:
#   sbatch scripts/run_eval_cutoff_sweep.sh <checkpoint.pt> [extra args]
#
# Examples:
#   sbatch scripts/run_eval_cutoff_sweep.sh \
#       results/runs/full_fashionmnist_2026-05-15_.../checkpoints/final_model.pt
#
#   sbatch scripts/run_eval_cutoff_sweep.sh \
#       results/runs/<run>/checkpoints/final_model.pt \
#       --test-fraction 0.5 --cutoffs 8 10
#
#   sbatch scripts/run_eval_cutoff_sweep.sh \
#       results/runs/<run>/checkpoints/final_model.pt \
#       --test-fraction 0.25 --cutoffs 6 8 10 12 14 --dtype complex128
# -----------------------------------------------------------------------
#SBATCH --job-name=cv_quixer_eval
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
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

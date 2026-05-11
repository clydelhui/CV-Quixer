#!/bin/bash
# -----------------------------------------------------------------------
# SLURM directives
# -----------------------------------------------------------------------
#SBATCH --job-name=cv_quixer_mini
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:nv:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $SLURMD_NODENAME"
echo "Started:  $(date)"

# -----------------------------------------------------------------------
# Navigate to project
# -----------------------------------------------------------------------
cd "$HOME/CV-Quixer"

# -----------------------------------------------------------------------
# uv — install if not in PATH (installed per-user, persists in $HOME)
# -----------------------------------------------------------------------
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
# Add both possible install locations (newer uv uses .local/bin, older used .cargo/bin)
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

# -----------------------------------------------------------------------
# Python environment
# Dedicated CUDA venv — separate from local .venv (which has MPS torch).
# UV_PROJECT_ENVIRONMENT overrides the default .venv location.
# cu124 (CUDA 12.4 runtime) is forward-compatible with CUDA 12.5 and 12.9.
# All nv GPUs (V100 sm_70, Titan RTX sm_75, T4 sm_75) are supported.
# uv sync is fast on subsequent runs (only re-installs changed deps).
# -----------------------------------------------------------------------
export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/cv-quixer-cuda"

# Remove any existing venv so uv starts clean (previous run may have CUDA 13 torch).
rm -rf "$UV_PROJECT_ENVIRONMENT"

# cu124 as PRIMARY index — uv resolves torch from here first, avoiding PyPI's cu130 build.
# PyPI as extra index supplies all non-torch packages as normal.
# cu124 (CUDA 12.4 runtime) is compatible with all nv nodes (CUDA 12.5 and 12.9).
echo "Syncing dependencies (cu124 as primary index)..."
UV_INDEX_URL="https://download.pytorch.org/whl/cu124" \
UV_EXTRA_INDEX_URL="https://pypi.org/simple" \
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
# Run experiment
# -----------------------------------------------------------------------
echo ""
echo "Starting mini experiment..."
uv run python experiments/mini_experiment.py

echo ""
echo "Finished: $(date)"

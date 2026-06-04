#!/bin/bash
# -----------------------------------------------------------------------
# SLURM directives
# -----------------------------------------------------------------------
#SBATCH --job-name=cv_quixer_mini
#SBATCH --output=slurm_logs/slurm-%x-%j.out
#SBATCH --error=slurm_logs/slurm-%x-%j.err
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
# uv + per-arch CUDA venv (auto-installed/built; no manual pre-build).
# Pass REBUILD_VENV=1 (sbatch --export=ALL,REBUILD_VENV=1) for a clean rebuild.
# -----------------------------------------------------------------------
source scripts/setup_cuda_env.sh

# -----------------------------------------------------------------------
# Sanity check — verify CUDA is visible to PyTorch
# -----------------------------------------------------------------------
uv run --no-sync python - <<'EOF'
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
echo "=== Import diagnostics ==="
PYTHONPATH="$HOME/CV-Quixer${PYTHONPATH:+:$PYTHONPATH}" \
    uv run --no-sync python scripts/debug_imports.py

echo ""
echo "Starting mini experiment..."
PYTHONPATH="$HOME/CV-Quixer${PYTHONPATH:+:$PYTHONPATH}" \
    uv run --no-sync python experiments/mini_experiment.py

echo ""
echo "Finished: $(date)"

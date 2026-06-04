#!/bin/bash
# -----------------------------------------------------------------------
# SLURM directives — A100 (40 GB), 3 h wall time.
#
# By default this runs full_experiment.py on the FULL 60k/10k FashionMNIST
# split (the ~13.5k default model, 3 epochs, ~1 h on A100). Any args passed
# after the script name are forwarded verbatim to full_experiment.py, so a
# single custom grid point can be (re-)run, e.g.:
#   sbatch --gres=gpu:a100-80:1 scripts/run_full_experiment.sh \
#       --target-params 128000 --scaling-knob num_heads --observables xpxsps_pnr \
#       --run-name <name> --runs-root results/sweeps/<sweep>/ ...
# Heavy/custom configs (many heads, larger budgets) should override
# `--gres` (more GPU memory) and `--time` at submit, as above.
#
# V100 fallback: change to `--gres=gpu:nv:1` if A100 queues are
# congested (V100 will be slower; bump --time if needed).
# -----------------------------------------------------------------------
#SBATCH --job-name=cv_quixer_full
#SBATCH --output=slurm_logs/slurm-%x-%j.out
#SBATCH --error=slurm_logs/slurm-%x-%j.err
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:a100-40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

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
echo "Starting full FashionMNIST experiment..."

# GPU utilization/memory sampling (every 30s) → slurm_logs/, cleaned up on exit.
# Complements full_experiment.py's in-process peak-memory logging: this captures
# the memory-over-time curve (useful for spotting an OOM climb on a single run).
mkdir -p slurm_logs
GPU_LOG="slurm_logs/gpu_util-${SLURM_JOB_ID:-0}.csv"
nvidia-smi \
    --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw \
    --format=csv -l 30 > "$GPU_LOG" &
NVSMI_PID=$!
trap 'kill "$NVSMI_PID" 2>/dev/null || true' EXIT
echo "Sampling GPU utilization → $GPU_LOG"

# Args after the script name are forwarded to full_experiment.py (so a single
# custom/re-run grid point can be submitted). With no args, run the full 60k/10k
# split with default settings. To resume, pass
# `--resume results/runs/<run>/checkpoints/latest.pt` (re-pass any
# --train-fraction/--test-fraction/--subset-seed used originally).
if [ "$#" -gt 0 ]; then
    PYTHONPATH="$HOME/CV-Quixer${PYTHONPATH:+:$PYTHONPATH}" \
        uv run --no-sync python experiments/full_experiment.py "$@"
else
    PYTHONPATH="$HOME/CV-Quixer${PYTHONPATH:+:$PYTHONPATH}" \
        uv run --no-sync python experiments/full_experiment.py
fi

echo ""
echo "Finished: $(date)"

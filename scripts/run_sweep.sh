#!/bin/bash
# -----------------------------------------------------------------------
# SLURM job ARRAY for a CV-Quixer hyperparameter sweep.
#
# One array task = one grid point = one full_experiment.py run. The grid is
# described by a manifest written by experiments/sweep.py; this script reads
# the entry for $SLURM_ARRAY_TASK_ID and runs it.
#
# Submit (the array range must match the manifest's n_runs):
#     uv run python experiments/sweep.py --target-params ... --observables ... \
#         --epochs 3 --train-fraction 0.1 --test-fraction 0.1 --launch slurm
# (sweep.py prints/runs the exact `sbatch --array=0-<N-1> ...` command.)
#
# Or submit manually:
#     sbatch --array=0-5 scripts/run_sweep.sh \
#         results/sweeps/<sweep>_<ts>/sweep_manifest.json
#
# Per-task resources (A100, 6 h) — sized for full 60k/10k FashionMNIST at
# num_layers=2 (the 2-layer circuit roughly doubles per-patch cost vs L=1).
# Bump --time or change --gres if your epochs/depth/fractions need more.
# -----------------------------------------------------------------------
#SBATCH --job-name=cv_quixer_sweep
#SBATCH --output=slurm_logs/slurm-%x-%A_%a.out
#SBATCH --error=slurm_logs/slurm-%x-%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:a100-40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail

MANIFEST="${1:?usage: sbatch --array=0-<N-1> scripts/run_sweep.sh <sweep_manifest.json>}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

echo "Job ID:    ${SLURM_JOB_ID:-?}  (array task ${TASK_ID})"
echo "Node:      ${SLURMD_NODENAME:-?}"
echo "Manifest:  $MANIFEST"
echo "Started:   $(date)"

cd "$HOME/CV-Quixer"

# uv — install if not in PATH (persists in $HOME).
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

# Dedicated CUDA venv — REUSED across array tasks (do NOT rebuild per task, the
# tasks share this path and run concurrently). Build it once before submitting,
# e.g. by running scripts/run_full_experiment.sh once, or:
#     UV_PROJECT_ENVIRONMENT="$HOME/.venvs/cv-quixer-cuda" uv sync
export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/cv-quixer-cuda"
if [ ! -x "$UV_PROJECT_ENVIRONMENT/bin/python" ]; then
    echo "ERROR: CUDA venv not found at $UV_PROJECT_ENVIRONMENT."
    echo "Pre-build it once (avoids concurrent rebuilds across array tasks):"
    echo "    UV_PROJECT_ENVIRONMENT=\"$UV_PROJECT_ENVIRONMENT\" uv sync"
    exit 1
fi

# Sanity check — CUDA visible to PyTorch.
uv run python - <<'EOF'
import torch
assert torch.cuda.is_available(), "CUDA not available — check GPU allocation"
print(f"Device:       {torch.cuda.get_device_name(0)}")
print(f"PyTorch:      {torch.__version__}  (CUDA {torch.version.cuda})")
EOF

# Pull this task's full_experiment.py arguments out of the manifest (one per
# line → bash array). The grid values contain no spaces, but readarray handles
# them robustly regardless.
readarray -t RUN_ARGS < <(uv run python - "$MANIFEST" "$TASK_ID" <<'EOF'
import json, sys
manifest_path, task_id = sys.argv[1], int(sys.argv[2])
manifest = json.load(open(manifest_path))
run = next(r for r in manifest["runs"] if r["index"] == task_id)
print(run["run_name"], file=sys.stderr)
print("\n".join(run["args"]))
EOF
)

# GPU utilization sampling for future resource planning. Polls every 30s into
# a CSV next to the slurm logs; cleaned up on exit (success or failure). wandb
# also auto-logs system metrics, but this gives a self-contained per-task record.
mkdir -p slurm_logs
GPU_LOG="slurm_logs/gpu_util-${SLURM_JOB_ID:-0}_${TASK_ID}.csv"
nvidia-smi \
    --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw \
    --format=csv -l 30 > "$GPU_LOG" &
NVSMI_PID=$!
trap 'kill "$NVSMI_PID" 2>/dev/null || true' EXIT
echo "Sampling GPU utilization → $GPU_LOG"

echo ""
echo "Running full_experiment.py with: ${RUN_ARGS[*]}"
echo ""

PYTHONPATH="$HOME/CV-Quixer${PYTHONPATH:+:$PYTHONPATH}" \
    uv run python experiments/full_experiment.py "${RUN_ARGS[@]}"

echo ""
echo "Finished task ${TASK_ID}: $(date)"

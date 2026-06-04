#!/bin/bash
# -----------------------------------------------------------------------
# SLURM job ARRAY for a CV-Quixer cutoff-dim sweep across a whole sweep dir.
#
# One array task = one sweep run = one eval_cutoff_sweep.py invocation. The
# per-run invocations are described by a manifest written by
# experiments/eval_cutoff_sweep_all.py; this script reads the entry for
# $SLURM_ARRAY_TASK_ID and runs it.
#
# Submit (the array range must match the manifest's n_runs):
#     uv run python experiments/eval_cutoff_sweep_all.py \
#         --sweep-dir results/sweeps/<sweep>_<ts>/ --launch slurm
# (eval_cutoff_sweep_all.py prints/runs the exact `sbatch --array=0-<N-1> ...`.)
#
# Or submit manually:
#     sbatch --array=0-5 scripts/run_eval_cutoff_sweep_array.sh \
#         results/sweeps/<sweep>_<ts>/cutoff_sweep_manifest.json
#
# Per-task resources mirror scripts/run_eval_cutoff_sweep.sh (A100-40, 12 h):
# the per-cutoff eval is dominated by the test-set pass and scales ~O(D^4) and
# linearly with num_heads. Default cutoffs are [6,8,10]; high-head-count runs
# (the num_heads budgets) at D=12 / full 10k test can exceed the 12 h wall —
# pass --test-fraction to eval_cutoff_sweep_all.py for those.
#
# V100 fallback: change to `--gres=gpu:nv:1` if A100 queues are congested.
# -----------------------------------------------------------------------
#SBATCH --job-name=cv_quixer_eval_all
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:a100-40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail

MANIFEST="${1:?usage: sbatch --array=0-<N-1> scripts/run_eval_cutoff_sweep_array.sh <cutoff_sweep_manifest.json>}"
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
# tasks share this path and run concurrently). Build it once before submitting:
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

# Pull this task's eval_cutoff_sweep.py arguments out of the manifest (one per
# line → bash array).
readarray -t EVAL_ARGS < <(uv run python - "$MANIFEST" "$TASK_ID" <<'EOF'
import json, sys
manifest_path, task_id = sys.argv[1], int(sys.argv[2])
manifest = json.load(open(manifest_path))
run = next(r for r in manifest["runs"] if r["index"] == task_id)
print(run["run_name"], file=sys.stderr)
print("\n".join(run["args"]))
EOF
)

# GPU utilization sampling for resource planning. Polls every 30s into a CSV
# next to the slurm logs; cleaned up on exit (success or failure).
GPU_LOG="gpu_util-${SLURM_JOB_ID:-0}_${TASK_ID}.csv"
nvidia-smi \
    --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw \
    --format=csv -l 30 > "$GPU_LOG" &
NVSMI_PID=$!
trap 'kill "$NVSMI_PID" 2>/dev/null || true' EXIT
echo "Sampling GPU utilization → $GPU_LOG"

echo ""
echo "Running eval_cutoff_sweep.py with: ${EVAL_ARGS[*]}"
echo ""

PYTHONPATH="$HOME/CV-Quixer${PYTHONPATH:+:$PYTHONPATH}" \
    uv run python experiments/eval_cutoff_sweep.py "${EVAL_ARGS[@]}"

echo ""
echo "Finished task ${TASK_ID}: $(date)"

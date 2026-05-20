#!/bin/bash
# -----------------------------------------------------------------------
# SLURM directives — A100 (40 GB), 2 h wall time.
#
# Replays the post-epoch evaluation + quantum-diagnostic passes the current
# training loop performs for every epoch_NNNN.pt in <run-dir>/checkpoints/,
# writing the artefacts that experiments/report_diagnostics.py needs:
#   predictions/test_images.npz, predictions/epoch_NNNN.npz,
#   diagnostics/epoch_NNNN.npz, subset_indices.npz, and an in-place patch
#   to history.json.
#
# Workload per epoch ≈ one forward pass over the test set + one forward
# pass over the 512-sample diagnostic subset. Pure inference, no gradients,
# so 2 h is generous even for a 3-epoch full FashionMNIST run on V100.
#
# V100 fallback: change to `--gres=gpu:nv:1` if A100 queues are congested
# (no time bump needed — the workload is small).
#
# The script writes into the existing run directory; no new run dir is
# created. The dedicated CUDA venv is shared with run_full_experiment.sh
# and run_eval_cutoff_sweep.sh; reused rather than rebuilt.
#
# Usage:
#   sbatch scripts/run_backfill_artefacts.sh <run-dir> [extra args]
#
# Examples:
#   # Backfill the target older run
#   sbatch scripts/run_backfill_artefacts.sh \
#       results/runs/full_fashionmnist_2026-05-15_01-55-34/
#
#   # Backfill just one epoch (pass-through arg)
#   sbatch scripts/run_backfill_artefacts.sh \
#       results/runs/<run>/ --epochs 3
#
#   # Force re-evaluation even if predictions/diagnostics exist
#   sbatch scripts/run_backfill_artefacts.sh \
#       results/runs/<run>/ --overwrite
# -----------------------------------------------------------------------
#SBATCH --job-name=cv_quixer_backfill
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:a100-40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail

RUN_DIR="${1:?Usage: sbatch $0 <run-dir> [extra args]}"
shift   # drop $1; "$@" now holds any extra backfill_artefacts.py flags

echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $SLURMD_NODENAME"
echo "Started:  $(date)"
echo "Run dir:  $RUN_DIR"
echo "Extra:    $*"

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
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

# -----------------------------------------------------------------------
# Python environment — reuse the dedicated CUDA venv (do NOT rebuild;
# this is a short-lived job and the venv from a recent training run is
# reusable).
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
# Run backfill
# -----------------------------------------------------------------------
echo ""
echo "=== Import diagnostics ==="
PYTHONPATH="$HOME/CV-Quixer${PYTHONPATH:+:$PYTHONPATH}" \
    uv run python scripts/debug_imports.py

echo ""
echo "Starting artefact backfill..."
PYTHONPATH="$HOME/CV-Quixer${PYTHONPATH:+:$PYTHONPATH}" \
    uv run python experiments/backfill_artefacts.py \
        --run-dir "$RUN_DIR" \
        "$@"

echo ""
echo "Finished: $(date)"

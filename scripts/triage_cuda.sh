#!/bin/bash
#SBATCH --job-name=cuda_triage
#SBATCH --output=slurm_logs/slurm-%x-%j.out
#SBATCH --error=slurm_logs/slurm-%x-%j.err
#SBATCH --time=00:15:00
#SBATCH --gres=gpu:nv:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G

set -euo pipefail

echo "=== Node info ==="
echo "Node: $SLURMD_NODENAME"
echo "Date: $(date)"

echo ""
echo "=== nvidia-smi ==="
nvidia-smi

echo ""
echo "=== CUDA toolkit on PATH ==="
nvcc --version 2>/dev/null || echo "nvcc not on PATH"

# Ensure uv is available
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

echo ""
echo "=== Currently installed torch (in CUDA venv) ==="
export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/cv-quixer-cuda"
uv run python - <<'EOF'
import torch
print(f"torch version:      {torch.__version__}")
print(f"torch.version.cuda: {torch.version.cuda}")
print(f"CUDA available:     {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device:             {torch.cuda.get_device_name(0)}")
EOF

echo ""
echo "=== Latest torch available: cu126 index ==="
uv pip index versions torch \
    --index-url https://download.pytorch.org/whl/cu126 \
    2>/dev/null | head -3 || echo "Failed to query cu126 index"

echo ""
echo "=== Latest torch available: cu128 index ==="
uv pip index versions torch \
    --index-url https://download.pytorch.org/whl/cu128 \
    2>/dev/null | head -3 || echo "Failed to query cu128 index"

echo ""
echo "Triage complete."

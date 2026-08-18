#!/bin/bash
# -----------------------------------------------------------------------
# Per-block success probabilities + truncation streams for a stacked run
# (experiments/eval_block_stages.py).
#
# Epochs are independent, so this is a job ARRAY: each task takes a
# round-robin stripe of the run's epochs via --num-shards/--shard. Submit
# with as many tasks as you want GPUs, e.g. 5 tasks over 25 epochs:
#
#     sbatch --array=0-4 scripts/run_block_stages.sh \
#         --run-dir results/sweeps/<sweep>/<run>/ --blocks all --epochs all
#
# Everything after the script name is forwarded to eval_block_stages.py, so
# --batch-size / --overwrite / an explicit --epochs list all work unchanged.
# Submitted without --array it runs as a single task doing every epoch.
#
# Concurrent tasks are safe: sidecar filenames are per (epoch, block) and each
# shard writes its own provenance file.
# -----------------------------------------------------------------------
#SBATCH --job-name=cv_quixer_blocks
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:a100-40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=slurm_logs/slurm-%x-%A_%a.out
#SBATCH --error=slurm_logs/slurm-%x-%A_%a.err

set -euo pipefail

cd "$HOME/CV-Quixer"
# shellcheck source=/dev/null
source scripts/setup_cuda_env.sh

# Absent when submitted without --array: one shard covering every epoch.
SHARD="${SLURM_ARRAY_TASK_ID:-0}"
NUM_SHARDS="${SLURM_ARRAY_TASK_COUNT:-1}"
# SLURM indexes tasks from the array spec, which need not start at 0.
TASK_MIN="${SLURM_ARRAY_TASK_MIN:-0}"
SHARD=$((SHARD - TASK_MIN))

echo "[block-stages] shard $((SHARD + 1))/$NUM_SHARDS on $(hostname)"

uv run --no-sync python experiments/eval_block_stages.py \
    --num-shards "$NUM_SHARDS" --shard "$SHARD" "$@"

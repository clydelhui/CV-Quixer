#!/bin/bash
# -----------------------------------------------------------------------
# SLURM job for experiments/thesis_run_figures.py — the thesis-ready per-run
# figure suite.
#
# Runs on the cluster because that is where the raw artefacts live: most of the
# suite reads predictions/*.npz and diagnostics/*.npz, which a `light` pull
# (scripts/pull_results.sh, the default tier) deliberately leaves behind. Render
# here, then bring back only the finished figures:
#
#     bash scripts/pull_results.sh <sweep_dir>... --tier figures
#
# CPU-ONLY (no --gres): no model rebuild, so it counts against MaxSubmitJobs
# (32) and never against the 8-GPU limit.
#
# A JOB ARRAY. The work is I/O-bound, not compute-bound: deriving the
# accuracy/loss curves streams every epoch's train-side predictions npz, and
# those are the ~94 MB/epoch files — roughly 2.4 GB per 25-epoch run off network
# storage. Runs are split across the array by a round-robin stripe (the script's
# own --shard/--num-shards, so the run filters stay in one place), so wall time
# falls almost linearly with the array width.
#
# Submit — one or more sweep dirs, extra args after a literal `--`:
#     sbatch --array=0-9 scripts/run_thesis_figures.sh \
#         results/sweeps/high_epoch_quantum_<ts>/ \
#         results/sweeps/high_epoch_shared_<ts>/ \
#         results/sweeps/high_epoch_stacked_<ts>/ -- --min-epochs 25
#
# Without --array it runs as a single task (shard 0 of 1) and does everything.
# Tasks beyond the run count simply find an empty stripe and exit cleanly, so
# over-sizing the array is harmless.
#
# Everything after `--` goes to thesis_run_figures.py (e.g. --min-epochs,
# --epoch final, --only <figure>, --runs <pattern>).
# -----------------------------------------------------------------------
#SBATCH --job-name=cv_quixer_thesis_figs
#SBATCH --output=slurm_logs/slurm-%x-%A_%a.out
#SBATCH --error=slurm_logs/slurm-%x-%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

if [ $# -eq 0 ]; then
    echo "usage: sbatch scripts/run_thesis_figures.sh <sweep_dir>... [-- thesis_run_figures args...]" >&2
    exit 2
fi

# Split argv at the first `--`: sweep dirs before, script flags after.
SWEEP_DIRS=()
EXTRA_ARGS=()
while [ $# -gt 0 ]; do
    if [ "$1" = "--" ]; then shift; EXTRA_ARGS=("$@"); break; fi
    SWEEP_DIRS+=("$1"); shift
done

if [ "${#SWEEP_DIRS[@]}" -eq 0 ]; then
    echo "error: no sweep dir given before --" >&2
    exit 2
fi

# Array coordinates -> shard index. TASK_MIN makes this correct for an array
# that does not start at 0 (e.g. --array=5-9), matching run_report_array.sh.
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
TASK_MIN="${SLURM_ARRAY_TASK_MIN:-0}"
TASK_MAX="${SLURM_ARRAY_TASK_MAX:-0}"
NUM_SHARDS="${SLURM_ARRAY_TASK_COUNT:-1}"
SHARD=$(( TASK_ID - TASK_MIN ))

# The shard index is a position within the array (TASK_ID - TASK_MIN), so the
# array must be contiguous for the indices to cover 0..NUM_SHARDS-1 exactly. A
# gappy spec (--array=0,3,7) would leave some runs assigned to a shard no task
# ever claims — silently dropping them. Refuse instead. A throttle
# (--array=0-9%3) is contiguous and fine.
if [ "$NUM_SHARDS" -ne $(( TASK_MAX - TASK_MIN + 1 )) ]; then
    echo "error: non-contiguous --array (min=$TASK_MIN max=$TASK_MAX count=$NUM_SHARDS)." >&2
    echo "       Use a contiguous range such as --array=0-$(( NUM_SHARDS - 1 ))" >&2
    echo "       (a throttle like --array=0-9%3 is contiguous and supported)." >&2
    exit 2
fi

echo "Job ID:     ${SLURM_JOB_ID:-?}  (array task ${TASK_ID}, shard ${SHARD}/${NUM_SHARDS})"
echo "Node:       ${SLURMD_NODENAME:-?}"
echo "Sweep dirs: ${SWEEP_DIRS[*]}"
echo "Extra args: ${EXTRA_ARGS[*]-<none>}"
echo "Started:    $(date)"

cd "$HOME/CV-Quixer"

# uv + per-arch venv (already built by the training jobs; reused, no GPU needed).
source scripts/setup_cuda_env.sh

# One --sweep-dir flag per positional dir.
SWEEP_FLAGS=()
for d in "${SWEEP_DIRS[@]}"; do SWEEP_FLAGS+=(--sweep-dir "$d"); done

echo ""
PYTHONPATH="$HOME/CV-Quixer${PYTHONPATH:+:$PYTHONPATH}" \
    uv run --no-sync python -u experiments/thesis_run_figures.py \
        "${SWEEP_FLAGS[@]}" \
        --shard "$SHARD" --num-shards "$NUM_SHARDS" \
        ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}

echo ""
echo "Finished thesis figures: $(date)"

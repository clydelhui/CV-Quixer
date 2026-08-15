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
# CPU-ONLY (no --gres): JSON + npz work with no model rebuild, so it counts
# against MaxSubmitJobs (32) and never against the 8-GPU limit.
#
# Submit — one or more sweep dirs, extra args forwarded verbatim:
#     sbatch scripts/run_thesis_figures.sh results/sweeps/<sweep>_<ts>/
#     sbatch scripts/run_thesis_figures.sh \
#         results/sweeps/high_epoch_quantum_<ts>/ \
#         results/sweeps/high_epoch_shared_<ts>/ \
#         results/sweeps/high_epoch_stacked_<ts>/ -- --min-epochs 25
#
# Everything after a literal `--` goes to thesis_run_figures.py (e.g.
# --min-epochs, --epoch final, --only <figure>, --runs <pattern>).
# -----------------------------------------------------------------------
#SBATCH --job-name=cv_quixer_thesis_figs
#SBATCH --output=slurm_logs/slurm-%x-%j.out
#SBATCH --error=slurm_logs/slurm-%x-%j.err
#SBATCH --time=01:00:00
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

echo "Job ID:     ${SLURM_JOB_ID:-?}"
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
        "${SWEEP_FLAGS[@]}" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}

echo ""
echo "Finished thesis figures: $(date)"

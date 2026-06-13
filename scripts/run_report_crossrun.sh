#!/bin/bash
# -----------------------------------------------------------------------
# SLURM job for the fast, cross-run half of report_sweep.py.
#
# Runs report_sweep.py --skip-per-run-figures: the JSON-only aggregation that
# writes summary.csv / summary.md + the cross-run figures/ (acc_vs_params,
# acc_by_observable, acc_vs_<field>, ...) into the sweep dir. No torch, no GPU,
# no per-run report_diagnostics — those are fanned out separately by
# scripts/run_report_array.sh. This pass does NOT depend on the per-run figures
# existing, so it can be submitted standalone at any time for the headline
# table, or chained after the array by scripts/submit_report.sh.
#
# CPU-ONLY (no --gres): counts against MaxSubmitJobs (32), never the GPU limit.
#
# Submit:
#     sbatch scripts/run_report_crossrun.sh results/sweeps/<sweep>_<ts>/
#
# Extra args after the sweep dir are forwarded verbatim to report_sweep.py
# (e.g. --max-epoch for an epoch-fair comparison, or --series-by):
#     sbatch scripts/run_report_crossrun.sh results/sweeps/<sweep>_<ts>/ \
#         --max-epoch 4 --series-by model observables scaling_knob
# -----------------------------------------------------------------------
#SBATCH --job-name=cv_quixer_report_crossrun
#SBATCH --output=slurm_logs/slurm-%x-%j.out
#SBATCH --error=slurm_logs/slurm-%x-%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

set -euo pipefail

SWEEP_DIR="${1:?usage: sbatch scripts/run_report_crossrun.sh <sweep_dir> [report_sweep args...]}"
shift
EXTRA_ARGS=("$@")   # forwarded verbatim to report_sweep.py

echo "Job ID:    ${SLURM_JOB_ID:-?}"
echo "Node:      ${SLURMD_NODENAME:-?}"
echo "Sweep dir: $SWEEP_DIR"
echo "Started:   $(date)"

cd "$HOME/CV-Quixer"

# uv + per-arch venv (already built by the sweep jobs; reused, no GPU needed).
source scripts/setup_cuda_env.sh

echo ""
echo "Running report_sweep.py --skip-per-run-figures ${EXTRA_ARGS[*]}"
echo ""

PYTHONPATH="$HOME/CV-Quixer${PYTHONPATH:+:$PYTHONPATH}" \
    uv run --no-sync python -u experiments/report_sweep.py \
        --sweep-dir "$SWEEP_DIR" --skip-per-run-figures ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}

echo ""
echo "Finished cross-run aggregation: $(date)"

#!/bin/bash
# -----------------------------------------------------------------------
# One-command launcher for cluster-side sweep reporting.
#
# Submits, for ONE sweep:
#   1. run_report_array.sh  — the striped per-run report_diagnostics figures
#                             (CPU array; the slow, parallel part);
#   2. run_report_crossrun.sh — the cross-run summary table + figures,
#                             chained with --dependency=afterany on the array
#                             so it runs once the per-run figures are done.
# Both are CPU-only (no --gres) so they cost nothing against the 8 GPU slots —
# only against MaxSubmitJobs (32). This script runs ON THE LOGIN NODE and
# returns in seconds (it only issues two sbatch calls — within the
# no-long-process policy); it is NOT itself sbatch'd.
#
# Usage:
#     bash scripts/submit_report.sh <sweep_dir> [array_spec] [-- diag args...]
#
#   <sweep_dir>   results/sweeps/<sweep>_<ts>/
#   [array_spec]  sbatch --array spec = your parallelism dial (default 0-23).
#   [-- ...]      anything after a literal `--` is forwarded to
#                 report_diagnostics.py (e.g. --epoch final).
#
# Queue math (every pending+running array task counts toward MaxSubmitJobs=32):
#     one sweep at a time : default 0-23  -> 24 tasks + 1 pending crossrun = 25
#     all three at once   : pass 0-7 to each -> 3 x (8 + 1) = 27
# No %N throttle is needed (CPU tasks are short).
#
# Examples:
#     # one sweep, full parallelism:
#     bash scripts/submit_report.sh results/sweeps/grid_quantum_<ts>/
#
#     # all three sweeps concurrently (run once per sweep dir):
#     bash scripts/submit_report.sh results/sweeps/grid_quantum_<ts>/ 0-7
#     bash scripts/submit_report.sh results/sweeps/grid_shared_<ts>/  0-7
#     bash scripts/submit_report.sh results/sweeps/grid_stacked_<ts>/ 0-7
#
#     # forward a non-default epoch to the per-run figures:
#     bash scripts/submit_report.sh results/sweeps/grid_quantum_<ts>/ 0-23 -- --epoch final
#
# Stop everything: scancel --name=cv_quixer_report --name=cv_quixer_report_crossrun
# -----------------------------------------------------------------------
set -euo pipefail

SWEEP_DIR="${1:?usage: bash scripts/submit_report.sh <sweep_dir> [array_spec] [-- diag args...]}"
shift

ARRAY_SPEC="0-23"
if [ "$#" -gt 0 ] && [ "$1" != "--" ]; then
    ARRAY_SPEC="$1"
    shift
fi
[ "${1:-}" = "--" ] && shift   # drop the separator; the rest go to the array script
DIAG_ARGS=("$@")               # forwarded verbatim to report_diagnostics.py

cd "$HOME/CV-Quixer"

if [ ! -d "$SWEEP_DIR" ]; then
    echo "error: sweep dir does not exist: $SWEEP_DIR" >&2
    exit 1
fi

ARR_JOB=$(sbatch --parsable --array="$ARRAY_SPEC" \
    scripts/run_report_array.sh "$SWEEP_DIR" ${DIAG_ARGS[@]+"${DIAG_ARGS[@]}"})
echo "submitted per-run figure array (--array=${ARRAY_SPEC}) as job ${ARR_JOB}"

CROSS_JOB=$(sbatch --parsable --dependency="afterany:${ARR_JOB}" \
    scripts/run_report_crossrun.sh "$SWEEP_DIR")
echo "submitted cross-run aggregation as job ${CROSS_JOB}, fires after ${ARR_JOB} drains"

echo ""
echo "sweep dir: $SWEEP_DIR"
echo "watch:     squeue --me   |   tail -f slurm_logs/slurm-cv_quixer_report-${ARR_JOB}_*.out"
echo "results:   per-run figures in each run's figures/; cross-run table + figures in the sweep dir"

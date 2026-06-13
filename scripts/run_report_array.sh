#!/bin/bash
# -----------------------------------------------------------------------
# SLURM job ARRAY for per-run report_diagnostics figure rendering.
#
# The slow half of report_sweep.py is render_per_run_figures(): one
# `report_diagnostics.py --run-dir <run>` subprocess per run, run SEQUENTIALLY
# (162 / 162 / 64 runs for the grid_quantum / grid_shared / grid_stacked
# sweeps — ~388 invocations total). report_diagnostics' default path is
# npz/JSON + matplotlib with torch imports deferred, so each run is a short,
# independent, CPU-ONLY job that writes only into that run's own figures/ dir.
# That makes it embarrassingly parallel — this script fans it across an array.
#
# CPU-ONLY: NO --gres. These tasks therefore do NOT count against the GPU GRES
# limit (8), only against MaxSubmitJobs (32). Run them AFTER the GPU sweep
# chains have drained and the full queue budget is free.
#
# STRIPING: each task processes a round-robin stripe of the sweep's runs
# (run i handled by task `i % SLURM_ARRAY_TASK_COUNT`), so the number of array
# tasks is YOUR parallelism dial, set purely via --array. Round-robin (not
# contiguous) balances the load even though runs are name-sorted and the slow
# num_modes=4 corners would otherwise clump together. There is no manifest and
# no chunking machinery: pick an array size <= MaxSubmitJobs and submit once.
#
# Queue math (every pending+running array task counts toward MaxSubmitJobs=32):
#     one sweep at a time : --array=0-23   (24 stripes, 8 spare)
#     all three at once   : --array=0-7    each (3 x 8 = 24 stripes)
# %N throttling is NOT needed (CPU tasks are short and cheap); a plain
# --array=0-<N-1> is fine as long as N <= 32.
#
# Submit (usually via scripts/submit_report.sh, which also chains the
# cross-run aggregation; or directly):
#     sbatch --array=0-23 scripts/run_report_array.sh \
#         results/sweeps/<sweep>_<ts>/
#
# Extra args after the sweep dir are forwarded verbatim to report_diagnostics
# (e.g. a non-default --epoch):
#     sbatch --array=0-15 scripts/run_report_array.sh \
#         results/sweeps/<sweep>_<ts>/ --epoch final
#
# A task that hits a broken/incomplete run logs it and continues — one bad run
# never aborts the stripe. The per-task exit code is non-zero iff that stripe
# had >=1 failure, so `sacct` / the .err files surface casualties for triage.
# -----------------------------------------------------------------------
#SBATCH --job-name=cv_quixer_report
#SBATCH --output=slurm_logs/slurm-%x-%A_%a.out
#SBATCH --error=slurm_logs/slurm-%x-%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

SWEEP_DIR="${1:?usage: sbatch --array=0-<N-1> scripts/run_report_array.sh <sweep_dir> [report_diagnostics args...]}"
shift
EXTRA_ARGS=("$@")   # forwarded verbatim to report_diagnostics.py

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
TASK_MIN="${SLURM_ARRAY_TASK_MIN:-0}"
TASK_COUNT="${SLURM_ARRAY_TASK_COUNT:-1}"
LOCAL_ID=$(( TASK_ID - TASK_MIN ))   # 0-based stripe index

echo "Job ID:    ${SLURM_JOB_ID:-?}  (array task ${TASK_ID}, stripe ${LOCAL_ID}/${TASK_COUNT})"
echo "Node:      ${SLURMD_NODENAME:-?}"
echo "Sweep dir: $SWEEP_DIR"
echo "Started:   $(date)"

cd "$HOME/CV-Quixer"

# uv + per-arch venv. The sweep jobs already built it, so this just sets
# UV_PROJECT_ENVIRONMENT and reuses it (no GPU is touched or required here).
source scripts/setup_cuda_env.sh

# This stripe's run dirs: subdirs of the sweep with a history.json (matches what
# report_sweep.py's render_per_run_figures iterates over, and naturally excludes
# the sweep-level figures/ dir), round-robin'd by index.
readarray -t RUN_DIRS < <(uv run --no-sync python - "$SWEEP_DIR" "$LOCAL_ID" "$TASK_COUNT" <<'EOF'
import sys
from pathlib import Path

sweep_dir, local_id, task_count = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
runs = sorted(
    p for p in Path(sweep_dir).iterdir()
    if p.is_dir() and (p / "history.json").is_file()
)
mine = [p for i, p in enumerate(runs) if i % task_count == local_id]
print(
    f"{len(runs)} run(s) with history.json under {sweep_dir}; "
    f"this stripe ({local_id}/{task_count}) owns {len(mine)}",
    file=sys.stderr,
)
for p in mine:
    print(p)
EOF
)

if [ "${#RUN_DIRS[@]}" -eq 0 ]; then
    echo "Stripe ${LOCAL_ID} has no runs to render (sweep may be incomplete) — done."
    exit 0
fi

echo ""
echo "Rendering ${#RUN_DIRS[@]} run(s) in this stripe:"
printf '  %s\n' "${RUN_DIRS[@]}"
echo ""

FAILED=()
for run_dir in "${RUN_DIRS[@]}"; do
    echo "=== report_diagnostics: $(basename "$run_dir") ==="
    # Don't let one bad run abort the stripe (set -e would otherwise kill us).
    if PYTHONPATH="$HOME/CV-Quixer${PYTHONPATH:+:$PYTHONPATH}" \
        uv run --no-sync python -u experiments/report_diagnostics.py \
            --run-dir "$run_dir" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}; then
        echo "    ok"
    else
        rc=$?
        echo "    FAILED (exit $rc): $run_dir"
        FAILED+=("$run_dir")
    fi
done

echo ""
echo "Finished stripe ${LOCAL_ID}: $(date)"
if [ "${#FAILED[@]}" -gt 0 ]; then
    echo "${#FAILED[@]} run(s) failed in this stripe:"
    printf '  %s\n' "${FAILED[@]}"
    exit 1
fi
echo "All ${#RUN_DIRS[@]} run(s) in this stripe rendered cleanly."

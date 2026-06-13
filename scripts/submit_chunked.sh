#!/bin/bash
# -----------------------------------------------------------------------
# Self-chaining chunked submitter for sweep manifests under a MaxSubmitJobs
# queue limit (every pending+running array task counts toward the cap).
#
# Submits array tasks [START, START+CHUNK-1] of run_sweep.sh for MANIFEST,
# then re-queues ITSELF for the chunk WINDOW positions ahead, with
# --dependency=afterany on the chunk just submitted. No login-node process
# stays alive between chunks — the chain lives entirely inside SLURM.
#
# WINDOW=1 (default): strictly serial chunks — chunk N+1 is submitted only
# after chunk N fully drains. Simple, but GPUs idle while a chunk's last
# straggler finishes.
#
# WINDOW>1: a sliding window — W independent chains are bootstrapped, so W
# chunks are in the queue at once and the scheduler backfills across all of
# them (MaxJobs caps how many RUN; eligible backlog stays full). When chunk
# N drains, chunk N+W is submitted. GPUs only idle if one straggler outlasts
# every newer in-flight chunk.
#
# Queue accounting: peak = WINDOW*(CHUNK+1) + 1 must stay <= MaxSubmitJobs
# (W chunks of C tasks + one pending submitter per chunk + one running
# submitter). For MaxSubmitJobs=32:
#     CHUNK=30 WINDOW=1  -> peak 32  (serial)
#     CHUNK=6  WINDOW=4  -> peak 29  (recommended: 24-task backlog, 3 spare)
#     CHUNK=9  WINDOW=3  -> peak 31
#
# Serial kick-off (one command from the repo root, returns in seconds):
#     sbatch scripts/submit_chunked.sh \
#         results/sweeps/<sweep>_<ts>/sweep_manifest.json 0 30 <n_runs>
#
# Windowed kick-off (bootstrap one chain per window position):
#     M=results/sweeps/<sweep>_<ts>/sweep_manifest.json
#     for i in 0 1 2 3; do
#         sbatch scripts/submit_chunked.sh "$M" $((i*6)) 6 <n_runs> 4
#     done
#
# Optional 6th arg = a --gres OVERRIDE; optional 7th arg = a single
# space-separated string of EXTRA sbatch flags. Both are forwarded to every
# chunk's run_sweep.sh array (empty/omitted = run_sweep.sh's #SBATCH defaults:
# a100-40, 8 h). CLI flags override the script's #SBATCH directives, so this
# retargets GPUs/partition/time without editing run_sweep.sh.
#
# The H200 lives ONLY in the `gpu` partition, which caps walltime at 3 h, so an
# H200 (or `gpu`-partition H100) run needs BOTH --partition=gpu AND --time<=3h
# (run_sweep.sh's 8 h would be rejected). Chunk-size to coexist with other
# already-queued jobs under MaxSubmitJobs (peak = chunk+2 for WINDOW=1):
#     # ~11 other tasks queued -> keep chunk+2 <= 21, e.g. chunk 16:
#     sbatch scripts/submit_chunked.sh \
#         results/sweeps/<sweep>_<ts>/resume_manifest_<ts>.json \
#         0 16 28 1 gpu:h200-141:1 "--partition=gpu --time=03:00:00"
# (NOTE: a run that overruns the 3 h cap is wall-time-killed with no run-dir
#  record — re-run such casualties with resume_sweep.py.)
#
# afterany (not afterok) means the chain advances even when chunk tasks
# fail or hit the wall — retry/top-up the casualties afterwards with
# experiments/resume_sweep.py.
#
# Stop everything: scancel --name=sweep_chain --name=cv_quixer_sweep
# -----------------------------------------------------------------------
#SBATCH --job-name=sweep_chain
#SBATCH --output=slurm_logs/slurm-%x-%j.out
#SBATCH --error=slurm_logs/slurm-%x-%j.err
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G

set -euo pipefail

MANIFEST="${1:?usage: sbatch scripts/submit_chunked.sh <sweep_manifest.json> <start> <chunk> <total> [window] [gres] [extra_sbatch_opts]}"
START="${2:?missing <start> (first array index of this chunk, e.g. 0)}"
CHUNK="${3:?missing <chunk> (chunk size; keep window*(chunk+1)+1 <= MaxSubmitJobs)}"
TOTAL="${4:?missing <total> (n_runs from the manifest)}"
WINDOW="${5:-1}"
GRES="${6:-}"    # optional --gres override (e.g. gpu:h200-141:1); empty = run_sweep.sh default
EXTRA="${7:-}"   # optional extra sbatch flags, ONE space-separated string, e.g.
                 # "--partition=gpu --time=03:00:00" (flag VALUES must contain no spaces)

cd "$HOME/CV-Quixer"

END=$(( START + CHUNK - 1 ))
[ "$END" -ge "$TOTAL" ] && END=$(( TOTAL - 1 ))

# ${GRES:+--gres=$GRES} expands to nothing when GRES is empty (preserving the
# historic behaviour: run_sweep.sh's own #SBATCH --gres default applies) and to
# a single --gres=... word otherwise (gres strings have no spaces). $EXTRA is
# deliberately UNQUOTED so a multi-flag string word-splits into separate args
# (e.g. --partition=gpu + --time=03:00:00; needed because the 3 h `gpu`
# partition that hosts the H200 rejects run_sweep.sh's 8 h #SBATCH --time). CLI
# flags override the script's #SBATCH directives, so this retargets every chunk.
JOB_ID=$(sbatch --parsable ${GRES:+--gres="$GRES"} ${EXTRA} \
    --array="${START}-${END}" scripts/run_sweep.sh "$MANIFEST")
echo "submitted chunk ${START}-${END} of 0-$(( TOTAL - 1 )) as job ${JOB_ID}  (window=${WINDOW}, gres=${GRES:-<default>}, extra='${EXTRA}', $(date))"

# This chain's next chunk sits WINDOW chunk-widths ahead; the WINDOW-1
# chunks in between belong to the other bootstrapped chains. GRES + EXTRA are
# forwarded (EXTRA quoted so it stays ONE arg) so every chunk inherits them.
NEXT=$(( START + CHUNK * WINDOW ))
if [ "$NEXT" -lt "$TOTAL" ]; then
    CHAIN_ID=$(sbatch --parsable --dependency="afterany:${JOB_ID}" \
        scripts/submit_chunked.sh "$MANIFEST" "$NEXT" "$CHUNK" "$TOTAL" "$WINDOW" "$GRES" "$EXTRA")
    echo "chained next submitter (start=${NEXT}) as job ${CHAIN_ID}, fires after ${JOB_ID} drains"
else
    echo "chain lane complete (next start ${NEXT} >= ${TOTAL})"
fi

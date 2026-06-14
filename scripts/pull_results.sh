#!/bin/bash
# -----------------------------------------------------------------------
# Tiered, additive pull of run/sweep artefacts FROM the cluster to local.
#
# This is the repo's only LOCAL-side script — every other scripts/*.sh runs
# on the cluster. It rsyncs a chosen ARTEFACT TIER (see CONTEXT.md and
# docs/adr/0005) of one or more repo-relative paths down to the identical
# local path, so you pull only what local results-analysis needs and leave
# the heavy raw artefacts (esp. the ~94 MB/epoch *_train.npz) on the cluster.
#
# Premise: report_sweep.py / report_diagnostics.py run ON THE CLUSTER and
# write summary.csv/md + figures/ in-place, so every derived figure already
# exists remotely; this tool just brings the figures (+ optionally the raw
# data to re-derive them) down.
#
# TIER LADDER (each rung is a superset of the one before):
#   figures           summary.csv/md + every .png/.txt under figures/ + manifests
#   light  (default)  + config.json, history.json, parameter_table.txt,
#                       subset_indices.npz, debug/, logs/   (small text/raw)
#   excl_train_ckpt   + test predictions/*.npz, diagnostics/*.npz, test_images.npz
#                       (everything EXCEPT the training payload)
#   full              + checkpoints/ and per-epoch *_train.npz  (complete mirror)
#
# The "training payload" (checkpoints/ + *_train.npz) is the heaviest artefact
# and is only needed to resume training or re-derive train-side figures, so it
# rides in `full` only.
#
# REMOTE is read from scripts/.pull_config (gitignored; copy from
# scripts/.pull_config.example). Required: REMOTE=user@host. Optional:
# REMOTE_ROOT (default ~/CV-Quixer).
#
# Usage:
#     bash scripts/pull_results.sh <repo-relative-path>... [options]
#
#   <repo-relative-path>   e.g. results/sweeps/<sweep>_<ts>/  (1 or more).
#                          Mirrors to the identical local path.
# Options:
#   --tier T     figures | light | excl_train_ckpt | full   (default: light)
#   --dry-run    show what rsync/prune would do; transfer/delete nothing
#   --prune      after the (additive) pull, DELETE local artefacts that live
#                in tiers ABOVE --tier (whitelist of heavy patterns only;
#                never touches figures/tables/config). Reclaims disk when you
#                drop a sweep back down a tier. Prompts unless --yes.
#   --yes        skip the --prune confirmation prompt
#   -h|--help    this help
#
# Examples:
#   # default light pull of a whole sweep:
#   bash scripts/pull_results.sh results/sweeps/grid_quantum_<ts>/
#
#   # pull the raw eval data too, to re-derive test-side figures locally:
#   bash scripts/pull_results.sh results/sweeps/grid_quantum_<ts>/ --tier excl_train_ckpt
#
#   # preview a full mirror of one run:
#   bash scripts/pull_results.sh results/runs/full_fashionmnist_<ts>/ --tier full --dry-run
#
#   # reclaim space: drop a sweep back to light, deleting the heavy npz:
#   bash scripts/pull_results.sh results/sweeps/grid_quantum_<ts>/ --tier light --prune
# -----------------------------------------------------------------------

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

usage() { sed -n '2,/^# ----.*$/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

TIER="light"
DRY_RUN=0
PRUNE=0
ASSUME_YES=0
PATHS=()

while [ $# -gt 0 ]; do
    case "$1" in
        --tier)     TIER="${2:?--tier needs a value}"; shift 2 ;;
        --tier=*)   TIER="${1#*=}"; shift ;;
        --dry-run)  DRY_RUN=1; shift ;;
        --prune)    PRUNE=1; shift ;;
        --yes|-y)   ASSUME_YES=1; shift ;;
        -h|--help)  usage; exit 0 ;;
        -*)         echo "unknown option: $1" >&2; exit 2 ;;
        *)          PATHS+=("$1"); shift ;;
    esac
done

case "$TIER" in
    figures|light|excl_train_ckpt|full) ;;
    *) echo "invalid --tier '$TIER' (figures|light|excl_train_ckpt|full)" >&2; exit 2 ;;
esac

[ "${#PATHS[@]}" -ge 1 ] || { echo "error: need at least one repo-relative path" >&2; echo; usage; exit 2; }

# --- remote config -----------------------------------------------------
CONFIG="$REPO_ROOT/scripts/.pull_config"
if [ ! -f "$CONFIG" ]; then
    echo "error: $CONFIG not found." >&2
    echo "  cp scripts/.pull_config.example scripts/.pull_config  and set REMOTE=user@host" >&2
    exit 1
fi
# shellcheck disable=SC1090
source "$CONFIG"
: "${REMOTE:?set REMOTE=user@host in scripts/.pull_config}"
REMOTE_ROOT="${REMOTE_ROOT:-CV-Quixer}"   # relative to the REMOTE home by default
# A `~` (or $HOME) in .pull_config expands on the LOCAL machine when sourced
# (-> /Users/you/...), which the cluster has no such path for. Normalise a
# local-home or literal-~ prefix back to a remote-home-relative path so the
# remote shell resolves it. An explicit absolute remote path (/home/...) is
# left untouched.
case "$REMOTE_ROOT" in
    "$HOME"/*) REMOTE_ROOT="${REMOTE_ROOT#"$HOME"/}" ;;   # /Users/you/CV-Quixer -> CV-Quixer
    "$HOME")   REMOTE_ROOT="." ;;
    "~/"*)     REMOTE_ROOT="${REMOTE_ROOT#"~/"}" ;;        # literal ~/CV-Quixer -> CV-Quixer
    "~")       REMOTE_ROOT="." ;;
esac

# --- tier -> rsync filter rules ----------------------------------------
# Inclusion tiers (figures, light) whitelist files: descend into every dir
# (+ */), keep the wanted patterns, drop the rest (- *), then --prune-empty-dirs
# removes the skeleton dirs left behind. Exclusion tiers (excl_train_ckpt,
# full) start from everything and drop only the heavy bits.
# All glob-bearing patterns are single-quoted so pathname expansion never fires
# (the function runs with CWD = repo root); rsync, not the shell, interprets them.
rsync_filters() {
    case "$TIER" in
        figures)
            printf '%s\n' \
                '--prune-empty-dirs' \
                '--include=*/' \
                '--include=figures/**' \
                '--include=summary.csv' '--include=summary.md' \
                '--include=*_manifest.json' \
                '--include=resume_manifest_*.json' \
                '--include=cutoff_summary.csv' '--include=cutoff_summary.md' \
                '--exclude=*' ;;
        light)
            printf '%s\n' \
                '--prune-empty-dirs' \
                '--include=*/' \
                '--include=figures/**' \
                '--include=summary.csv' '--include=summary.md' \
                '--include=*_manifest.json' \
                '--include=resume_manifest_*.json' \
                '--include=cutoff_summary.csv' '--include=cutoff_summary.md' \
                '--include=config.json' '--include=history.json' \
                '--include=meta.json' \
                '--include=parameter_table.txt' \
                '--include=subset_indices.npz' \
                '--include=debug/**' \
                '--include=logs/**' \
                '--exclude=*' ;;
        excl_train_ckpt)
            printf '%s\n' \
                '--exclude=*_train.npz' \
                '--exclude=checkpoints/' \
                '--exclude=.DS_Store' ;;
        full)
            printf '%s\n' '--exclude=.DS_Store' ;;
    esac
}

# -P = --partial --progress: resumable + per-file progress, supported by both
# GNU rsync and macOS's openrsync. --info=progress2 (one aggregate bar) is a
# GNU rsync >= 3.1 flag absent from openrsync / old rsync, so add it only when
# this rsync advertises it (else it errors out: "unrecognized option").
RSYNC_OPTS=(-a -P)                    # no -z: png/npz already compressed
if rsync --help 2>&1 | grep -q 'info='; then RSYNC_OPTS+=(--info=progress2); fi
[ "$DRY_RUN" -eq 1 ] && RSYNC_OPTS+=(--dry-run)
# read into array via a loop (not `mapfile` — absent on macOS's bash 3.2)
FILTERS=()
while IFS= read -r line; do FILTERS+=("$line"); done < <(rsync_filters)

echo "tier=$TIER  remote=$REMOTE:$REMOTE_ROOT  dry_run=$DRY_RUN  prune=$PRUNE"
echo

# --- pull --------------------------------------------------------------
for p in "${PATHS[@]}"; do
    rel="${p#./}"; rel="${rel%/}"           # normalise: strip ./ and trailing /
    src="$REMOTE:$REMOTE_ROOT/$rel/"         # trailing slashes => sync dir contents
    dst="$REPO_ROOT/$rel/"
    echo "==> $rel"
    [ "$DRY_RUN" -eq 1 ] || mkdir -p "$dst"
    rsync "${RSYNC_OPTS[@]}" "${FILTERS[@]}" "$src" "$dst"
    echo
done

[ "$PRUNE" -eq 0 ] && exit 0

# --- prune: delete local artefacts in tiers ABOVE --tier ---------------
# Whitelist of heavy patterns, cumulative down the ladder. find never touches
# figures/, summary.*, manifests, or anything not named here.
prune_find_args() {   # emits a find expression (OR-joined) for the current tier
    local exprs=()
    # tiers strictly above `full`: none
    if [ "$TIER" != "full" ]; then
        exprs+=( -type d -name checkpoints -o -type f -name '*_train.npz' )
    fi
    if [ "$TIER" = "figures" ] || [ "$TIER" = "light" ]; then
        exprs+=( -o -type d -name predictions -o -type d -name diagnostics )
    fi
    if [ "$TIER" = "figures" ]; then
        exprs+=( -o -type f -name config.json -o -type f -name history.json \
                 -o -type f -name parameter_table.txt -o -type f -name subset_indices.npz \
                 -o -type d -name debug -o -type d -name logs )
    fi
    printf '%s\n' "${exprs[@]}"
}

PRUNE_EXPR=()
while IFS= read -r line; do PRUNE_EXPR+=("$line"); done < <(prune_find_args)
if [ "${#PRUNE_EXPR[@]}" -eq 0 ]; then
    echo "prune: nothing above tier '$TIER' — nothing to delete."
    exit 0
fi

# collect matches across all target paths
VICTIMS=()
for p in "${PATHS[@]}"; do
    rel="${p#./}"; rel="${rel%/}"
    [ -d "$REPO_ROOT/$rel" ] || continue
    while IFS= read -r line; do VICTIMS+=("$line"); done \
        < <(find "$REPO_ROOT/$rel" \( "${PRUNE_EXPR[@]}" \) -prune -print)
done

if [ "${#VICTIMS[@]}" -eq 0 ]; then
    echo "prune: no local artefacts above tier '$TIER' present — nothing to delete."
    exit 0
fi

echo "prune: these local paths are above tier '$TIER' and will be DELETED:"
printf '    %s\n' "${VICTIMS[@]}"
if [ "$DRY_RUN" -eq 1 ]; then
    echo "(--dry-run: nothing deleted)"
    exit 0
fi
if [ "$ASSUME_YES" -eq 0 ]; then
    printf "proceed? type 'yes': "
    read -r ans
    [ "$ans" = "yes" ] || { echo "aborted."; exit 1; }
fi
for v in "${VICTIMS[@]}"; do rm -rf "$v"; done
echo "pruned ${#VICTIMS[@]} path(s)."

# shellcheck shell=bash
# -----------------------------------------------------------------------
# Shared CUDA environment setup for the SLURM scripts. SOURCE this (don't
# execute) from a script that has already `cd "$HOME/CV-Quixer"`:
#
#     source scripts/setup_cuda_env.sh
#
# Provides a working `uv` and the project CUDA venv with NO manual pre-build:
#   - installs/repairs `uv` if missing OR not executable on this node
#     (corrupt download or wrong-arch -> "Exec format error");
#   - builds the venv once, serialised with an flock so concurrent array tasks
#     never race; later tasks reuse it.
#
# Architecture-keyed via $(uname -m): a job array spanning mixed hardware
# (x86_64 / aarch64) gets a separate `uv` binary AND venv per architecture, so
# they never share an incompatible binary or set of compiled wheels.
#
# Env knobs:
#   REBUILD_VENV=1   force a clean rebuild of this arch's venv
#                    (e.g. `sbatch --export=ALL,REBUILD_VENV=1 ...`).
#
# Exports UV_PROJECT_ENVIRONMENT (the per-arch venv) for subsequent `uv run`.
# Uses `if` (not `&&`) so it is safe under the caller's `set -euo pipefail`.
# -----------------------------------------------------------------------

ARCH="$(uname -m)"

# --- uv: per-arch install dir; repair if missing or not executable here ---
export UV_INSTALL_DIR="$HOME/.local/uv/$ARCH"
export UV_NO_MODIFY_PATH=1          # we manage PATH ourselves (below)
export PATH="$UV_INSTALL_DIR:$PATH"
if ! uv --version &> /dev/null; then
    echo "[setup] installing/repairing uv for $ARCH ..."
    rm -f "$UV_INSTALL_DIR/uv" "$UV_INSTALL_DIR/uvx"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$UV_INSTALL_DIR:$PATH"
fi

# --- venv: per-arch; build once under an flock so concurrent tasks don't race ---
export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/cv-quixer-cuda-$ARCH"
mkdir -p "$HOME/.venvs"
if [ "${REBUILD_VENV:-0}" = "1" ]; then
    echo "[setup] REBUILD_VENV=1 -> removing $UV_PROJECT_ENVIRONMENT"
    rm -rf "$UV_PROJECT_ENVIRONMENT"
fi
(
    flock 9
    if [ ! -x "$UV_PROJECT_ENVIRONMENT/bin/python" ]; then
        echo "[setup] building venv for $ARCH (first task builds; others wait) ..."
        uv sync
    fi
) 9> "$HOME/.venvs/cv-quixer-cuda-$ARCH.lock"

echo "[setup] uv $(uv --version) | venv $UV_PROJECT_ENVIRONMENT ready ($ARCH)"

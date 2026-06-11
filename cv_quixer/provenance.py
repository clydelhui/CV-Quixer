"""Launch provenance — record the exact command that started an entry point.

Every experiment entry point appends one *invocation* record (see CONTEXT.md:
Invocation) to the artefact it already owns — ``sweep_manifest.json``,
``cutoff_sweep_manifest.json``, ``history["meta"]``, or the eval ``meta.json`` —
so any artefact can answer "what command produced this?" without
reverse-engineering argv from the stored config. The ``invocations`` key is an
append-only list: entry 0 is the launch that created the artefact; each resume
or top-up appends. Purely additive — readers tolerate artefacts that pre-date
the key.

Top-level module rather than ``utils/`` so the torch-free orchestrators
(``sweep.py``, ``eval_cutoff_sweep_all.py``) can import it without pulling in
``utils.params``' torch dependency via ``utils/__init__``.
"""

from __future__ import annotations

import platform
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _git(*args: str) -> str | None:
    """Run a git command at the repo root; None on any failure (no git, no repo)."""
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout if result.returncode == 0 else None


def invocation_record() -> dict:
    """Build one invocation entry for the calling script.

    ``argv`` is ``sys.argv`` verbatim (machine-readable, lossless); ``command``
    is the shell-quoted re-run line under the project's ``uv run python``
    convention. Git fields are None when git or the repo is unavailable
    (e.g. a stripped cluster node).
    """
    sha = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain") if sha is not None else None
    return {
        "launched_at": datetime.now().isoformat(timespec="seconds"),
        "argv": list(sys.argv),
        "command": "uv run python " + shlex.join(sys.argv),
        "hostname": platform.node(),
        "git_sha": sha.strip() if sha is not None else None,
        "git_dirty": bool(status.strip()) if status is not None else None,
    }

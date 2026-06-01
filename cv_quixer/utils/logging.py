"""Experiment logging setup (wandb and/or console).

wandb is optional: every function is a no-op unless `use_wandb` is set, and the
`wandb` package is imported lazily so a console-only run needs nothing installed
at import time. On cluster nodes without outbound network, set
`WANDB_MODE=offline` and sync later with `wandb sync`.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Optional

from cv_quixer.config.schema import ExperimentConfig


def init_logging(
    config: ExperimentConfig,
    *,
    group: Optional[str] = None,
    tags: Optional[Sequence[str]] = None,
    name: Optional[str] = None,
    dir: Optional[str | Path] = None,
) -> None:
    """Initialise wandb if enabled in config, otherwise do nothing.

    Args:
        config: ExperimentConfig with `name` and `use_wandb`. The full config is
                captured (`config=asdict(config)`), which powers wandb's
                cross-run parallel-coordinates / scatter views over a sweep.
        group:  wandb run group — pass the sweep name so all runs in one sweep
                are grouped together in the UI.
        tags:   wandb tags, e.g. ["params=13760", "obs=xpxsps"].
        name:   wandb run display name (defaults to `config.name`).
        dir:    Directory for wandb's local files (point at the run dir so its
                artefacts live alongside the run's other outputs).
    """
    if not config.use_wandb:
        return

    import wandb
    from dataclasses import asdict

    wandb.init(
        project="cv-quixer",
        name=name or config.name,
        group=group,
        tags=list(tags) if tags else None,
        dir=str(dir) if dir is not None else None,
        config=asdict(config),
    )

    # Custom x-axes so per-epoch and per-batch series don't fight over the
    # global `step` counter. Callers include the axis value in the logged dict
    # (e.g. {"epoch": e, "epoch/test_acc": ...}) and omit `step`.
    wandb.define_metric("epoch")
    wandb.define_metric("batch_step")
    wandb.define_metric("epoch/*", step_metric="epoch")
    wandb.define_metric("batch/*", step_metric="batch_step")


def log_metrics(
    metrics: dict[str, float],
    use_wandb: bool,
    *,
    step: Optional[int] = None,
) -> None:
    """Push a dict of scalar metrics to wandb (no-op if `use_wandb` is False).

    Args:
        metrics:   Metric name → value. Prefix keys with `epoch/` or `batch/`
                   and include the matching axis value (`epoch` / `batch_step`)
                   to bind them to the custom x-axes set up in `init_logging`.
        use_wandb: Whether wandb logging is active.
        step:      Optional explicit wandb step. Usually omit it and rely on the
                   `define_metric` axes instead.
    """
    if not use_wandb:
        return

    import wandb

    if step is not None:
        wandb.log(metrics, step=step)
    else:
        wandb.log(metrics)


def finish_logging(use_wandb: bool) -> None:
    """Close the wandb run if active."""
    if use_wandb:
        import wandb
        wandb.finish()

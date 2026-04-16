"""Experiment logging setup (wandb and/or console)."""

from __future__ import annotations

from cv_quixer.config.schema import ExperimentConfig


def init_logging(config: ExperimentConfig) -> None:
    """Initialise wandb if enabled in config, otherwise use console only.

    Args:
        config: ExperimentConfig with name and use_wandb flag.
    """
    if config.use_wandb:
        import wandb
        from dataclasses import asdict
        wandb.init(project="cv-quixer", name=config.name, config=asdict(config))


def log_metrics(metrics: dict[str, float], step: int, use_wandb: bool) -> None:
    """Log a dict of scalar metrics at a given step.

    Args:
        metrics:   Dict mapping metric name to float value.
        step:      Current step/epoch number.
        use_wandb: Whether to push to wandb in addition to printing.
    """
    if use_wandb:
        import wandb
        wandb.log(metrics, step=step)


def finish_logging(use_wandb: bool) -> None:
    """Close the wandb run if active."""
    if use_wandb:
        import wandb
        wandb.finish()

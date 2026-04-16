import json
from dataclasses import asdict
from pathlib import Path

import dacite
import yaml

from cv_quixer.config.schema import ExperimentConfig


def load_config(path: str | Path, overrides: dict | None = None) -> ExperimentConfig:
    """Load an ExperimentConfig from a YAML file with optional key overrides.

    Args:
        path: Path to a YAML config file. Values here override schema defaults.
        overrides: Dict of dot-notation overrides, e.g. {"training.lr": 1e-4}.
                   Useful for quick ablations without duplicating YAML files.

    Returns:
        A fully populated ExperimentConfig dataclass.
    """
    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    if overrides:
        raw = _apply_overrides(raw, overrides)

    return dacite.from_dict(
        data_class=ExperimentConfig,
        data=raw,
        config=dacite.Config(strict=False),
    )


def save_config(config: ExperimentConfig, path: str | Path) -> None:
    """Serialize a config to JSON alongside a checkpoint for full reproducibility."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(asdict(config), f, indent=2)


def _apply_overrides(raw: dict, overrides: dict) -> dict:
    """Apply dot-notation overrides to a nested dict (mutates and returns raw)."""
    for key, value in overrides.items():
        parts = key.split(".")
        d = raw
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        d[parts[-1]] = value
    return raw

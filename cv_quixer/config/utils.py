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


def experiment_config_from_dict(raw: dict) -> ExperimentConfig:
    """Reconstruct an ExperimentConfig from a saved config.json dict via dacite,
    with a loud guard for pre-CVQNN runs.

    The CVQNN block W (``QuantumConfig.cvqnn_num_layers``, default 1) post-dates
    the original frozen model. dacite fills a missing field with the dataclass
    *default* — so an old ``config.json`` (no ``cvqnn_num_layers`` key) would be
    silently rebuilt with W on (``cvqnn_num_layers=1``) and then fail to match its
    W-free checkpoint with a cryptic ``state_dict`` key mismatch.

    This guard refuses that path *loudly* and *without fabricating a value*: it
    raises with a hint to run the one-shot migration that bakes
    ``cvqnn_num_layers: 0`` / ``cvqnn_trunc_lambda: 0.0`` into the old config
    (after which the run rebuilds correctly as a pre-W model). It deliberately
    does NOT inject a default itself — a silent "absence means 0" shim would be a
    permanent footgun masking future config bugs. Mirrors the existing
    "run backfill_artefacts.py first" idiom in report_diagnostics.py.

    Args:
        raw: The parsed config.json dict (the ``ExperimentConfig`` shape).

    Returns:
        The reconstructed ExperimentConfig.

    Raises:
        ValueError: if the ``quantum`` block lacks ``cvqnn_num_layers``.
    """
    quantum = raw.get("quantum", {})
    if isinstance(quantum, dict) and "cvqnn_num_layers" not in quantum:
        raise ValueError(
            "This run predates the CVQNN block W (config.json has no "
            "quantum.cvqnn_num_layers). Run "
            "`uv run python experiments/migrate_add_cvqnn_field.py "
            "--runs-root <dir>` first to bake cvqnn_num_layers=0 / "
            "cvqnn_trunc_lambda=0.0 into the old config(s); the run then rebuilds "
            "correctly as a pre-W (W-free) model that loads its existing "
            "checkpoint."
        )
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

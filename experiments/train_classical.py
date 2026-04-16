"""Train the classical ViT baseline.

Usage:
    uv run python experiments/train_classical.py --config configs/classical_vit.yaml
    uv run python experiments/train_classical.py --config configs/classical_vit.yaml \
        --overrides training.lr=5e-4 training.epochs=10
"""

import argparse
import json
from pathlib import Path

from cv_quixer.config.utils import load_config, save_config
from cv_quixer.data.mnist import get_dataloaders
from cv_quixer.models import build_model
from cv_quixer.training.trainer import Trainer
from cv_quixer.utils.logging import finish_logging, init_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the classical ViT baseline")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument(
        "--overrides", nargs="*", default=[],
        help="Key=value overrides, e.g. training.lr=1e-4",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides = dict(kv.split("=", 1) for kv in args.overrides)
    config = load_config(args.config, overrides or None)

    assert config.model == "classical", (
        f"Config model is '{config.model}', expected 'classical'. "
        f"Use experiments/train_quantum.py for the quantum model."
    )

    init_logging(config)

    train_loader, test_loader = get_dataloaders(config.data)
    model = build_model(config)

    print(f"Model: Classical ViT | Parameters: {model.get_num_parameters():,}")

    trainer = Trainer(model, config, train_loader, test_loader)
    history = trainer.train()

    # Save config alongside results for reproducibility
    out_dir = Path(config.training.checkpoint_dir) / config.name
    save_config(config, out_dir / "config.json")
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    finish_logging(config.use_wandb)
    print(f"\nBest test accuracy: {max(history['test_acc']):.4f}")


if __name__ == "__main__":
    main()

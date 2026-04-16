"""CLI entry point for CV-Quixer.

Delegates to the experiment scripts. For full options use the scripts directly:
    uv run python experiments/train_quantum.py --help
    uv run python experiments/train_classical.py --help
    uv run python experiments/compare_models.py --help
"""

import argparse
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CV-Quixer: Continuous Variable Quantum Vision Transformer"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_q = subparsers.add_parser("train-quantum", help="Train the CV-Quixer model")
    train_q.add_argument("--config", required=True)
    train_q.add_argument("--overrides", nargs="*", default=[])

    train_c = subparsers.add_parser("train-classical", help="Train the classical ViT baseline")
    train_c.add_argument("--config", required=True)
    train_c.add_argument("--overrides", nargs="*", default=[])

    compare = subparsers.add_parser("compare", help="Compare model results")
    compare.add_argument("--classical", required=True)
    compare.add_argument("--quantum", required=True)
    compare.add_argument("--save", default="results/figures/comparison.png")

    args = parser.parse_args()

    if args.command == "train-quantum":
        cmd = ["python", "experiments/train_quantum.py", "--config", args.config]
        if args.overrides:
            cmd += ["--overrides"] + args.overrides
    elif args.command == "train-classical":
        cmd = ["python", "experiments/train_classical.py", "--config", args.config]
        if args.overrides:
            cmd += ["--overrides"] + args.overrides
    elif args.command == "compare":
        cmd = [
            "python", "experiments/compare_models.py",
            "--classical", args.classical,
            "--quantum", args.quantum,
            "--save", args.save,
        ]

    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()

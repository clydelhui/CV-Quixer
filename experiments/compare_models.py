"""Compare classical ViT and CV-Quixer results and produce thesis figures.

Usage:
    uv run python experiments/compare_models.py \
        --classical results/checkpoints/classical_vit_baseline/history.json \
        --quantum   results/checkpoints/cv_quixer_baseline/history.json \
        --classical-params 41354 \
        --quantum-params 1234
"""

import argparse
import json

from cv_quixer.evaluation.compare import plot_training_curves, print_comparison_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare model results")
    parser.add_argument("--classical", required=True, help="Path to classical history.json")
    parser.add_argument("--quantum", required=True, help="Path to quantum history.json")
    parser.add_argument("--classical-params", type=int, default=0)
    parser.add_argument("--quantum-params", type=int, default=0)
    parser.add_argument(
        "--save", default="results/figures/comparison.png",
        help="Where to save the comparison figure",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.classical) as f:
        classical_history = json.load(f)
    with open(args.quantum) as f:
        quantum_history = json.load(f)

    print_comparison_table(
        classical_history, quantum_history,
        args.classical_params, args.quantum_params,
    )
    plot_training_curves(classical_history, quantum_history, save_path=args.save)


if __name__ == "__main__":
    main()

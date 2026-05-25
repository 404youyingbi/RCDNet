import argparse
import ast
import subprocess
import sys
from pathlib import Path
from typing import Tuple

PROJECT_ROOT = Path(__file__).resolve().parent

DATASET_CHOICES = ("LOLv1", "LOLv2_Real", "LOLv2_Synthetic")


def parse_shape(shape_text: str) -> Tuple[int, int, int]:
    """Parse an input shape string such as '(256,256,3)' or '256,256,3'."""
    try:
        value = ast.literal_eval(shape_text)
    except (SyntaxError, ValueError):
        value = tuple(int(part.strip()) for part in shape_text.split(","))

    if not isinstance(value, (tuple, list)) or len(value) != 3:
        raise argparse.ArgumentTypeError("shape must have the form '(H,W,C)'")

    shape = tuple(int(dim) for dim in value)
    if any(dim <= 0 for dim in shape):
        raise argparse.ArgumentTypeError("all shape dimensions must be positive")
    return shape


def run_train(args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "train.py"),
        "--dataset",
        args.dataset,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--seed",
        str(args.seed),
    ]
    subprocess.run(cmd, check=True)


def run_test(args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "test.py"),
        "--dataset",
        args.dataset,
        "--weights",
        args.weights,
        "--output-dir",
        args.output_dir,
    ]
    subprocess.run(cmd, check=True)


def run_complexity_check(args: argparse.Namespace) -> None:
    shape = f"({args.shape[0]},{args.shape[1]},{args.shape[2]})"
    subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / "complexity_check.py"), "--shape", shape], check=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run RCDNet training, testing, or complexity analysis.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train the model.")
    train_parser.add_argument("--dataset", choices=DATASET_CHOICES, required=True)
    train_parser.add_argument("--epochs", type=int, default=1000)
    train_parser.add_argument("--batch-size", type=int, default=1)
    train_parser.add_argument("--seed", type=int, default=100)
    train_parser.set_defaults(func=run_train)

    test_parser = subparsers.add_parser("test", help="Evaluate a trained model.")
    test_parser.add_argument("--dataset", choices=DATASET_CHOICES, required=True)
    test_parser.add_argument("--weights", type=str, required=True, help="Path to model weights.")
    test_parser.add_argument("--output-dir", type=str, default="./results", help="Directory for enhanced images.")
    test_parser.set_defaults(func=run_test)

    complexity_parser = subparsers.add_parser("complexity", help="Compute FLOPs and parameter count.")
    complexity_parser.add_argument("--shape", type=parse_shape, default=(256, 256, 3), help="Input shape, e.g. '(256,256,3)'.")
    complexity_parser.set_defaults(func=run_complexity_check)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

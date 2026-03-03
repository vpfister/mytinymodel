"""CLI entrypoint for the tiny GPT-2 model with train and eval subcommands."""

import argparse
import logging

from .evaluator import evaluate
from .model import TinyGPT2
from .trainer import train

logger = logging.getLogger(__name__)


def main():
    """Run the main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Tiny GPT-2 Model - Train and evaluate a small GPT-2 like model"
    )
    subparsers = parser.add_subparsers(
        dest="command", required=True, help="Available commands"
    )

    # Train subcommand
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument(
        "--dataset",
        type=str,
        default="imdb",
        help="Dataset name to use for training (default: imdb)",
    )
    train_parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training (default: 32)",
    )
    train_parser.add_argument(
        "--epochs", type=int, default=3, help="Number of training epochs (default: 3)"
    )
    train_parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate (default: 1e-3)",
    )
    train_parser.add_argument(
        "--max-seq-length",
        type=int,
        default=128,
        help="Maximum sequence length (default: 128)",
    )

    # Eval subcommand
    eval_parser = subparsers.add_parser("eval", help="Evaluate the model")
    eval_parser.add_argument(
        "--dataset",
        type=str,
        default="imdb",
        help="Dataset name to use for evaluation (default: imdb)",
    )
    eval_parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation (default: 32)",
    )
    eval_parser.add_argument(
        "--max-seq-length",
        type=int,
        default=128,
        help="Maximum sequence length (default: 128)",
    )

    args = parser.parse_args()

    if args.command == "train":
        logger.info("Starting training...")
        train(
            dataset_name=args.dataset,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            learning_rate=args.learning_rate,
            max_seq_length=args.max_seq_length,
        )
    elif args.command == "eval":
        logger.info("Starting evaluation...")
        # Try to load trained model weights if they exist
        import os

        import torch

        model_weights_path = "trained_model_weights.pth"

        if os.path.exists(model_weights_path):
            logger.info(f"Loading trained model weights from: {model_weights_path}")
            model = TinyGPT2(vocab_size=50257)
            model.load_state_dict(torch.load(model_weights_path))
        else:
            logger.info("No trained model weights found, using dummy model")
            model = TinyGPT2(vocab_size=50257)

        evaluate(
            model,
            dataset_name=args.dataset,
            batch_size=args.batch_size,
            max_seq_length=args.max_seq_length,
        )


if __name__ == "__main__":
    main()

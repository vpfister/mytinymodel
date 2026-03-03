"""CLI entrypoint for the tiny GPT-2 model with train and eval subcommands."""

import argparse
import logging

from .evaluator import evaluate
from .trainer import train
from .utils import load_trained_model_if_exists

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
        # Load trained model weights if they exist, otherwise create new model
        model, loaded_from_file = load_trained_model_if_exists(vocab_size=50257)
        if loaded_from_file:
            logger.info("Using trained model weights")
        else:
            logger.info("Using new model (no trained weights found)")

        evaluate(
            model,
            dataset_name=args.dataset,
            batch_size=args.batch_size,
            max_seq_length=args.max_seq_length,
        )


if __name__ == "__main__":
    main()

"""Utility functions for the tiny GPT-2 model."""

import logging
import os

import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Set specific packages to WARNING level only
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("datasets").setLevel(logging.WARNING)
logging.getLogger("wandb").setLevel(logging.WARNING)


def get_device() -> torch.device:
    """Get the appropriate device (CUDA if available, otherwise CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_and_tokenize_dataset(
    dataset_name: str = "imdb",
    max_seq_length: int = 128,
    split: str = "train",
) -> tuple:
    """Load and tokenize a dataset.

    Returns:
        tuple: (tokenized_dataset, tokenizer)

    """
    from datasets import load_dataset
    from transformers import GPT2Tokenizer

    logger = logging.getLogger(__name__)
    logger.info(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split=split)
    logger.info(f"Dataset loaded: {len(dataset)} samples")

    logger.info("Loading GPT-2 tokenizer")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    logger.info(f"Tokenizer loaded: vocab_size={tokenizer.vocab_size}")

    # Tokenize dataset
    logger.info("Tokenizing dataset")

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_seq_length,
            padding="max_length",
            return_tensors="pt",
        )

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    logger.info("Dataset tokenization completed")

    return tokenized_dataset, tokenizer


def load_and_tokenize_train_val_dataset(
    dataset_name: str = "imdb",
    max_seq_length: int = 128,
    val_fraction: float = 0.1,
) -> tuple:
    """Load and tokenize a dataset, splitting into train and validation sets.

    Returns:
        tuple: (train_dataset, val_dataset, tokenizer)

    """
    from datasets import load_dataset
    from transformers import GPT2Tokenizer

    logger = logging.getLogger(__name__)
    logger.info(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split="train")
    logger.info(f"Dataset loaded: {len(dataset)} samples")

    # Split into train and validation
    split = dataset.train_test_split(test_size=val_fraction)
    train_dataset = split["train"]
    val_dataset = split["test"]
    logger.info(
        f"Split into {len(train_dataset)} train / {len(val_dataset)} val samples"
    )

    logger.info("Loading GPT-2 tokenizer")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    logger.info(f"Tokenizer loaded: vocab_size={tokenizer.vocab_size}")

    # Tokenize datasets
    logger.info("Tokenizing datasets")

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_seq_length,
            padding="max_length",
            return_tensors="pt",
        )

    for ds_name, ds in [("train", train_dataset), ("val", val_dataset)]:
        tokenized = ds.map(tokenize_function, batched=True)
        tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])
        if ds_name == "train":
            train_dataset = tokenized
        else:
            val_dataset = tokenized

    logger.info("Dataset tokenization completed")
    return train_dataset, val_dataset, tokenizer


def load_trained_model_if_exists(vocab_size: int = 50257) -> tuple:
    """Load trained model weights if they exist, otherwise create a new model.

    Returns:
        tuple: (model, loaded_from_file)

    """
    from .model import TinyGPT2

    logger = logging.getLogger(__name__)
    model_weights_path = "trained_model_weights.pth"

    if os.path.exists(model_weights_path):
        logger.info(f"Loading trained model weights from: {model_weights_path}")
        model = TinyGPT2(vocab_size=vocab_size)
        model.load_state_dict(torch.load(model_weights_path))
        return model, True
    else:
        logger.info("No trained model weights found, creating new model")
        return TinyGPT2(vocab_size=vocab_size), False

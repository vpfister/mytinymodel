"""Evaluation logic for the tiny GPT-2 model."""

import logging

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .model import TinyGPT2
from .utils import get_device, load_and_tokenize_dataset

logger = logging.getLogger(__name__)


def evaluate(
    model: TinyGPT2,
    dataset_name: str = "imdb",
    batch_size: int = 32,
    max_seq_length: int = 128,
) -> float:
    """Evaluate the tiny GPT-2 model on a Hugging Face dataset."""
    logger.info("Starting evaluation process")
    logger.info(f"Evaluating on dataset: {dataset_name}")

    device = get_device()
    logger.info(f"Using device: {device}")
    model.eval()
    model.to(device)

    # Load and tokenize dataset
    tokenized_dataset, _ = load_and_tokenize_dataset(
        dataset_name=dataset_name,
        max_seq_length=max_seq_length,
        split="test",
    )

    # Create data loader
    data_loader = DataLoader(tokenized_dataset, batch_size=batch_size)
    logger.info(f"Evaluation data loader created with {len(data_loader)} batches")

    # Evaluation loop
    logger.info("Starting evaluation loop")
    total_loss = 0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        progress_bar = tqdm(data_loader, desc="Evaluating")
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            # attention_mask is loaded but not used in this simple evaluation
            _ = batch["attention_mask"].to(device)

            outputs = model(input_ids)
            loss = criterion(outputs.view(-1, model.vocab_size), input_ids.view(-1))
            total_loss += loss.item()

    average_loss = total_loss / len(data_loader)
    perplexity = torch.exp(torch.tensor(average_loss)).item()

    logger.info(
        f"Evaluation completed - Loss: {average_loss:.4f}, Perplexity: {perplexity:.2f}"
    )

    return perplexity

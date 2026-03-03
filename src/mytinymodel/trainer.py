"""Training logic for the tiny GPT-2 model."""

import logging

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .model import TinyGPT2
from .utils import get_device, load_and_tokenize_dataset

logger = logging.getLogger(__name__)


def train(
    dataset_name: str = "imdb",
    batch_size: int = 32,
    num_epochs: int = 3,
    learning_rate: float = 1e-3,
    max_seq_length: int = 128,
) -> TinyGPT2:
    """Train the tiny GPT-2 model on a Hugging Face dataset."""
    logger.info("Starting training process")
    logger.info(
        f"Configuration: dataset={dataset_name}, batch_size={batch_size}, "
        f"epochs={num_epochs}, lr={learning_rate}"
    )

    device = get_device()
    logger.info(f"Using device: {device}")

    # Load and tokenize dataset
    tokenized_dataset, tokenizer = load_and_tokenize_dataset(
        dataset_name=dataset_name,
        max_seq_length=max_seq_length,
        split="train",
    )

    # Create data loader
    data_loader = DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=True)
    logger.info(f"Data loader created with {len(data_loader)} batches")

    # Initialize model
    logger.info("Initializing model")
    model = TinyGPT2(vocab_size=tokenizer.vocab_size).to(device)
    logger.info(
        f"Model initialized: {sum(p.numel() for p in model.parameters())} parameters"
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    # Training loop
    model.train()
    logger.info("Starting training loop")
    for epoch in range(num_epochs):
        total_loss = 0
        progress_bar = tqdm(data_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            # attention_mask is loaded but not used in this simple training
            _ = batch["attention_mask"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids)
            loss = criterion(outputs.view(-1, model.vocab_size), input_ids.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})

        epoch_loss = total_loss / len(data_loader)
        logger.info(f"Epoch {epoch + 1} completed - Average loss: {epoch_loss:.4f}")

    logger.info("Training completed successfully")

    # Save model weights
    model_weights_path = "trained_model_weights.pth"
    torch.save(model.state_dict(), model_weights_path)
    logger.info(f"Model weights saved to: {model_weights_path}")

    return model

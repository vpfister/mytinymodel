"""Training logic for the tiny GPT-2 model."""

import logging
import math

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .model import TinyGPT2
from .utils import get_device, load_and_tokenize_train_val_dataset

logger = logging.getLogger(__name__)


def _run_validation(
    model: TinyGPT2,
    val_loader: DataLoader,
    criterion: torch.nn.CrossEntropyLoss,
    device: torch.device,
) -> tuple[float, float]:
    """Run validation and return (val_loss, val_perplexity)."""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            outputs = model(input_ids)
            loss = criterion(outputs.view(-1, model.vocab_size), input_ids.view(-1))
            total_loss += loss.item()
    model.train()

    avg_loss = total_loss / len(val_loader)
    perplexity = math.exp(avg_loss)
    return avg_loss, perplexity


def train(
    dataset_name: str = "imdb",
    batch_size: int = 32,
    num_epochs: int = 3,
    learning_rate: float = 1e-3,
    max_seq_length: int = 128,
    validate_every_n_samples: int = 1000,
    use_wandb: bool = True,
) -> TinyGPT2:
    """Train the tiny GPT-2 model on a Hugging Face dataset."""
    logger.info("Starting training process")
    logger.info(
        f"Configuration: dataset={dataset_name}, batch_size={batch_size}, "
        f"epochs={num_epochs}, lr={learning_rate}"
    )

    device = get_device()
    logger.info(f"Using device: {device}")

    # Load and tokenize dataset with train/val split
    train_dataset, val_dataset, tokenizer = load_and_tokenize_train_val_dataset(
        dataset_name=dataset_name,
        max_seq_length=max_seq_length,
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    logger.info(
        f"Data loaders created: {len(train_loader)} train batches, "
        f"{len(val_loader)} val batches"
    )

    # Initialize model
    logger.info("Initializing model")
    model = TinyGPT2(vocab_size=tokenizer.vocab_size).to(device)
    logger.info(
        f"Model initialized: {sum(p.numel() for p in model.parameters())} parameters"
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    # Initialize wandb
    if use_wandb:
        import wandb

        wandb.init(
            project="mytinymodel",
            config={
                "dataset": dataset_name,
                "batch_size": batch_size,
                "num_epochs": num_epochs,
                "learning_rate": learning_rate,
                "max_seq_length": max_seq_length,
                "validate_every_n_samples": validate_every_n_samples,
            },
        )

    # Training loop
    model.train()
    logger.info("Starting training loop")
    global_step = 0
    samples_since_last_val = 0

    last_val_loss = None
    last_val_ppl = None

    for epoch in range(num_epochs):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids)
            loss = criterion(outputs.view(-1, model.vocab_size), input_ids.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            global_step += 1
            samples_since_last_val += input_ids.size(0)

            if use_wandb:
                wandb.log({"train/loss": loss.item()}, step=global_step)

            # Periodic validation
            if samples_since_last_val >= validate_every_n_samples:
                last_val_loss, last_val_ppl = _run_validation(
                    model, val_loader, criterion, device
                )
                if use_wandb:
                    wandb.log(
                        {
                            "val/loss": last_val_loss,
                            "val/perplexity": last_val_ppl,
                        },
                        step=global_step,
                    )
                samples_since_last_val = 0

            postfix = {"loss": f"{loss.item():.4f}"}
            if last_val_loss is not None:
                postfix["val_loss"] = f"{last_val_loss:.4f}"
                postfix["val_ppl"] = f"{last_val_ppl:.2f}"
            progress_bar.set_postfix(postfix)

        epoch_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch + 1} completed - Average loss: {epoch_loss:.4f}")
        if use_wandb:
            wandb.log({"train/epoch_loss": epoch_loss}, step=global_step)

    logger.info("Training completed successfully")

    if use_wandb:
        wandb.finish()

    # Save model weights
    model_weights_path = "trained_model_weights.pth"
    torch.save(model.state_dict(), model_weights_path)
    logger.info(f"Model weights saved to: {model_weights_path}")

    return model

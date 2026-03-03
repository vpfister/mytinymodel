"""Training logic for the tiny GPT-2 model."""

import logging
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import GPT2Tokenizer
from tqdm import tqdm

from .model import TinyGPT2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set specific packages to WARNING level only
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("datasets").setLevel(logging.WARNING)


def train(
    dataset_name: str = "imdb",
    batch_size: int = 32,
    num_epochs: int = 3,
    learning_rate: float = 1e-3,
    max_seq_length: int = 128,
) -> TinyGPT2:
    """Train the tiny GPT-2 model on a Hugging Face dataset."""
    logger.info("Starting training process")
    logger.info(f"Configuration: dataset={dataset_name}, batch_size={batch_size}, epochs={num_epochs}, lr={learning_rate}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load dataset and tokenizer
    logger.info(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split="train")
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

    # Create data loader
    data_loader = DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=True)
    logger.info(f"Data loader created with {len(data_loader)} batches")

    # Initialize model
    logger.info("Initializing model")
    model = TinyGPT2(vocab_size=tokenizer.vocab_size).to(device)
    logger.info(f"Model initialized: {sum(p.numel() for p in model.parameters())} parameters")
    
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
            attention_mask = batch["attention_mask"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids)
            loss = criterion(
                outputs.view(-1, model.vocab_size), input_ids.view(-1)
            )
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})

        epoch_loss = total_loss / len(data_loader)
        logger.info(f"Epoch {epoch + 1} completed - Average loss: {epoch_loss:.4f}")
        print(f"Epoch {epoch + 1} average loss: {epoch_loss}")

    logger.info("Training completed successfully")
    
    # Save model weights
    model_weights_path = "trained_model_weights.pth"
    torch.save(model.state_dict(), model_weights_path)
    logger.info(f"Model weights saved to: {model_weights_path}")
    
    return model




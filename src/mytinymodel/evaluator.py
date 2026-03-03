"""Evaluation logic for the tiny GPT-2 model."""

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


def evaluate(
    model: TinyGPT2,
    dataset_name: str = "imdb",
    batch_size: int = 32,
    max_seq_length: int = 128,
) -> float:
    """Evaluate the tiny GPT-2 model on a Hugging Face dataset."""
    logger.info("Starting evaluation process")
    logger.info(f"Evaluating on dataset: {dataset_name}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    model.eval()
    model.to(device)

    # Load dataset and tokenizer
    logger.info("Loading evaluation dataset")
    dataset = load_dataset(dataset_name, split="test")
    logger.info(f"Dataset loaded: {len(dataset)} samples")
    
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Tokenize dataset
    logger.info("Tokenizing evaluation dataset")
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
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids)
            loss = criterion(
                outputs.view(-1, model.vocab_size), input_ids.view(-1)
            )
            total_loss += loss.item()

    average_loss = total_loss / len(data_loader)
    perplexity = torch.exp(torch.tensor(average_loss)).item()

    logger.info(f"Evaluation completed - Loss: {average_loss:.4f}, Perplexity: {perplexity:.2f}")
    print(f"Evaluation loss: {average_loss}")
    print(f"Evaluation perplexity: {perplexity}")

    return perplexity


if __name__ == "__main__":
    # Example usage when run as a script
    from .model import TinyGPT2
    import os
    
    # Try to load trained model weights if they exist
    model_weights_path = "trained_model_weights.pth"
    if os.path.exists(model_weights_path):
        print(f"Loading trained model weights from: {model_weights_path}")
        dummy_model = TinyGPT2(vocab_size=50257)
        dummy_model.load_state_dict(torch.load(model_weights_path))
    else:
        print("No trained model weights found, using dummy model")
        dummy_model = TinyGPT2(vocab_size=50257)
    
    print("Running evaluation...")
    evaluate(dummy_model)

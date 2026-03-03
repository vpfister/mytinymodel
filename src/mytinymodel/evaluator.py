"""Evaluation logic for the tiny GPT-2 model."""

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import GPT2Tokenizer
from tqdm import tqdm

from .model import TinyGPT2


def evaluate(
    model: TinyGPT2,
    dataset_name: str = "imdb",
    batch_size: int = 32,
    max_seq_length: int = 128,
) -> float:
    """Evaluate the tiny GPT-2 model on a Hugging Face dataset."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    # Load dataset and tokenizer
    dataset = load_dataset(dataset_name, split="test")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Tokenize dataset
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

    # Create data loader
    data_loader = DataLoader(tokenized_dataset, batch_size=batch_size)

    # Evaluation loop
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

    print(f"Evaluation loss: {average_loss}")
    print(f"Evaluation perplexity: {perplexity}")

    return perplexity

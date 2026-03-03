"""Training logic for the tiny GPT-2 model."""

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import GPT2Tokenizer
from tqdm import tqdm

from .model import TinyGPT2


def train(
    dataset_name: str = "imdb",
    batch_size: int = 32,
    num_epochs: int = 3,
    learning_rate: float = 1e-3,
    max_seq_length: int = 128,
) -> TinyGPT2:
    """Train the tiny GPT-2 model on a Hugging Face dataset."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset and tokenizer
    dataset = load_dataset(dataset_name, split="train")
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
    data_loader = DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=True)

    # Initialize model
    model = TinyGPT2(vocab_size=tokenizer.vocab_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    # Training loop
    model.train()
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

        print(f"Epoch {epoch + 1} average loss: {total_loss / len(data_loader)}")

    return model

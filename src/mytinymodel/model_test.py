"""Tests for the TinyGPT2 model."""

import torch
from .model import TinyGPT2


def test_tiny_gpt2_forward():
    """Test forward pass of TinyGPT2."""
    batch_size = 2
    seq_length = 10
    vocab_size = 50257

    model = TinyGPT2(vocab_size=vocab_size)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))

    output = model(input_ids)

    assert output.shape == (batch_size, seq_length, vocab_size)


def test_tiny_gpt2_initialization():
    """Test initialization of TinyGPT2."""
    vocab_size = 50257
    embedding_dim = 64
    hidden_dim = 128
    num_heads = 4
    num_layers = 3

    model = TinyGPT2(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=num_layers,
    )

    assert model.vocab_size == vocab_size
    assert model.embedding_dim == embedding_dim
    assert model.hidden_dim == hidden_dim
    assert model.num_heads == num_heads
    assert model.num_layers == num_layers

"""Tiny GPT-2 like model."""

import torch
import torch.nn as nn


class TinyGPT2(nn.Module):
    """A tiny GPT-2 like model."""

    def __init__(
        self,
        vocab_size: int = 50257,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.token_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.position_embeddings = nn.Embedding(1024, embedding_dim)

        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=embedding_dim,
                    nhead=num_heads,
                    dim_feedforward=hidden_dim,
                    batch_first=True,
                )
                for _ in range(num_layers)
            ]
        )

        self.ln_f = nn.LayerNorm(embedding_dim)
        self.lm_head = nn.Linear(embedding_dim, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        _, seq_len = input_ids.shape
        token_embeddings = self.token_embeddings(input_ids)
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        position_embeddings = self.position_embeddings(position_ids)
        hidden_states = token_embeddings + position_embeddings

        for layer in self.layers:
            hidden_states = layer(hidden_states)

        hidden_states = self.ln_f(hidden_states)
        logits = self.lm_head(hidden_states)

        return logits

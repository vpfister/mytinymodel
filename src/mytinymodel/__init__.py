"""A tiny GPT-2 like model for demonstration."""

from .evaluator import evaluate
from .model import TinyGPT2
from .trainer import train

__all__ = ["TinyGPT2", "evaluate", "train"]

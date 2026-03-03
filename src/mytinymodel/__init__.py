"""A tiny GPT-2 like model for demonstration."""

from .model import TinyGPT2
from .trainer import train
from .evaluator import evaluate

__all__ = ["TinyGPT2", "train", "evaluate"]

# MyTinyModel

A minimal GPT-2 model implementation for educational purposes.

## Installation

```bash
# Clone the repository
git clone git@github.com:vpfister/mytinymodel.git
cd mytinymodel

# Install dependencies using uv
uv sync
```

## Usage

The project now uses a unified CLI with subcommands for training and evaluation.

### Training

To train the model:

```bash
# Train with default parameters
uv run tiny train

# Train with custom parameters
uv run tiny train --dataset imdb --batch-size 64 --epochs 5 --learning-rate 0.001 --max-seq-length 256
```

Show training help:
```bash
uv run tiny train --help
```

### Evaluation

To evaluate a trained model:

```bash
# Evaluate with default parameters
uv run tiny eval

# Evaluate with custom parameters
uv run tiny eval --dataset imdb --batch-size 64 --max-seq-length 256
```

Show evaluation help:
```bash
uv run tiny eval --help
```

### Running Tests

```bash
# Run all tests
uv run pytest
```

## Project Structure

- `src/mytinymodel/model.py`: Core GPT-2 model implementation
- `src/mytinymodel/trainer.py`: Training logic and configuration
- `src/mytinymodel/evaluator.py`: Evaluation and inference functionality
- `src/mytinymodel/cli.py`: CLI entrypoint with train/eval subcommands
- `src/mytinymodel/*_test.py`: Unit tests for each component

## Configuration

The model architecture and training parameters are defined in:
- `model.py`: Model hyperparameters (embedding size, number of layers, etc.)
- `trainer.py`: Training configuration (batch size, learning rate, etc.)

## Requirements

- Python 3.13+
- PyTorch
- Other dependencies listed in `pyproject.toml`

## License

MIT
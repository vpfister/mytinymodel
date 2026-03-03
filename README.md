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

### Training

To train the model:

```bash
# Run the training script
uv run python -m mytinymodel.trainer
```

Training parameters can be configured in the `trainer.py` file or passed as arguments.

### Inference

To perform inference with a trained model:

```bash
# Run the evaluator
uv run python -m mytinymodel.evaluator
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
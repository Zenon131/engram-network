# Baby Dragon Hathcling (BDH) Model

This repository contains the implementation of the Baby Dragon Hathcling (BDH) model, a novel neural network architecture with PyTorch and MLX implementations.

## Project Structure

```
bdh/
├── configs/
│   ├── base.yaml
│   ├── tiny.yaml
│   ├── small.yaml
│   └── mlx_tiny.yaml
├── dragon/
│   ├── __init__.py
│   ├── model_torch.py       # BDH-GPU in PyTorch
│   ├── model_mlx.py         # (later) MLX version
│   ├── layers.py            # BDHLayer, LinearAttention, FF blocks
│   ├── tokenizer.py         # byte/char/subword tokenizer
│   ├── data.py              # dataset/dataloader abstractions
│   ├── train_loop.py        # one-epoch training step logic
│   ├── eval_loop.py         # evaluation logic
│   ├── generation.py        # sampling / greedy / beam search
│   └── utils.py             # logging, checkpoint helpers, seeding
├── scripts/
│   ├── train_torch.py       # entrypoint for PyTorch training
│   ├── eval_torch.py        # entrypoint for PyTorch eval
│   ├── generate_torch.py    # CLI text generation
│   └── prepare_data.py      # from raw txt → tokenized dataset
├── tests/
│   ├── test_layers.py       # unit tests for shapes, forward
│   ├── test_train_step.py   # can overfit a tiny batch
│   └── test_tokenizer.py
├── data/
│   ├── raw/                 # raw text dumps
│   └── processed/           # tokenized, serialized
├── checkpoints/
│   └── ...                  # saved models
├── pyproject.toml or setup.cfg
└── README.md
```

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd bdh
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Data Preparation

Prepare your data using the `prepare_data.py` script:

```
python scripts/prepare_data.py --input data/raw/sample.txt --output data/processed --tokenizer byte
```

### Training

Train the model using the `train_torch.py` script:

```
python scripts/train_torch.py
```

### Evaluation

Evaluate the trained model using the `eval_torch.py` script:

```
python scripts/eval_torch.py
```

### Text Generation

Generate text using the `generate_torch.py` script:

```
python scripts/generate_torch.py
```

## Configuration

The model can be configured using YAML files in the `configs/` directory. The available configurations are:

- `base.yaml`: Base configuration
- `tiny.yaml`: Tiny configuration for testing
- `small.yaml`: Small configuration
- `mlx_tiny.yaml`: Tiny configuration for MLX testing

## Testing

Run the tests using:

```
python -m unittest discover tests
```

## License

This project is licensed under the MIT License.
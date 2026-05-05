import torch
import torch.nn as nn
import yaml
import sys
import os

# Add the parent directory to Python path to find the dragon module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dragon.model_torch import BDH_GPU
from dragon.tokenizer import ByteTokenizer, SubwordTokenizer
from dragon.generation import generate_text, generate_text_greedy


def load_config(config_path: str) -> dict:
    """
    Loads configuration from a YAML file.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        dict: Configuration dictionary.
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    # Load configuration
    config = load_config('configs/tiny.yaml')  # Using tiny config for testing
    
    # Set device
    device = torch.device(
        'cuda' if torch.cuda.is_available()
        else 'mps' if torch.backends.mps.is_available()
        else 'cpu'
    )
    print(f"Using device: {device}")
    
    # Create model
    spatial_cfg = config.get('spatial_clustering', None)
    model = BDH_GPU(
        vocab_size=config['model']['vocab_size'],
        n_neurons=config['model']['n_neurons'],
        d_model=config['model']['d_model'],
        n_layers=config['model']['n_layers'],
        spatial_clustering=spatial_cfg,
    ).to(device)
    if spatial_cfg:
        print(f"Spatial clustering: {spatial_cfg['n_clusters']} clusters, "
              f"decay={spatial_cfg['decay_type']}, \u03c3={spatial_cfg['sigma']}")
    
    # Load trained model weights
    model.load_state_dict(torch.load('checkpoints/final_model.pth', map_location=device))
    print("Model loaded from checkpoints/final_model.pth")
    
    # Create tokenizer
    tokenizer = ByteTokenizer()  # Using byte tokenizer for this example
    
    # Generate text
    prompt = "Once upon a time"
    max_length = 100
    temperature = 0.8
    
    print(f"Generating text with prompt: '{prompt}'")
    generated_text = generate_text(model, tokenizer, prompt, max_length, device, temperature)
    print(f"Generated text:\n{generated_text}")
    
    print("\n" + "="*50 + "\n")
    
    print(f"Generating text with prompt: '{prompt}' (greedy decoding)")
    generated_text_greedy = generate_text_greedy(model, tokenizer, prompt, max_length, device)
    print(f"Generated text (greedy):\n{generated_text_greedy}")


if __name__ == "__main__":
    main()
import torch
import torch.nn as nn
import yaml
import os
import sys

# Add the parent directory to Python path to find the dragon module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dragon.model_torch import BDH_GPU
from dragon.data import TextDataset, TextDataLoader
from dragon.train_loop import train_model


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
    config = load_config('configs/base.yaml')  # Using tiny config for testing
    
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
        print(f"Spatial clustering enabled: {spatial_cfg['n_clusters']} clusters, "
              f"decay={spatial_cfg['decay_type']}, σ={spatial_cfg['sigma']}")
    
    # Load dataset
    dataset = TextDataset('data/raw/synthetic_data.txt', 'byte', config['data']['max_sequence_length'])
    data_loader = TextDataLoader(dataset, config['training']['batch_size'], shuffle=True)
    
    # Train model
    train_model(
        model=model,
        data_loader=data_loader,
        num_epochs=config['training']['num_epochs'],
        learning_rate=config['training']['learning_rate'],
        device=device
    )
    
    # Save final model
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), 'checkpoints/final_model.pth')
    print("Model saved to checkpoints/final_model.pth")


if __name__ == "__main__":
    main()
import torch
import torch.nn as nn
import yaml
import sys
import os
import argparse

# Add the parent directory to Python path to find the dragon module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dragon.model_torch import BDH_GPU
from dragon.data import TextDataset, TextDataLoader
from dragon.eval_loop import print_evaluation_results


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
    import argparse
    
    parser = argparse.ArgumentParser(description="BDH model evaluation")
    parser.add_argument("--config", type=str, default="configs/large.yaml",
                       help="Path to configuration file")
    parser.add_argument("--data", type=str, default=None,
                       help="Path to evaluation data file")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set device
    device = torch.device(
        'cuda' if torch.cuda.is_available()
        else 'mps' if torch.backends.mps.is_available()
        else 'cpu'
    )
    print(f"Using device: {device}")
    
    # Create model with configuration
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
    
    # Load trained model weights (try final_model.pth first, then best_model.pth)
    if os.path.exists('checkpoints/final_model.pth'):
        model.load_state_dict(torch.load('checkpoints/final_model.pth', map_location=device))
        print("Model loaded from checkpoints/final_model.pth")
    elif os.path.exists('checkpoints/best_model.pth'):
        model.load_state_dict(torch.load('checkpoints/best_model.pth', map_location=device))
        print("Model loaded from checkpoints/best_model.pth")
    else:
        print("Warning: No trained model found. Using randomly initialized model.")
    
    # Load dataset
    if args.data is None:
        # Use expanded dataset if available, otherwise use sample
        if os.path.exists('data/raw/expanded_dataset.txt'):
            data_file = 'data/raw/expanded_dataset.txt'
        else:
            data_file = 'data/raw/sample.txt'
    else:
        data_file = args.data
        
    dataset = TextDataset(data_file, 'byte', config['data']['max_sequence_length'])
    data_loader = TextDataLoader(dataset, config['training']['batch_size'], shuffle=False)
    
    print(f"Evaluating on: {data_file}")
    print(f"Dataset size: {len(dataset)} sequences")
    
    # Evaluate model
    print_evaluation_results(model, data_loader, device)


if __name__ == "__main__":
    main()
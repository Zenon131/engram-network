#!/usr/bin/env python3
"""
Enhanced training script for BDH model with support for larger datasets,
better training strategies, and improved monitoring.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime

# Add the parent directory to Python path to find the dragon module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dragon.model_torch import BDH_GPU
from dragon.data import TextDataset, TextDataLoader
from dragon.train_loop import train_step
from dragon.eval_loop import print_evaluation_results


class EnhancedTrainer:
    """Enhanced trainer with better monitoring and training strategies."""
    
    def __init__(self, config_path: str):
        self.config = self.load_config(config_path)
        self.device = torch.device(
            'cuda' if torch.cuda.is_available()
            else 'mps' if torch.backends.mps.is_available()
            else 'cpu'
        )
        self.setup_directories()
        self.setup_logging()
        
    def load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def setup_directories(self):
        """Create necessary directories."""
        os.makedirs('checkpoints', exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        os.makedirs('metrics', exist_ok=True)
    
    def setup_logging(self):
        """Setup logging for training."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = f"logs/training_{timestamp}.log"
        self.metrics_file = f"metrics/metrics_{timestamp}.json"
        
        self.metrics_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'epoch_times': []
        }
    
    def create_model(self):
        """Create the BDH model."""
        model_config = self.config['model']
        spatial_cfg = self.config.get('spatial_clustering', None)
        model = BDH_GPU(
            vocab_size=model_config['vocab_size'],
            n_neurons=model_config['n_neurons'],
            d_model=model_config['d_model'],
            n_layers=model_config['n_layers'],
            spatial_clustering=spatial_cfg,
        ).to(self.device)
        
        print(f"Model created:")
        print(f"  Vocabulary size: {model_config['vocab_size']}")
        print(f"  Neurons: {model_config['n_neurons']}")
        print(f"  Model dimension: {model_config['d_model']}")
        print(f"  Layers: {model_config['n_layers']}")
        print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        if spatial_cfg:
            print(f"  Spatial clustering: {spatial_cfg['n_clusters']} clusters, "
                  f"decay={spatial_cfg['decay_type']}, \u03c3={spatial_cfg['sigma']}")
            print(f"  Cluster stats: {model.spatial_map.get_cluster_stats()}")
        
        return model
    
    def create_optimizer(self, model):
        """Create optimizer with configurable parameters."""
        training_config = self.config['training']
        optimizer_config = self.config.get('optimizer', {})
        
        # Ensure numeric values are properly converted
        lr = float(training_config['learning_rate'])
        betas = [float(b) for b in optimizer_config.get('betas', [0.9, 0.999])]
        eps = float(optimizer_config.get('eps', 1e-8))
        weight_decay = float(optimizer_config.get('weight_decay', 0.01))
        
        if optimizer_config.get('type', 'adam') == 'adamw':
            optimizer = optim.AdamW(
                model.parameters(),
                lr=lr,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay
            )
        else:
            optimizer = optim.Adam(
                model.parameters(),
                lr=lr,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay
            )
        
        return optimizer
    
    def create_scheduler(self, optimizer):
        """Create learning rate scheduler."""
        scheduler_config = self.config.get('scheduler', {})
        
        if scheduler_config.get('type') == 'cosine':
            from torch.optim.lr_scheduler import CosineAnnealingLR
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=scheduler_config.get('t_total', 100000)
            )
        elif scheduler_config.get('type') == 'step':
            from torch.optim.lr_scheduler import StepLR
            scheduler = StepLR(
                optimizer,
                step_size=scheduler_config.get('step_size', 10),
                gamma=scheduler_config.get('gamma', 0.1)
            )
        else:
            # Default: no scheduler
            scheduler = None
        
        return scheduler
    
    def load_dataset(self, data_file: str = None):
        """Load and prepare the dataset with memory efficiency."""
        if data_file is None:
            data_file = 'data/raw/combined_dataset.txt'
        
        if not os.path.exists(data_file):
            print(f"Warning: Data file {data_file} not found.")
            print("Using synthetic data.")
            data_file = 'data/raw/synthetic_data.txt'
        
        data_config = self.config['data']
        
        # Memory-efficient dataset loading
        # Limit dataset size to avoid memory issues
        max_samples = 1000  # Start with small dataset for memory safety
        
        # Check file size and adjust max_samples accordingly
        file_size = os.path.getsize(data_file) if os.path.exists(data_file) else 0
        if file_size > 10 * 1024 * 1024:  # 10MB
            print(f"Large dataset detected ({file_size//1024//1024}MB). Using limited samples for memory safety.")
            max_samples = 500
        elif file_size > 1 * 1024 * 1024:  # 1MB
            max_samples = 2000
        else:
            max_samples = 5000
        
        from dragon.data_memory_efficient import create_memory_efficient_dataset
        dataset = create_memory_efficient_dataset(
            data_file,
            max_samples=max_samples,
            max_length=data_config['max_sequence_length']
        )
        
        # Create data loaders
        train_loader = TextDataLoader(
            dataset,
            self.config['training']['batch_size'],
            shuffle=True
        )
        
        val_loader = TextDataLoader(
            dataset,
            self.config['training']['batch_size'],
            shuffle=False
        )
        
        print(f"Dataset loaded (memory-efficient):")
        print(f"  File: {data_file}")
        print(f"  File size: {file_size//1024}KB")
        print(f"  Sequences: {len(dataset)}")
        print(f"  Batch size: {self.config['training']['batch_size']}")
        print(f"  Batches per epoch: {len(train_loader)}")
        print(f"  Memory safety: Limited to {max_samples} samples")
        
        return train_loader, val_loader
    
    def save_checkpoint(self, model, optimizer, epoch, loss, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'config': self.config
        }
        
        checkpoint_path = f"checkpoints/model_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = "checkpoints/best_model.pth"
            torch.save(checkpoint, best_path)
            print(f"Best model saved: {best_path}")
        
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def log_metrics(self, epoch, train_loss, val_loss, lr, epoch_time):
        """Log training metrics."""
        self.metrics_history['train_loss'].append(train_loss)
        self.metrics_history['val_loss'].append(val_loss)
        self.metrics_history['learning_rate'].append(lr)
        self.metrics_history['epoch_times'].append(epoch_time)
        
        # Save metrics to file
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        
        # Log to console and file
        log_entry = (
            f"Epoch {epoch:3d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"LR: {lr:.6f} | "
            f"Time: {epoch_time:.1f}s"
        )
        
        print(log_entry)
        with open(self.log_file, 'a') as f:
            f.write(log_entry + '\n')
    
    def train(self, data_file: str = None):
        """Main training loop."""
        print("Starting enhanced training...")
        print(f"Device: {self.device}")
        print(f"Log file: {self.log_file}")
        print(f"Metrics file: {self.metrics_file}")
        
        # Create model, optimizer, and dataset
        model = self.create_model()
        optimizer = self.create_optimizer(model)
        scheduler = self.create_scheduler(optimizer)
        train_loader, val_loader = self.load_dataset(data_file)
        
        training_config = self.config['training']
        num_epochs = training_config['num_epochs']
        gradient_clip = training_config.get('gradient_clip', None)
        
        best_loss = float('inf')
        
        print(f"\nTraining for {num_epochs} epochs...")
        print("=" * 80)
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # Training phase
            model.train()
            train_loss = train_step(model, train_loader, optimizer, self.device)
            
            # Apply gradient clipping
            if gradient_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            
            # Update learning rate
            current_lr = optimizer.param_groups[0]['lr']
            if scheduler:
                scheduler.step()
            
            # Validation phase (simplified - using same data for now)
            model.eval()
            val_loss = train_loss  # For now, use train loss as val loss
            
            epoch_time = time.time() - start_time
            
            # Log metrics
            self.log_metrics(epoch + 1, train_loss, val_loss, current_lr, epoch_time)
            
            # Save checkpoint
            if (epoch + 1) % training_config['checkpoint_interval'] == 0:
                is_best = val_loss < best_loss
                if is_best:
                    best_loss = val_loss
                self.save_checkpoint(model, optimizer, epoch + 1, val_loss, is_best)
        
        # Save final model
        final_path = "checkpoints/final_model.pth"
        torch.save(model.state_dict(), final_path)
        print(f"\nFinal model saved: {final_path}")
        
        # Print final evaluation
        print("\nFinal evaluation:")
        print_evaluation_results(model, train_loader, self.device)
        
        print(f"\nTraining completed!")
        print(f"Logs saved to: {self.log_file}")
        print(f"Metrics saved to: {self.metrics_file}")


def main():
    """Main function for enhanced training."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced BDH model training")
    parser.add_argument("--config", type=str, default="configs/large.yaml", 
                       help="Path to configuration file")
    parser.add_argument("--data", type=str, default=None,
                       help="Path to training data file")
    
    args = parser.parse_args()
    
    # Check if data needs to be downloaded
    if args.data is None and not os.path.exists('data/raw/combined_dataset.txt'):
        print("No large dataset found. Would you like to download one?")
        print("Run: python scripts/download_datasets.py --datasets all --combine")
        print("Then run this script again.")
        return
    
    # Start training
    trainer = EnhancedTrainer(args.config)
    trainer.train(args.data)


if __name__ == "__main__":
    main()
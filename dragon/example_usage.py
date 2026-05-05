import torch
from .model_torch_enhanced import EnhancedBDH_GPU
from .sr_utils import SuccessorRepresentation
from .grid_utils import NeuronGrid


def example_2d_neuron_grid():
    """Example of creating a 2D neuron grid with place coordinates and grid features."""
    # Create a neuron grid of size 16x16 (256 neurons)
    grid = NeuronGrid(grid_size=(16, 16), n_neurons=256)
    
    print(f"Grid shape: {grid.grid_h} x {grid.grid_w}")
    print(f"Place coordinates shape: {grid.place_coords.shape}")
    print(f"Grid features shape: {grid.grid_features.shape}")
    
    # Example place coordinates for first 5 neurons
    print("First 5 place coordinates:")
    print(grid.place_coords[:5])
    
    # Example grid features for first 5 neurons
    print("First 5 grid features:")
    print(grid.grid_features[:5])


def example_receptive_field_conv():
    """Example of applying receptive field convolution to neuron activations."""
    from .grid_utils import ReceptiveFieldConv2D
    
    # Create a receptive field conv layer for a 16x16 grid
    conv_layer = ReceptiveFieldConv2D(n_neurons=256, grid_size=(16, 16), kernel_size=3)
    
    # Example neuron activations (batch size 2)
    x = torch.randn(2, 256)
    
    # Apply receptive field processing
    x_processed = conv_layer(x)
    
    print(f"Original shape: {x.shape}")
    print(f"Processed shape: {x_processed.shape}")
    print(f"Sample processed values: {x_processed[0, :5]}")


def example_grid_cell_features():
    """Example of generating grid cell sinusoidal features."""
    # This is already implemented in NeuronGrid._create_grid_features()
    # Here's how you might use it:
    
    grid = NeuronGrid(grid_size=(8, 8), n_neurons=64)
    
    # Get grid features for all neurons
    features = grid.grid_features
    
    print(f"Grid features shape: {features.shape}")
    print("Sample grid features for neuron 0:")
    print(features[0])


def example_sr_td_update():
    """Example of Successor Representation TD update."""
    # Create SR module
    sr = SuccessorRepresentation(n_neurons=256, feature_dim=128, gamma=0.95, learning_rate=0.01)
    
    # Example state features
    current_features = torch.randn(4, 128)  # Batch of 4
    next_features = torch.randn(4, 128)
    
    # Compute SR for current state
    current_sr = sr.compute_sr(current_features)
    print(f"Current SR shape: {current_sr.shape}")
    
    # Update SR matrix
    sr.update_sr(current_features, next_features)
    print("SR matrix updated")


def example_combined_bdh_sr():
    """Example of combined BDH + SR step."""
    # Create enhanced BDH model
    model = EnhancedBDH_GPU(vocab_size=1000, n_neurons=1024, d_model=128, n_layers=4, grid_size=(32, 32))
    
    # Example input tokens
    tokens = torch.randint(0, 1000, (2, 10))  # Batch of 2, sequence length 10
    
    # Forward pass
    logits, state = model(tokens)
    
    print(f"Logits shape: {logits.shape}")
    print(f"State keys: {state.keys()}")
    if state['state_features'] is not None:
        print(f"State features shape: {state['state_features'].shape}")
    if state['sr_features'] is not None:
        print(f"SR features shape: {state['sr_features'].shape}")


def example_readout_with_sr():
    """Example of readout with SR concatenation."""
    # This is demonstrated in the EnhancedBDH_GPU model
    # The key part is:
    # 
    # # Concatenate SR features with the final state representation
    # combined_features = torch.cat([v_final, sr_features], dim=-1)
    # logits = self.readout(combined_features)
    #
    # This allows the model to use both the current state representation
    # and the learned successor representation for prediction
    
    print("Readout with SR concatenation:")
    print("1. Final state representation: v_final (B, d)")
    print("2. SR features: sr_features (B, d)")
    print("3. Combined: torch.cat([v_final, sr_features], dim=-1) (B, 2*d)")
    print("4. Logits: self.readout(combined_features) (B, vocab_size)")


if __name__ == "__main__":
    print("=== 2D Neuron Grid Example ===")
    example_2d_neuron_grid()
    
    print("\n=== Receptive Field Conv Example ===")
    example_receptive_field_conv()
    
    print("\n=== Grid Cell Features Example ===")
    example_grid_cell_features()
    
    print("\n=== SR TD Update Example ===")
    example_sr_td_update()
    
    print("\n=== Combined BDH + SR Example ===")
    example_combined_bdh_sr()
    
    print("\n=== Readout with SR Example ===")
    example_readout_with_sr()
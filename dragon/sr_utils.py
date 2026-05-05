import torch
import torch.nn as nn
import math
from typing import Tuple, Optional
from .grid_utils import NeuronGrid


class SuccessorRepresentation:
    """
    Successor Representation (SR) module for learning and representing future state expectations.
    """
    
    def __init__(self, n_neurons: int, feature_dim: int, gamma: float = 0.95, learning_rate: float = 0.01):
        """
        Initializes the Successor Representation module.
        
        Args:
            n_neurons (int): Number of neurons in the grid.
            feature_dim (int): Dimension of the state features.
            gamma (float): Discount factor for temporal relationships.
            learning_rate (float): Learning rate for SR updates.
        """
        self.n_neurons = n_neurons
        self.feature_dim = feature_dim
        self.gamma = gamma
        self.learning_rate = learning_rate
        
        # Initialize SR matrix W: (feature_dim, feature_dim)
        self.W = nn.Parameter(torch.randn(feature_dim, feature_dim) * 0.01)
        
    def compute_sr(self, state_features: torch.Tensor) -> torch.Tensor:
        """
        Computes the successor representation for a given state.
        
        Args:
            state_features (torch.Tensor): State features of shape (B, feature_dim).
            
        Returns:
            torch.Tensor: Successor representation of shape (B, feature_dim).
        """
        # ψ_t = Wϕ(s_t)
        return torch.matmul(state_features, self.W.T)
        
    def update_sr(self, current_features: torch.Tensor, next_features: torch.Tensor):
        """
        Updates the SR matrix using the temporal difference rule.
        
        Args:
            current_features (torch.Tensor): Current state features (B, feature_dim).
            next_features (torch.Tensor): Next state features (B, feature_dim).
        """
        # ψ_t = Wϕ(s_t)
        current_sr = self.compute_sr(current_features)
        
        # ψ_{t+1} = Wϕ(s_{t+1})
        next_sr = self.compute_sr(next_features)
        
        # TD error: δ = ϕ(s_t) + γψ_{t+1} - ψ_t
        td_error = current_features + self.gamma * next_sr - current_sr
        
        # W ← W + α * δ * ϕ(s_t)^T
        with torch.no_grad():
            # Update using outer product
            self.W += self.learning_rate * torch.einsum("bd,bD->dD", td_error, current_features)


class EnhancedBDHLayer(nn.Module):
    """
    Enhanced BDH layer with receptive fields, grid features, and SR integration.
    """
    
    def __init__(self, n_neurons, d_model, grid_size: Tuple[int, int], kernel_size: int = 3):
        """
        Initializes the enhanced BDH layer.
        
        Args:
            n_neurons (int): Number of neurons.
            d_model (int): Model dimension.
            grid_size (Tuple[int, int]): Size of the 2D neuron grid (height, width).
            kernel_size (int): Size of the convolutional kernel for receptive fields.
        """
        super().__init__()
        self.n = n_neurons
        self.d = d_model
        self.grid_size = grid_size
        
        # Initialize neuron grid with place coordinates and grid features
        self.neuron_grid = NeuronGrid(grid_size, n_neurons)
        
        # Receptive field convolution
        self.receptive_field = nn.Conv2d(1, 1, kernel_size=kernel_size, 
                                        padding=kernel_size//2, bias=False)
        self._init_receptive_weights()
        
        # Grid cell features (sinusoidal)
        self.grid_features = self.neuron_grid.grid_features
        
        # Standard BDH parameters
        self.E = nn.Parameter(torch.randn(self.d, self.n) * 0.02)
        self.Dx = nn.Parameter(torch.randn(self.n, self.d) * 0.02)
        self.Dy = nn.Parameter(torch.randn(self.n, self.d) * 0.02)
        
        # Layer norms
        self.ln_y = nn.LayerNorm(self.d, elementwise_affine=False)
        self.ln_rho = nn.LayerNorm(self.d, elementwise_affine=False)
        
        # SR module
        self.sr_module = SuccessorRepresentation(n_neurons, d_model)
        
    def _init_receptive_weights(self):
        """Initializes receptive field weights to emphasize local connections."""
        with torch.no_grad():
            if self.receptive_field.kernel_size[0] == 3:
                self.receptive_field.weight.copy_(torch.tensor([[[[0.0, 0.1, 0.0],
                                                                 [0.1, 0.6, 0.1],
                                                                 [0.0, 0.1, 0.0]]]], 
                                                               dtype=torch.float32))
            else:
                # For other kernel sizes, initialize with a Gaussian-like pattern
                kernel_size = self.receptive_field.kernel_size[0]
                center = kernel_size // 2
                for i in range(kernel_size):
                    for j in range(kernel_size):
                        dist = math.sqrt((i - center)**2 + (j - center)**2)
                        self.receptive_field.weight[0, 0, i, j] = math.exp(-dist**2 / (2 * (center/2)**2))
                        
    def apply_receptive_field(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies receptive field convolution to neuron activations.
        
        Args:
            x (torch.Tensor): Neuron activations of shape (B, n).
            
        Returns:
            torch.Tensor: Processed activations of shape (B, n).
        """
        B = x.shape[0]
        # Reshape to 2D grid
        x_grid = x.view(B, 1, self.grid_size[0], self.grid_size[1])
        # Apply convolution
        x_processed = self.receptive_field(x_grid)
        # Reshape back to 1D
        return x_processed.view(B, self.n)
        
    def extract_state_features(self, y: torch.Tensor) -> torch.Tensor:
        """
        Extracts state features from neuron activations.
        
        Args:
            y (torch.Tensor): Neuron activations of shape (B, n).
            
        Returns:
            torch.Tensor: State features of shape (B, d).
        """
        # Use grid features as a basis for state representation
        # Weighted sum of grid features based on neuron activations
        grid_features = self.grid_features.to(y.device)  # (n, feature_dim)
        # (B, n) @ (n, feature_dim) -> (B, feature_dim)
        state_features = torch.matmul(y, grid_features)
        return state_features
        
    def forward(self, x, y, rho, prev_state_features=None):
        """
        Forward pass with SR integration.
        
        Args:
            x, y: (B, n)
            rho: (B, n, d)
            prev_state_features: (B, d) or None
            
        Returns:
            x, y, rho, state_features, sr_features
        """
        # Apply receptive field processing to x
        x = self.apply_receptive_field(x)
        
        # 1) ReLU-lowrank feedforward on x via previous y
        # v = LN(E y)
        v = torch.matmul(y, self.E.T)               # (B, d)
        v = self.ln_y(v)
        
        # x_t = x + ReLU(Dx v)
        x = x + F.relu(torch.matmul(v, self.Dx.T))  # (B, n)
        
        # 2) Hebbian & attention via rho
        # outer = v x^T
        outer = torch.einsum("bd,bn->bnd", v, x)
        rho = rho + outer                           # (B, n, d)
        
        # q = LN(rho x)
        q = torch.einsum("bnd,bn->bd", rho, x)      # (B, d)
        q = self.ln_rho(q)
        
        # y_t = ReLU(Dy q) ⊙ x
        y_tilde = torch.matmul(q, self.Dy.T)        # (B, n)
        y = F.relu(y_tilde) * x
        
        # Extract state features from current y
        state_features = self.extract_state_features(y)
        
        # Compute SR features if previous state features are available
        sr_features = None
        if prev_state_features is not None:
            # Update SR matrix
            self.sr_module.update_sr(prev_state_features, state_features)
            # Compute current SR features
            sr_features = self.sr_module.compute_sr(state_features)
            
        return x, y, rho, state_features, sr_features
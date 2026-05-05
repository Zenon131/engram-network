
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional


class NeuronGrid:
    """
    A 2D grid layout for neurons with place cell coordinates and grid cell features.
    """

    def __init__(self, grid_size: Tuple[int, int], n_neurons: int):
        """
        Initializes the neuron grid.

        Args:
            grid_size (Tuple[int, int]): Size of the 2D grid (height, width).
            n_neurons (int): Total number of neurons.
        """
        self.grid_h, self.grid_w = grid_size
        self.n_neurons = n_neurons
        
        # Ensure n_neurons matches grid size
        assert self.grid_h * self.grid_w == n_neurons, "Grid size must match number of neurons"
        
        # Create place cell coordinates
        self.place_coords = self._create_place_coords()
        
        # Create grid cell features (sinusoidal)
        self.grid_features = self._create_grid_features()

    def _create_place_coords(self) -> torch.Tensor:
        """
        Creates 2D place cell coordinates for each neuron.

        Returns:
            torch.Tensor: Place coordinates of shape (n_neurons, 2).
        """
        coords = []
        for i in range(self.grid_h):
            for j in range(self.grid_w):
                coords.append([i, j])
        return torch.tensor(coords, dtype=torch.float32)

    def _create_grid_features(self, n_scales: int = 3, n_orientations: int = 4) -> torch.Tensor:
        """
        Creates grid cell sinusoidal features for each neuron.

        Args:
            n_scales (int): Number of spatial scales.
            n_orientations (int): Number of orientations.

        Returns:
            torch.Tensor: Grid features of shape (n_neurons, n_scales * n_orientations * 2).
        """
        features = []
        for i in range(self.grid_h):
            for j in range(self.grid_w):
                feat = []
                for scale in range(1, n_scales + 1):
                    for orient in range(n_orientations):
                        # Convert orientation to radians
                        theta = orient * math.pi / n_orientations
                        
                        # Calculate sinusoidal features
                        x_proj = i * math.cos(theta) + j * math.sin(theta)
                        y_proj = -i * math.sin(theta) + j * math.cos(theta)
                        
                        feat.append(math.sin(2 * math.pi * x_proj / (scale * self.grid_w)))
                        feat.append(math.cos(2 * math.pi * y_proj / (scale * self.grid_h)))
                features.append(feat)
        return torch.tensor(features, dtype=torch.float32)


class ReceptiveFieldConv2D(nn.Module):
    """
    2D convolutional layer implementing local receptive fields for neuron grid.
    """

    def __init__(self, n_neurons: int, grid_size: Tuple[int, int], kernel_size: int = 3):
        """
        Initializes the receptive field convolution.

        Args:
            n_neurons (int): Number of neurons.
            grid_size (Tuple[int, int]): Size of the 2D grid (height, width).
            kernel_size (int): Size of the convolutional kernel.
        """
        super().__init__()
        self.n_neurons = n_neurons
        self.grid_h, self.grid_w = grid_size
        self.kernel_size = kernel_size
        
        # Ensure n_neurons matches grid size
        assert self.grid_h * self.grid_w == n_neurons, "Grid size must match number of neurons"
        
        # Conv layer for 2D grid
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        
        # Initialize weights to implement local connectivity
        self._init_weights()

    def _init_weights(self):
        """
        Initializes convolutional weights to implement local connectivity.
        """
        with torch.no_grad():
            # Initialize with a simple pattern that emphasizes local connections
            if self.kernel_size == 3:
                self.conv.weight.copy_(torch.tensor([[[[0.0, 0.1, 0.0],
                                                     [0.1, 0.6, 0.1],
                                                     [0.0, 0.1, 0.0]]]], dtype=torch.float32))
            else:
                # For other kernel sizes, initialize with a Gaussian-like pattern
                center = self.kernel_size // 2
                for i in range(self.kernel_size):
                    for j in range(self.kernel_size):
                        dist = math.sqrt((i - center)**2 + (j - center)**2)
                        self.conv.weight[0, 0, i, j] = math.exp(-dist**2 / (2 * (center/2)**2))


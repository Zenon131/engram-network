"""
Distance-based neuron clustering for biologically-inspired weight modulation.

Assigns neurons to spatial clusters and applies distance-based decay to
neuron-neuron interactions. Nearby neurons form stronger connections while
distant neurons are weakly coupled — mimicking cortical column structure
and physical wiring constraints in biological brains.

Efficiency: operates on a C×C cluster distance matrix (C << n) rather than
materializing the full n×n pairwise distance matrix.
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class NeuronSpatialMap(nn.Module):
    """
    Maps neurons to spatial clusters and provides distance-based
    interaction scaling.

    Each neuron belongs to one of C clusters. Each cluster has a position
    in a low-dimensional spatial coordinate system. The distance between
    cluster centroids determines how strongly neurons in those clusters
    can interact: close clusters → strong coupling, far clusters → weak
    coupling.

    The cluster positions and the decay length-scale (sigma) can be
    learned end-to-end via backpropagation, allowing the network to
    discover its own spatial organisation during training.
    """

    def __init__(
        self,
        n_neurons: int,
        n_clusters: int = 64,
        spatial_dim: int = 3,
        sigma: float = 1.0,
        decay_type: str = "gaussian",
        learnable_positions: bool = True,
        learnable_sigma: bool = True,
        min_decay: float = 0.01,
    ):
        """
        Args:
            n_neurons: Total number of neurons in the layer.
            n_clusters: Number of spatial clusters (C). Neurons are evenly
                distributed across clusters in contiguous blocks.
            spatial_dim: Dimensionality of the spatial coordinate space
                (2 = cortical sheet, 3 = volumetric brain).
            sigma: Initial length-scale for the decay function. Larger σ
                means more permissive long-range connections.
            decay_type: Shape of the distance decay curve. One of:
                'gaussian'    — exp(-d²/2σ²)    smooth, local
                'exponential' — exp(-d/σ)       heavier tail
                'inverse'     — 1/(1 + d/σ)     long-range friendly
                'cosine'      — cos(πd/2σ)      hard cutoff at d=σ
            learnable_positions: If True, cluster positions are nn.Parameters
                and will be optimised during training.
            learnable_sigma: If True, sigma is a learnable parameter.
            min_decay: Floor value for the decay (prevents total
                disconnection between any pair of clusters).
        """
        super().__init__()
        self.n_neurons = n_neurons
        self.n_clusters = n_clusters
        self.spatial_dim = spatial_dim
        self.decay_type = decay_type
        self.min_decay = min_decay

        # --- Cluster centroid positions ---
        init_pos = self._init_positions()
        if learnable_positions:
            self.positions = nn.Parameter(init_pos)
        else:
            self.register_buffer("positions", init_pos)

        # --- Decay length-scale ---
        if learnable_sigma:
            self.log_sigma = nn.Parameter(torch.log(torch.tensor(float(sigma))))
        else:
            self.register_buffer("log_sigma", torch.log(torch.tensor(float(sigma))))

        # --- Neuron → cluster assignment (fixed, contiguous blocks) ---
        assignments = torch.div(
            torch.arange(n_neurons) * n_clusters,
            n_neurons,
            rounding_mode="floor",
        ).long()
        self.register_buffer("assignments", assignments)

        # Cluster sizes (for normalisation)
        cluster_sizes = torch.zeros(n_clusters)
        for c in range(n_clusters):
            cluster_sizes[c] = (assignments == c).float().sum()
        self.register_buffer("cluster_sizes", cluster_sizes)

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_positions(self) -> torch.Tensor:
        """Place clusters on a regular grid in the chosen spatial_dim."""
        n = self.n_clusters
        if self.spatial_dim == 2:
            side = int(math.ceil(math.sqrt(n)))
            coords = []
            for i in range(side):
                for j in range(side):
                    if len(coords) < n:
                        coords.append([i / max(side - 1, 1), j / max(side - 1, 1)])
            return torch.tensor(coords, dtype=torch.float32)
        elif self.spatial_dim == 3:
            side = int(math.ceil(n ** (1.0 / 3.0)))
            coords = []
            for i in range(side):
                for j in range(side):
                    for k in range(side):
                        if len(coords) < n:
                            coords.append([
                                i / max(side - 1, 1),
                                j / max(side - 1, 1),
                                k / max(side - 1, 1),
                            ])
            return torch.tensor(coords, dtype=torch.float32)
        else:
            # Arbitrary dimensionality — random init
            return torch.randn(n, self.spatial_dim) * 0.5

    # ------------------------------------------------------------------
    # Core distance / decay computation
    # ------------------------------------------------------------------

    @property
    def effective_sigma(self) -> torch.Tensor:
        """Current sigma value (always positive via exp)."""
        return torch.exp(self.log_sigma)

    def compute_pairwise_distances(self) -> torch.Tensor:
        """Euclidean distances between all cluster pairs. Shape: (C, C)."""
        diff = self.positions.unsqueeze(1) - self.positions.unsqueeze(0)
        return torch.norm(diff, dim=-1)

    def compute_decay_matrix(self) -> torch.Tensor:
        """
        Compute the C×C decay matrix.

        Returns:
            Tensor of shape (C, C) with values in [min_decay, 1.0].
            decay[i, i] ≈ 1 (same cluster = full strength).
        """
        distances = self.compute_pairwise_distances()
        sigma = self.effective_sigma

        if self.decay_type == "gaussian":
            decay = torch.exp(-(distances ** 2) / (2 * sigma ** 2))
        elif self.decay_type == "exponential":
            decay = torch.exp(-distances / sigma)
        elif self.decay_type == "inverse":
            decay = 1.0 / (1.0 + distances / sigma)
        elif self.decay_type == "cosine":
            decay = torch.cos(
                torch.clamp(distances / sigma * math.pi / 2, 0, math.pi / 2)
            )
        else:
            raise ValueError(f"Unknown decay_type: {self.decay_type}")

        return torch.clamp(decay, min=self.min_decay)

    # ------------------------------------------------------------------
    # Interaction modulation
    # ------------------------------------------------------------------

    def get_interaction_scale(self, source: torch.Tensor) -> torch.Tensor:
        """
        Compute per-neuron scaling factors for target neurons based on
        which source neurons are currently active.

        The idea: if source activity is concentrated in cluster A, then
        target neurons in cluster A get scale ≈ 1 while target neurons
        in distant cluster Z get scale ≈ min_decay.

        Scales are normalised so their mean ≈ 1 (preserves signal
        magnitude while redistributing it spatially).

        Args:
            source: (B, n) source neuron activations.

        Returns:
            (B, n) per-neuron multiplicative scales.
        """
        B = source.shape[0]
        decay = self.compute_decay_matrix()  # (C, C)

        # Aggregate |source| per cluster → mean activation per cluster
        source_abs = source.abs()
        cluster_source = torch.zeros(
            B, self.n_clusters, device=source.device, dtype=source.dtype
        )
        cluster_source.scatter_add_(
            1,
            self.assignments.unsqueeze(0).expand(B, -1),
            source_abs,
        )
        cluster_source = cluster_source / self.cluster_sizes.unsqueeze(0).clamp(min=1)

        # Decay-weighted influence on each target cluster
        # target_scale[b, c_target] = Σ_{c_src} cluster_source[b, c_src] * decay[c_src, c_target]
        target_scale = torch.matmul(cluster_source, decay)  # (B, C)

        # Normalise so mean scale ≈ 1
        target_scale = target_scale / (target_scale.mean(dim=1, keepdim=True) + 1e-8)

        # Expand to per-neuron
        neuron_scale = target_scale.gather(
            1, self.assignments.unsqueeze(0).expand(B, -1)
        )

        return neuron_scale

    def modulate(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Apply distance-based modulation: scale *target* activations by
        their spatial proximity to the active *source* neurons.

        Args:
            source: (B, n) — which neurons produced the signal.
            target: (B, n) — the signal to modulate.

        Returns:
            (B, n) modulated target (same shape).
        """
        scale = self.get_interaction_scale(source)
        return target * scale

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_cluster_stats(self) -> dict:
        """Return summary statistics for logging / debugging."""
        with torch.no_grad():
            decay = self.compute_decay_matrix()
            distances = self.compute_pairwise_distances()
            mask = distances > 0  # exclude self-distance
            return {
                "sigma": self.effective_sigma.item(),
                "mean_distance": distances.mean().item(),
                "min_nonzero_distance": distances[mask].min().item() if mask.any() else 0.0,
                "max_distance": distances.max().item(),
                "mean_decay": decay.mean().item(),
                "min_decay": decay.min().item(),
                "max_decay": decay.max().item(),
                "n_clusters": self.n_clusters,
                "n_neurons": self.n_neurons,
                "spatial_dim": self.spatial_dim,
                "decay_type": self.decay_type,
            }

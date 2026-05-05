import torch
from torch import nn
import torch.nn.functional as F
from .layers import BDHLayer, SpatialBDHLayer
from .spatial import NeuronSpatialMap


class BDH_GPU(nn.Module):
    def __init__(self, vocab_size=256, n_neurons=32768, d_model=256, n_layers=6,
                 spatial_clustering=None):
        """
        Args:
            vocab_size: Size of the token vocabulary.
            n_neurons: Number of neurons per layer.
            d_model: Dimensionality of the low-rank projection space.
            n_layers: Number of BDH layers.
            spatial_clustering: Optional dict to enable distance-based
                neuron clustering.  Keys (all optional, shown with defaults):
                  n_clusters:          64      — number of spatial clusters
                  spatial_dim:         3       — coordinate dimensions
                  sigma:               1.0     — decay length-scale
                  decay_type:          'gaussian' — decay curve shape
                  learnable_positions: True    — optimise cluster positions
                  learnable_sigma:     True    — optimise sigma
                  min_decay:           0.01    — floor for the decay
                  modulate_feedforward: True   — apply decay on y→x path
                  modulate_hebbian:    True    — apply decay on Hebbian update
                  modulate_output:     True    — apply decay on rho→y path
        """
        super().__init__()
        self.n = n_neurons
        self.d = d_model
        self.L = n_layers

        self.emb = nn.Embedding(vocab_size, d_model)

        # --- Spatial clustering (optional) ---
        if spatial_clustering is not None:
            sc = spatial_clustering
            self.spatial_map = NeuronSpatialMap(
                n_neurons=n_neurons,
                n_clusters=sc.get("n_clusters", 64),
                spatial_dim=sc.get("spatial_dim", 3),
                sigma=sc.get("sigma", 1.0),
                decay_type=sc.get("decay_type", "gaussian"),
                learnable_positions=sc.get("learnable_positions", True),
                learnable_sigma=sc.get("learnable_sigma", True),
                min_decay=sc.get("min_decay", 0.01),
            )
            self.layers = nn.ModuleList([
                SpatialBDHLayer(
                    n_neurons, d_model,
                    spatial_map=self.spatial_map,
                    modulate_feedforward=sc.get("modulate_feedforward", True),
                    modulate_hebbian=sc.get("modulate_hebbian", True),
                    modulate_output=sc.get("modulate_output", True),
                )
                for _ in range(n_layers)
            ])
        else:
            self.spatial_map = None
            self.layers = nn.ModuleList(
                [BDHLayer(n_neurons, d_model) for _ in range(n_layers)]
            )

        self.readout = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, tokens, state=None):
        """
        tokens: (B, T)
        state: optional dict with 'x', 'y', 'rho' for B,T or recurrent streaming
        """
        B, T = tokens.shape
        h = self.emb(tokens)  # (B, T, d)

        # For simplicity, we recompute from scratch per T
        # Map token reps into neuron space: project h to n and sum over time
        # (toy: you'll want something more principled)
        x = torch.zeros(B, self.n, device=h.device)
        y = torch.zeros_like(x)
        rho = torch.zeros(B, self.n, self.d, device=h.device)

        # Store logits for each time step
        logits_sequence = []

        # Process sequence step by step
        for t in range(T):
            # seed y from token embedding at step t
            y_seed = torch.matmul(h[:, t, :], torch.randn(self.d, self.n, device=h.device))  # toy
            y = F.relu(y_seed)

            for layer in self.layers:
                x, y, rho = layer(x, y, rho)

            # Compute logits for this time step
            v_final = torch.matmul(y, self.layers[-1].E.T)  # (B, d)
            logits = self.readout(v_final)                  # (B, vocab)
            logits_sequence.append(logits)

        # Stack logits for all time steps
        logits = torch.stack(logits_sequence, dim=1)  # (B, T, vocab)

        return logits, {"x": x, "y": y, "rho": rho}
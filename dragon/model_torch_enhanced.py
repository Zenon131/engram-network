import torch
from torch import nn
import torch.nn.functional as F
from .sr_utils import EnhancedBDHLayer


class EnhancedBDH_GPU(nn.Module):
    def __init__(self, vocab_size=256, n_neurons=32768, d_model=256, n_layers=6, grid_size=(256, 128)):
        super().__init__()
        self.n = n_neurons
        self.d = d_model
        self.L = n_layers
        self.grid_size = grid_size

        self.emb = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [EnhancedBDHLayer(n_neurons, d_model, grid_size) for _ in range(n_layers)]
        )
        # Enhanced readout that incorporates SR features
        self.readout = nn.Linear(d_model * 2, vocab_size, bias=False)  # *2 for SR concatenation

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
        
        # Initialize state features
        state_features = None
        sr_features = None

        # Collapse sequence into a final "reasoned state"
        for t in range(T):
            # seed y from token embedding at step t
            y_seed = torch.matmul(h[:, t, :], torch.randn(self.d, self.n, device=h.device))  # toy
            y = F.relu(y_seed)

            # Forward through enhanced layers
            for i, layer in enumerate(self.layers):
                x, y, rho, state_features, sr_features = layer(x, y, rho, state_features)

        # Final logits depend on last y via E and SR features
        v_final = torch.matmul(y, self.layers[-1].E.T)  # (B, d)
        
        # Incorporate SR features into the final representation
        if sr_features is not None:
            # Concatenate SR features with the final state representation
            combined_features = torch.cat([v_final, sr_features], dim=-1)  # (B, d*2)
        else:
            # If no SR features, just use the original representation duplicated
            combined_features = torch.cat([v_final, v_final], dim=-1)  # (B, d*2)
            
        logits = self.readout(combined_features)  # (B, vocab)

        return logits, {"x": x, "y": y, "rho": rho, "state_features": state_features, "sr_features": sr_features}
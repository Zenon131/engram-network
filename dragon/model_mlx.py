import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from typing import Tuple, Optional, Dict, Any
from .layers import BDHLayer


class BDH_MLX(nn.Module):
    def __init__(self, vocab_size=256, n_neurons=32768, d_model=256, n_layers=6):
        super().__init__()
        self.n = n_neurons
        self.d = d_model
        self.L = n_layers

        self.emb = nn.Embedding(vocab_size, d_model)
        self.layers = [BDHLayer(n_neurons, d_model) for _ in range(n_layers)]
        self.readout = nn.Linear(d_model, vocab_size, bias=False)

    def __call__(self, tokens: mx.array, state: Optional[Dict[str, mx.array]] = None) -> Tuple[mx.array, Dict[str, mx.array]]:
        """
        tokens: (B, T)
        state: optional dict with 'x', 'y', 'rho' for B,T or recurrent streaming
        """
        B, T = tokens.shape
        h = self.emb(tokens)  # (B, T, d)

        # For simplicity, we recompute from scratch per T
        # Map token reps into neuron space: project h to n and sum over time
        # (toy: you'll want something more principled)
        x = mx.zeros((B, self.n))
        y = mx.zeros_like(x)
        rho = mx.zeros((B, self.n, self.d))

        # Collapse sequence into a final "reasoned state"
        for t in range(T):
            # seed y from token embedding at step t
            y_seed = h[:, t, :] @ mx.random.normal((self.d, self.n))  # toy
            y = nn.relu(y_seed)

            for layer in self.layers:
                x, y, rho = layer(x, y, rho)

        # Final logits depend on last y via E
        v_final = y @ self.layers[-1].E.T  # (B, d)
        logits = self.readout(v_final)      # (B, vocab)

        return logits, {"x": x, "y": y, "rho": rho}
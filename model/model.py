import torch
from torch import nn
import torch.nn.functional as F

class LinearAttention(nn.Module):
    def __init__(self, n_neurons, d_model):
        super().__init__()
        self.n = n_neurons
        self.d = d_model

    def forward(self, x, y, rho):
        """
        x: (B, n)   neuron activations at time t, layer l
        y: (B, n)   sparse-ish spikes at time t-1, layer l
        rho: (B, n, d)  fast weight state rho_{t-1,l}
        returns: new_y, new_rho
        """
        # keys/values from previous spikes
        kv = F.layer_norm(torch.matmul(y, self.E), (self.d,))   # (B, d)

        # Hebbian update: outer product kv x^T added to rho
        # kv: (B, d), x: (B, n) => outer: (B, n, d)
        outer = torch.einsum("bd,bn->bnd", kv, x)
        rho = rho + outer   # could also apply U / RoPE here

        # query via rho * x
        # rho: (B, n, d), x: (B, n) -> (B, d)
        attn_pre = torch.einsum("bnd,bn->bd", rho, x)
        attn_pre = F.layer_norm(attn_pre, (self.d,))

        # decode back to neurons using Dy
        y_tilde = torch.matmul(attn_pre, self.Dy.T)  # (B, n)
        y_new = F.relu(y_tilde) * x                  # elementwise ⊙

        return y_new, rho

class BDHLayer(nn.Module):
    def __init__(self, n_neurons, d_model):
        super().__init__()
        self.n = n_neurons
        self.d = d_model

        self.E = nn.Parameter(torch.randn(self.d, self.n) * 0.02)
        self.Dx = nn.Parameter(torch.randn(self.n, self.d) * 0.02)
        self.Dy = nn.Parameter(torch.randn(self.n, self.d) * 0.02)

        self.ln_y = nn.LayerNorm(self.d, elementwise_affine=False)
        self.ln_rho = nn.LayerNorm(self.d, elementwise_affine=False)

    def forward(self, x, y, rho):
        """
        x, y: (B, n)
        rho: (B, n, d)
        """
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

        return x, y, rho

class BDH_GPU(nn.Module):
    def __init__(self, vocab_size=256, n_neurons=32768, d_model=256, n_layers=6):
        super().__init__()
        self.n = n_neurons
        self.d = d_model
        self.L = n_layers

        self.emb = nn.Embedding(vocab_size, d_model)
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
        # (toy: you’ll want something more principled)
        x = torch.zeros(B, self.n, device=h.device)
        y = torch.zeros_like(x)
        rho = torch.zeros(B, self.n, self.d, device=h.device)

        # Collapse sequence into a final “reasoned state”
        for t in range(T):
            # seed y from token embedding at step t
            y_seed = torch.matmul(h[:, t, :], torch.randn(self.d, self.n, device=h.device))  # toy
            y = F.relu(y_seed)

            for layer in self.layers:
                x, y, rho = layer(x, y, rho)

        # Final logits depend on last y via E
        v_final = torch.matmul(y, self.layers[-1].E.T)  # (B, d)
        logits = self.readout(v_final)                  # (B, vocab)

        return logits, {"x": x, "y": y, "rho": rho}

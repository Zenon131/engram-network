import torch
from torch import nn
import torch.nn.functional as F

from typing import Optional


class LinearAttention(nn.Module):
    def __init__(self, n_neurons, d_model):
        super().__init__()
        self.n = n_neurons
        self.d = d_model

        self.E = nn.Parameter(torch.randn(self.d, self.n) * 0.02)
        self.Dy = nn.Parameter(torch.randn(self.n, self.d) * 0.02)

    def forward(self, x, y, rho):
        """
        x: (B, n)   neuron activations at time t, layer l
        y: (B, n)   sparse-ish spikes at time t-1, layer l
        rho: (B, n, d)  fast weight state rho_{t-1,l}
        returns: new_y, new_rho
        """
        # keys/values from previous spikes
        kv = F.layer_norm(torch.matmul(y, self.E.T), (self.d,))   # (B, d)

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


class SpatialBDHLayer(nn.Module):
    """
    BDH layer with distance-based neuron clustering.

    Extends the standard BDH computation with spatial modulation:
    neurons that are physically closer together in the spatial map have
    naturally stronger connections, while distant neurons interact more
    weakly.  This models biological wiring constraints where axon length
    (and therefore metabolic cost) limits long-range connectivity.

    Three interaction points are modulated by distance:
      1. Feedforward path  (y → E → Dx → x update)
      2. Hebbian update    (outer product into rho)  [optional]
      3. Attention output   (x → rho → Dy → y update)
    """

    def __init__(
        self,
        n_neurons: int,
        d_model: int,
        spatial_map,  # NeuronSpatialMap instance (shared across layers)
        modulate_feedforward: bool = True,
        modulate_hebbian: bool = True,
        modulate_output: bool = True,
    ):
        """
        Args:
            n_neurons: Number of neurons.
            d_model: Model dimension.
            spatial_map: A NeuronSpatialMap providing distance-based scaling.
                Typically shared across all layers so every layer sees the
                same physical neuron layout.
            modulate_feedforward: Apply spatial decay to the y→x feedforward.
            modulate_hebbian: Apply spatial decay to the Hebbian outer product.
            modulate_output: Apply spatial decay to the rho→y output path.
        """
        super().__init__()
        self.n = n_neurons
        self.d = d_model
        self.spatial_map = spatial_map
        self.modulate_feedforward = modulate_feedforward
        self.modulate_hebbian = modulate_hebbian
        self.modulate_output = modulate_output

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
        # 1) Feedforward: y → E → d → Dx → n
        v = torch.matmul(y, self.E.T)                   # (B, d)
        v = self.ln_y(v)

        x_update = F.relu(torch.matmul(v, self.Dx.T))   # (B, n)
        if self.modulate_feedforward:
            # Scale each target neuron's update by proximity to active y neurons
            x_update = self.spatial_map.modulate(source=y, target=x_update)
        x = x + x_update

        # 2) Hebbian update into rho
        if self.modulate_hebbian:
            # Attenuate x's contribution to rho by distance from y
            scale = self.spatial_map.get_interaction_scale(y)   # (B, n)
            outer = torch.einsum("bd,bn->bnd", v, x * scale)
        else:
            outer = torch.einsum("bd,bn->bnd", v, x)
        rho = rho + outer                                # (B, n, d)

        # 3) Attention readout: rho · x → q → Dy → y
        q = torch.einsum("bnd,bn->bd", rho, x)          # (B, d)
        q = self.ln_rho(q)

        y_tilde = torch.matmul(q, self.Dy.T)            # (B, n)
        if self.modulate_output:
            # Scale y output by proximity to active x neurons
            y_tilde = self.spatial_map.modulate(source=x, target=y_tilde)
        y = F.relu(y_tilde) * x

        return x, y, rho
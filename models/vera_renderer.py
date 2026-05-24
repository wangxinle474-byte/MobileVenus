"""VeraRetouch-style Encoder + per-pixel MLP Renderer.

Reference: [R3] VeraRetouch (arXiv 2604.27375, 2026)
Implementation: simplified single-latent version (no MLLM, no disentangled
masks). Replaces fixed 7D ISP / Bezier curves with a fully-learnable
per-pixel color mapping conditioned on a global latent.

Architecture:
  fused_feat (visual + action_emb) → MLP → latent z (R^Lz)
  per-pixel input  = sin/cos(RGB) + sin/cos(x,y) + z (broadcast)
  per-pixel output = RGB residual (added to input image)

Used in v10c experiment to test whether a fully data-driven per-pixel
renderer can break the 7D ISP ceiling on retouching tasks.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn


class _SineEncodingND(nn.Module):
    """Multi-channel sinusoidal positional encoding."""
    def __init__(self, in_dim: int, n_freq: int = 4):
        super().__init__()
        self.in_dim = in_dim
        self.n_freq = n_freq
        freqs = (2.0 ** torch.arange(n_freq).float()) * math.pi
        self.register_buffer('freqs', freqs)
        self.out_dim = in_dim + 2 * in_dim * n_freq

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., in_dim) → (..., in_dim + 2*in_dim*n_freq)"""
        scaled = x.unsqueeze(-1) * self.freqs   # (..., in_dim, n_freq)
        sin_enc = torch.sin(scaled)             # (..., in_dim, n_freq)
        cos_enc = torch.cos(scaled)             # (..., in_dim, n_freq)
        return torch.cat([x,
                          sin_enc.flatten(-2),
                          cos_enc.flatten(-2)], dim=-1)


class VeraRenderer(nn.Module):
    """Per-pixel conditional MLP color renderer (VeraRetouch-style).

    Args:
        fused_dim:    Input feature dim (e.g. 416 = visual_dim + action_emb_dim)
        latent_dim:   Compressed latent z dimension (default 32)
        rgb_n_freq:   Sinusoidal freqs on RGB (default 4)
        coord_n_freq: Sinusoidal freqs on (x, y) (default 4)
        hidden:       Per-pixel MLP hidden dim (default 64)
        n_layers:     Per-pixel MLP # hidden layers (default 4)
        residual:     Output added to input (True) or absolute (False)
        gate_init:    Initial gate scalar (0 → identity at start)
    """
    def __init__(self,
                 fused_dim: int,
                 latent_dim: int = 32,
                 rgb_n_freq: int = 4,
                 coord_n_freq: int = 4,
                 hidden: int = 64,
                 n_layers: int = 4,
                 dropout: float = 0.1,
                 residual: bool = True,
                 gate_init: float = 0.0):
        super().__init__()
        self.residual = residual

        # Project fused → latent z
        self.latent_proj = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Dropout(dropout),
            nn.Linear(fused_dim, 128),
            nn.GELU(),
            nn.Linear(128, latent_dim),
        )

        # Sinusoidal encodings
        self.rgb_enc = _SineEncodingND(in_dim=3, n_freq=rgb_n_freq)
        self.coord_enc = _SineEncodingND(in_dim=2, n_freq=coord_n_freq)

        # Per-pixel MLP input dim
        in_dim = self.rgb_enc.out_dim + self.coord_enc.out_dim + latent_dim

        # Per-pixel MLP
        layers = []
        for i in range(n_layers):
            layers.append(nn.Linear(in_dim if i == 0 else hidden, hidden))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden, 3))
        self.mlp = nn.Sequential(*layers)
        # Zero-init final layer → at start MLP outputs ~0, residual=0
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

        # Learnable gate scalar
        self.gate = nn.Parameter(torch.tensor(gate_init, dtype=torch.float32))

    def _make_coord_grid(self, B: int, H: int, W: int,
                          device: torch.device) -> torch.Tensor:
        """Generate normalized (x, y) coords ∈ [-1, 1] for each pixel.

        Returns (B, 2, H, W) tensor.
        """
        xs = torch.linspace(-1.0, 1.0, W, device=device)
        ys = torch.linspace(-1.0, 1.0, H, device=device)
        gy, gx = torch.meshgrid(ys, xs, indexing='ij')   # (H, W) each
        coord = torch.stack([gx, gy], dim=0)              # (2, H, W)
        return coord.unsqueeze(0).expand(B, 2, H, W)     # (B, 2, H, W)

    def forward(self, img: torch.Tensor, fused: torch.Tensor) -> torch.Tensor:
        """Apply per-pixel conditional MLP to render output.

        Args:
            img:   (B, 3, H, W) input image in [0, 1]
            fused: (B, fused_dim) global feature (visual + action_emb)
        Returns:
            (B, 3, H, W) output image in [0, 1]
        """
        B, C, H, W = img.shape

        # Project fused → latent z (B, latent_dim)
        z = self.latent_proj(fused)

        # Encode RGB per-pixel: (B, 3, H, W) → (B*H*W, rgb_enc_dim)
        rgb_pp = img.permute(0, 2, 3, 1).reshape(-1, 3)   # (BHW, 3)
        rgb_enc = self.rgb_enc(rgb_pp)                     # (BHW, rgb_enc_dim)

        # Encode (x, y) coords per-pixel
        coord = self._make_coord_grid(B, H, W, img.device)         # (B,2,H,W)
        coord_pp = coord.permute(0, 2, 3, 1).reshape(-1, 2)        # (BHW, 2)
        coord_enc = self.coord_enc(coord_pp)                        # (BHW, c_dim)

        # Broadcast latent z to per-pixel: (B, Lz) → (BHW, Lz)
        z_pp = z.unsqueeze(1).unsqueeze(1).expand(B, H, W, z.shape[1])
        z_pp = z_pp.reshape(-1, z.shape[1])

        # Concatenate inputs and run MLP
        x = torch.cat([rgb_enc, coord_enc, z_pp], dim=-1)  # (BHW, in_dim)
        delta = self.mlp(x)                                 # (BHW, 3)
        delta = delta.reshape(B, H, W, 3).permute(0, 3, 1, 2)  # (B,3,H,W)

        if self.residual:
            out = img + self.gate * delta
        else:
            out = self.gate * delta + (1.0 - self.gate) * img
        return out.clamp(0.0, 1.0)

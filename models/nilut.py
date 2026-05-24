"""Neural Implicit LUT (NILUT) — Conditional Neural Implicit 3D Lookup Tables
for Image Enhancement.

Reference: [R6] NILUT (Conde et al., AAAI 2024, arxiv 2306.11920)
Implementation: simplified single-style version (no multi-style conditioning).

Used in v10b experiment: residual color correction on top of v9 NamedCurves
output. Adds expressivity beyond the fixed 7D ISP / Bezier curves to capture
hue rotation, per-channel curves, split-toning that 7D cannot represent.

Design choices:
- Sinusoidal positional encoding on (R, G, B) for high-frequency capture
- Compact MLP (3-layer, hidden=32) → ~3K parameters
- Output is residual color delta, not absolute color → safe gradient init
- A learnable scalar `gate` (init=0) controls residual magnitude → progressive
  residual learning that doesn't disrupt the 7D ISP/Bezier pretrained signal.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn


class _SineEncoding(nn.Module):
    """Sinusoidal positional encoding for 3D color coords.

    Maps (R, G, B) ∈ [0,1] → 3 + 6*L dim feature where L is # frequencies.
    Standard NeRF-style encoding for capturing high-frequency color variations.
    """
    def __init__(self, n_freq: int = 4):
        super().__init__()
        self.n_freq = n_freq
        # Frequencies: 2^0, 2^1, ..., 2^(n_freq-1) cycles per [0,1]
        freqs = (2.0 ** torch.arange(n_freq).float()) * math.pi
        self.register_buffer('freqs', freqs)  # (n_freq,)

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        """rgb: (..., 3) → (..., 3 + 3*2*n_freq)"""
        # Encode each color channel separately with sin/cos at multiple freqs
        # rgb: (..., 3), freqs: (n_freq,)
        # Broadcast to (..., 3, n_freq)
        scaled = rgb.unsqueeze(-1) * self.freqs  # (..., 3, n_freq)
        sin_enc = torch.sin(scaled)              # (..., 3, n_freq)
        cos_enc = torch.cos(scaled)              # (..., 3, n_freq)
        flat = torch.cat([rgb,
                          sin_enc.flatten(-2),
                          cos_enc.flatten(-2)], dim=-1)
        return flat


class NILUT(nn.Module):
    """Neural Implicit LUT — small MLP predicting RGB → RGB color transform.

    Args:
        n_freq:     Sinusoidal positional encoding frequencies (default 4)
        hidden:     MLP hidden dimension (default 32)
        n_layers:   Number of hidden layers (default 3)
        residual:   If True, output is treated as delta (added to input).
                    If False, output is absolute color. Default True.
        gate_init:  Initial value of learnable gate scalar (default 0.0).
                    With gate=0, NILUT contributes nothing initially → safe
                    drop-in residual head that doesn't disrupt pretrained model.
    """
    def __init__(self,
                 n_freq: int = 4,
                 hidden: int = 32,
                 n_layers: int = 3,
                 residual: bool = True,
                 gate_init: float = 0.0):
        super().__init__()
        self.residual = residual
        self.encoding = _SineEncoding(n_freq=n_freq)
        in_dim = 3 + 3 * 2 * n_freq  # 3 raw + 3*2*n_freq sin/cos

        layers = []
        for i in range(n_layers):
            layers.append(nn.Linear(in_dim if i == 0 else hidden, hidden))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden, 3))
        self.mlp = nn.Sequential(*layers)
        # Zero-init final layer → at start the MLP outputs ~0 (residual=0)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

        # Learnable gate scalar (single value) controlling residual magnitude.
        # init=0 → no residual at start; rapidly learnable since gradient ~ MLP_out.
        self.gate = nn.Parameter(torch.tensor(gate_init, dtype=torch.float32))

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """Apply NILUT color transform to an image.

        Args:
            img: (B, 3, H, W) in [0, 1]
        Returns:
            (B, 3, H, W) in [0, 1] (residual: img + gate * MLP(img))
        """
        B, C, H, W = img.shape
        # Reshape to per-pixel (BHW, 3) for MLP
        x = img.permute(0, 2, 3, 1).reshape(-1, 3)        # (BHW, 3)
        x_enc = self.encoding(x)                           # (BHW, in_dim)
        delta = self.mlp(x_enc)                            # (BHW, 3)
        delta = delta.reshape(B, H, W, 3).permute(0, 3, 1, 2)  # (B,3,H,W)

        if self.residual:
            out = img + self.gate * delta
        else:
            out = self.gate * delta
        return out.clamp(0.0, 1.0)

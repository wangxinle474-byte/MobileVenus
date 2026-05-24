"""Path X: Implicit Residual Head (INRetouch-inspired).

A small U-Net that learns the pixel residual between v11a's 7D-ISP-bounded
output and the FireRed PNG target. This breaks the 7D ISP rendering ceiling
by allowing arbitrary per-pixel corrections.

Architecture:
  Input:  concat(orig, refined) = 6ch  [256×256]
  Action conditioning: FiLM at each decoder level
  Encoder: 4 levels (32→64→128→256)
  Decoder: 4 levels (256→128→64→32) with skip connections
  Output: 3ch residual, zero-init final conv
  Final:  refined + gate * tanh(residual)
  gate starts at 0 → identity at init (stable training)

~1.2M parameters.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ConvBlock(nn.Module):
    """Two 3×3 convs with GroupNorm + GELU."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class _DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.down = nn.Conv2d(in_ch, in_ch, 4, stride=2, padding=1, bias=False)
        self.conv = _ConvBlock(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.down(x))


class _UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch, 4, stride=2, padding=1,
                                     bias=False)
        self.conv = _ConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # Handle size mismatch from odd dimensions
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear',
                              align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class _FiLM(nn.Module):
    """Feature-wise Linear Modulation for action conditioning."""

    def __init__(self, cond_dim: int, n_ch: int):
        super().__init__()
        self.fc = nn.Linear(cond_dim, n_ch * 2)
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x, cond):
        """x: (B, C, H, W), cond: (B, cond_dim)."""
        params = self.fc(cond)  # (B, 2C)
        gamma, beta = params.chunk(2, dim=1)  # (B, C) each
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return x * (1 + gamma) + beta


class ImplicitResidualHead(nn.Module):
    """Small U-Net that predicts pixel residual to break 7D ISP ceiling.

    Args:
        in_ch: input channels (default 6 = orig + refined)
        base_ch: base channel width (32 → ~1.2M params)
        n_actions: number of action classes for FiLM conditioning
        gate_init: initial gate value. Default 1.0 because the final conv
            is zero-init (residual=0 at start → identity output even with
            gate=1). Setting gate_init=0 creates a fixed point where both
            residual AND gate gradients are 0, blocking learning entirely.
    """

    def __init__(self, in_ch: int = 6, base_ch: int = 32,
                 n_actions: int = 5, gate_init: float = 1.0):
        super().__init__()
        ch1, ch2, ch3, ch4 = base_ch, base_ch*2, base_ch*4, base_ch*8

        # Encoder
        self.enc1 = _ConvBlock(in_ch, ch1)      # 256 → 256, 32ch
        self.enc2 = _DownBlock(ch1, ch2)         # 256 → 128, 64ch
        self.enc3 = _DownBlock(ch2, ch3)         # 128 → 64,  128ch
        self.enc4 = _DownBlock(ch3, ch4)         # 64  → 32,  256ch

        # Action conditioning (FiLM at each decoder level)
        cond_dim = n_actions
        self.film4 = _FiLM(cond_dim, ch4)
        self.film3 = _FiLM(cond_dim, ch3)
        self.film2 = _FiLM(cond_dim, ch2)
        self.film1 = _FiLM(cond_dim, ch1)

        # Decoder
        self.dec3 = _UpBlock(ch4, ch3, ch3)      # 32 → 64,  128ch
        self.dec2 = _UpBlock(ch3, ch2, ch2)      # 64 → 128, 64ch
        self.dec1 = _UpBlock(ch2, ch1, ch1)      # 128→ 256, 32ch

        # Final projection → 3ch residual (zero-init for identity start)
        self.final = nn.Conv2d(ch1, 3, 1, bias=True)
        nn.init.zeros_(self.final.weight)
        nn.init.zeros_(self.final.bias)

        # Learnable gate (starts at gate_init, typically 0)
        self.gate = nn.Parameter(torch.tensor(gate_init))

    def forward(self, orig: torch.Tensor, refined: torch.Tensor,
                action_onehot: torch.Tensor) -> torch.Tensor:
        """
        Args:
            orig: (B, 3, H, W) original input image
            refined: (B, 3, H, W) v11a backbone output (7D ISP-bounded)
            action_onehot: (B, n_actions) action conditioning

        Returns:
            final: (B, 3, H, W) = refined + gate * tanh(residual)
        """
        x = torch.cat([orig, refined], dim=1)  # (B, 6, H, W)

        # Encoder path
        e1 = self.enc1(x)       # (B, ch1, H, W)
        e2 = self.enc2(e1)      # (B, ch2, H/2, W/2)
        e3 = self.enc3(e2)      # (B, ch3, H/4, W/4)
        e4 = self.enc4(e3)      # (B, ch4, H/8, W/8)

        # Bottleneck FiLM
        e4 = self.film4(e4, action_onehot)

        # Decoder path with FiLM
        d3 = self.dec3(e4, e3)
        d3 = self.film3(d3, action_onehot)

        d2 = self.dec2(d3, e2)
        d2 = self.film2(d2, action_onehot)

        d1 = self.dec1(d2, e1)
        d1 = self.film1(d1, action_onehot)

        # Residual prediction
        residual = self.final(d1)  # (B, 3, H, W)

        # Gated addition (tanh bounds residual to [-1, 1])
        final = refined + self.gate * torch.tanh(residual)
        return final.clamp(0.0, 1.0)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

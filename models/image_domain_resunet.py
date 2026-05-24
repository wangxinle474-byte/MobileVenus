"""Path Z: pure image-domain ResUNet for FireRed pseudo-label distillation.

Drops the 7D ISP parameterization entirely. Predicts the edited image
directly from (orig, action) → refined. No Bezier, no LUT, no NamedCurves.

This is the "abandon ISP ceiling" baseline — directly compares against
Path A (v11a + 7D ISP) and Path X (v11a backbone + residual head).

Architecture:
  Input:  orig (3ch) + action one-hot (5)
  Encoder: 4-level U-Net with FiLM action conditioning
  Decoder: 4-level mirror with skip connections
  Output: refined = orig + tanh(delta), where delta ∈ [-1, 1]
  Initialization: zero-init final conv → identity start

base_ch=48 → ~10M params (matches v11a backbone scale).
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
        params = self.fc(cond)
        gamma, beta = params.chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return x * (1 + gamma) + beta


class ImageDomainResUNet(nn.Module):
    """Pure image-domain ResUNet — no 7D ISP intermediate.

    Args:
        in_ch: input channels (default 3 = RGB)
        base_ch: base channel width (48 → ~10M params)
        n_actions: number of action classes
        delta_scale: bound for the residual delta (default 1.0)
    """

    def __init__(self, in_ch: int = 3, base_ch: int = 48,
                 n_actions: int = 5, delta_scale: float = 1.0):
        super().__init__()
        self.delta_scale = delta_scale
        ch1, ch2, ch3, ch4 = base_ch, base_ch*2, base_ch*4, base_ch*8

        # Encoder
        self.enc1 = _ConvBlock(in_ch, ch1)
        self.enc2 = _DownBlock(ch1, ch2)
        self.enc3 = _DownBlock(ch2, ch3)
        self.enc4 = _DownBlock(ch3, ch4)

        # FiLM at each level for action conditioning
        cond_dim = n_actions
        self.film_e1 = _FiLM(cond_dim, ch1)
        self.film_e2 = _FiLM(cond_dim, ch2)
        self.film_e3 = _FiLM(cond_dim, ch3)
        self.film_e4 = _FiLM(cond_dim, ch4)
        self.film_d3 = _FiLM(cond_dim, ch3)
        self.film_d2 = _FiLM(cond_dim, ch2)
        self.film_d1 = _FiLM(cond_dim, ch1)

        # Decoder
        self.dec3 = _UpBlock(ch4, ch3, ch3)
        self.dec2 = _UpBlock(ch3, ch2, ch2)
        self.dec1 = _UpBlock(ch2, ch1, ch1)

        # Final 1×1 conv → 3-channel residual (zero-init for identity start)
        self.final = nn.Conv2d(ch1, 3, 1, bias=True)
        nn.init.zeros_(self.final.weight)
        nn.init.zeros_(self.final.bias)

    def forward(self, orig: torch.Tensor,
                action_onehot: torch.Tensor) -> torch.Tensor:
        """
        Args:
            orig: (B, 3, H, W) input image, range [0, 1]
            action_onehot: (B, n_actions) action conditioning

        Returns:
            refined: (B, 3, H, W) = clamp(orig + delta_scale * tanh(delta),
                                          0, 1)
        """
        # Encoder
        e1 = self.film_e1(self.enc1(orig), action_onehot)
        e2 = self.film_e2(self.enc2(e1), action_onehot)
        e3 = self.film_e3(self.enc3(e2), action_onehot)
        e4 = self.film_e4(self.enc4(e3), action_onehot)

        # Decoder
        d3 = self.film_d3(self.dec3(e4, e3), action_onehot)
        d2 = self.film_d2(self.dec2(d3, e2), action_onehot)
        d1 = self.film_d1(self.dec1(d2, e1), action_onehot)

        # Residual prediction
        delta = self.final(d1)  # zero at init → identity output
        refined = orig + self.delta_scale * torch.tanh(delta)
        return refined.clamp(0.0, 1.0)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

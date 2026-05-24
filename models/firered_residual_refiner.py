"""v13: lightweight image-domain residual refiner on top of v11-d base.

Goal:
  v11-d (7D ISP / NamedCurves) gives a good but conservative parametric base.
  This refiner learns a small residual on top of the base so that the final
  output matches FireRed pseudo-labels more closely (especially for wb /
  saturation / brightness where parametric ISP is the weakest).

Pipeline (training & inference):
  base = v11d(orig, action)            # frozen
  delta = refiner(orig, base, action)  # learned, zero-init
  final = clamp(base + delta_scale * tanh(delta), 0, 1)

Architecture:
  4-level U-Net with FiLM action conditioning. Input is 6 channels
  (orig RGB + base RGB) so the network can attend to where v11-d disagrees
  with FireRed and only edit those regions. Final 1x1 conv is zero-init so
  the network starts as exact identity (final == base) and only deviates
  when supervision pulls it that way.

Param budget:
  base_ch=32 → ~4.5 M
  base_ch=48 → ~10  M (default, matches Path Z scale)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ConvBlock(nn.Module):
    """Two 3x3 convs with GroupNorm + GELU."""

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
    """Feature-wise Linear Modulation for action conditioning.

    Zero-init both gamma and beta so action conditioning is identity at start.
    """

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


class FireRedResidualRefiner(nn.Module):
    """v13 residual refiner on top of v11-d base.

    Args:
        base_ch: U-Net base channels (32 → ~4.5M, 48 → ~10M).
        n_actions: number of action classes (5 or 7).
        delta_scale: bound for the residual delta (default 0.5 = ±0.5).

    Forward:
        orig:           (B, 3, H, W) input image in [0, 1]
        base:           (B, 3, H, W) v11-d output in [0, 1]
        action_onehot:  (B, n_actions)
    Returns:
        refined:        (B, 3, H, W) = clamp(base + delta_scale * tanh(delta))
        delta:          (B, 3, H, W) raw residual (pre-clamp), useful for
                        regularization losses (L1 on delta encourages
                        sparseness).
    """

    def __init__(self, base_ch: int = 48, n_actions: int = 7,
                 delta_scale: float = 0.5):
        super().__init__()
        self.delta_scale = delta_scale
        self.n_actions = n_actions
        ch1, ch2, ch3, ch4 = base_ch, base_ch * 2, base_ch * 4, base_ch * 8

        # 6-channel input: orig (3) + base (3)
        self.enc1 = _ConvBlock(6, ch1)
        self.enc2 = _DownBlock(ch1, ch2)
        self.enc3 = _DownBlock(ch2, ch3)
        self.enc4 = _DownBlock(ch3, ch4)

        cond_dim = n_actions
        self.film_e1 = _FiLM(cond_dim, ch1)
        self.film_e2 = _FiLM(cond_dim, ch2)
        self.film_e3 = _FiLM(cond_dim, ch3)
        self.film_e4 = _FiLM(cond_dim, ch4)
        self.film_d3 = _FiLM(cond_dim, ch3)
        self.film_d2 = _FiLM(cond_dim, ch2)
        self.film_d1 = _FiLM(cond_dim, ch1)

        self.dec3 = _UpBlock(ch4, ch3, ch3)
        self.dec2 = _UpBlock(ch3, ch2, ch2)
        self.dec1 = _UpBlock(ch2, ch1, ch1)

        # zero-init final conv → delta=0 at start → refined == base
        self.final = nn.Conv2d(ch1, 3, 1, bias=True)
        nn.init.zeros_(self.final.weight)
        nn.init.zeros_(self.final.bias)

    def forward(self, orig: torch.Tensor, base: torch.Tensor,
                action_onehot: torch.Tensor):
        x = torch.cat([orig, base], dim=1)  # (B, 6, H, W)

        e1 = self.film_e1(self.enc1(x), action_onehot)
        e2 = self.film_e2(self.enc2(e1), action_onehot)
        e3 = self.film_e3(self.enc3(e2), action_onehot)
        e4 = self.film_e4(self.enc4(e3), action_onehot)

        d3 = self.film_d3(self.dec3(e4, e3), action_onehot)
        d2 = self.film_d2(self.dec2(d3, e2), action_onehot)
        d1 = self.film_d1(self.dec1(d2, e1), action_onehot)

        delta = self.final(d1)  # zero at init
        refined = base + self.delta_scale * torch.tanh(delta)
        refined = refined.clamp(0.0, 1.0)
        return refined, delta

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

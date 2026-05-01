"""Refinement Net — 轻量图像增强精修网络 (~500K params)"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        return x * self.sigmoid(self.conv(y))


class EnhanceBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm2d(channels, affine=True)
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()
        self.act = nn.LeakyReLU(0.2, inplace=True)
    def forward(self, x):
        residual = x
        out = self.act(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out = self.ca(out)
        out = self.sa(out)
        return self.act(out + residual)


class MultiScaleFeature(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.branch1 = nn.Conv2d(in_ch, out_ch//4, 1)
        self.branch3 = nn.Conv2d(in_ch, out_ch//4, 3, padding=1)
        self.branch5 = nn.Conv2d(in_ch, out_ch//4, 5, padding=2)
        self.branch7 = nn.Conv2d(in_ch, out_ch//4, 7, padding=3)
        self.fuse = nn.Conv2d(out_ch, out_ch, 1)
        self.act = nn.LeakyReLU(0.2, inplace=True)
    def forward(self, x):
        return self.act(self.fuse(torch.cat([self.branch1(x), self.branch3(x), self.branch5(x), self.branch7(x)], dim=1)))


class RefinementNet(nn.Module):
    def __init__(self, base_ch=32, n_blocks=6):
        super().__init__()
        self.ms_entry = MultiScaleFeature(3, base_ch)
        self.blocks = nn.Sequential(*[EnhanceBlock(base_ch) for _ in range(n_blocks)])
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_fc = nn.Sequential(
            nn.Linear(base_ch, base_ch), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(base_ch, base_ch), nn.Sigmoid(),
        )
        self.out_conv = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, 3, padding=1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_ch, 3, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_out')
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.2)
                if m.bias is not None: nn.init.zeros_(m.bias)
        nn.init.zeros_(self.out_conv[-1].weight)
        nn.init.zeros_(self.out_conv[-1].bias)

    def forward(self, x):
        feat = self.ms_entry(x)
        feat = self.blocks(feat)
        g = self.global_pool(feat).flatten(1)
        g = self.global_fc(g).unsqueeze(-1).unsqueeze(-1)
        feat = feat * g
        residual = self.out_conv(feat)
        return (x + residual).clamp(0, 1)

    @torch.no_grad()
    def count_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

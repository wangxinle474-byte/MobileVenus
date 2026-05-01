"""
轻量神经 ISP — FiLM-conditioned 残差网络
替代 diff_isp，学习 (raw_image, params) → expert_image 的映射。
~300K params，适合移动端。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FiLMLayer(nn.Module):
    def __init__(self, param_dim, feature_dim):
        super().__init__()
        self.fc = nn.Linear(param_dim, feature_dim * 2)
    def forward(self, feature, params):
        modulation = self.fc(params)
        gamma, beta = modulation.chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return feature * (1.0 + gamma) + beta


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.norm = nn.InstanceNorm2d(out_ch, affine=True)
        self.act = nn.LeakyReLU(0.2, inplace=True)
    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class ResBlock(nn.Module):
    def __init__(self, channels, param_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm2d(channels, affine=True)
        self.film = FiLMLayer(param_dim, channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)
    def forward(self, x, params):
        residual = x
        out = self.act(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out = self.film(out, params)
        return self.act(out + residual)


class NeuralISP(nn.Module):
    def __init__(self, param_dim=6, base_ch=32, n_res_blocks=4):
        super().__init__()
        self.param_dim = param_dim
        self.param_embed = nn.Sequential(
            nn.Linear(param_dim, 64), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 64), nn.LeakyReLU(0.2, inplace=True),
        )
        pe_dim = 64
        ch1, ch2, ch3 = base_ch, base_ch*2, base_ch*4
        self.enc1 = ConvBlock(3, ch1, stride=1)
        self.enc2 = ConvBlock(ch1, ch2, stride=2)
        self.enc3 = ConvBlock(ch2, ch3, stride=2)
        self.bottleneck = nn.ModuleList([ResBlock(ch3, pe_dim) for _ in range(n_res_blocks)])
        self.up3 = nn.ConvTranspose2d(ch3, ch2, 4, stride=2, padding=1)
        self.dec3 = ConvBlock(ch2*2, ch2)
        self.film3 = FiLMLayer(pe_dim, ch2)
        self.up2 = nn.ConvTranspose2d(ch2, ch1, 4, stride=2, padding=1)
        self.dec2 = ConvBlock(ch1*2, ch1)
        self.film2 = FiLMLayer(pe_dim, ch1)
        self.out_conv = nn.Sequential(
            nn.Conv2d(ch1, ch1, 3, padding=1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ch1, 3, 1), nn.Tanh(),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_out')
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.2)
                if m.bias is not None: nn.init.zeros_(m.bias)
        nn.init.zeros_(self.out_conv[-2].weight)

    def forward(self, img, params):
        pe = self.param_embed(params)
        e1 = self.enc1(img)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        x = e3
        for resblk in self.bottleneck:
            x = resblk(x, pe)
        x = self.up3(x)
        x = torch.cat([x, e2], dim=1)
        x = self.dec3(x)
        x = self.film3(x, pe)
        x = self.up2(x)
        x = torch.cat([x, e1], dim=1)
        x = self.dec2(x)
        x = self.film2(x, pe)
        residual = self.out_conv(x) * 0.5
        return (img + residual).clamp(0, 1)

    @torch.no_grad()
    def count_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


def neural_isp_loss(pred, target, l1_weight=1.0, ssim_weight=1.0):
    from models.diff_isp import ssim_loss
    l1 = F.l1_loss(pred, target)
    ssim_val = ssim_loss(pred, target)
    return {'total': l1_weight*l1 + ssim_weight*ssim_val, 'l1': l1, 'ssim': ssim_val}

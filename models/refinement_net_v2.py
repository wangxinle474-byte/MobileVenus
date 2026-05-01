"""RefinementNet v2 — U-Net ~1M params"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class CBAM(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.ca_fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
        )
        self.sa_conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)
    def forward(self, x):
        b, c, _, _ = x.size()
        avg_c = self.ca_fc(self.avg_pool(x).view(b, c))
        max_c = self.ca_fc(self.max_pool(x).view(b, c))
        ca = torch.sigmoid(avg_c + max_c).view(b, c, 1, 1)
        x = x * ca
        avg_s = x.mean(dim=1, keepdim=True)
        max_s, _ = x.max(dim=1, keepdim=True)
        sa = torch.sigmoid(self.sa_conv(torch.cat([avg_s, max_s], dim=1)))
        return x * sa

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm2d(channels, affine=True)
        self.cbam = CBAM(channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)
    def forward(self, x):
        out = self.act(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out = self.cbam(out)
        return self.act(out + x)

class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, n_res=2):
        super().__init__()
        self.down = nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1, bias=False)
        self.norm = nn.InstanceNorm2d(out_ch, affine=True)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.blocks = nn.Sequential(*[ResBlock(out_ch) for _ in range(n_res)])
    def forward(self, x):
        x = self.act(self.norm(self.down(x)))
        return self.blocks(x)

class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, n_res=2):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1, bias=False)
        self.norm = nn.InstanceNorm2d(out_ch, affine=True)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.fuse = nn.Conv2d(out_ch + skip_ch, out_ch, 1, bias=False)
        self.blocks = nn.Sequential(*[ResBlock(out_ch) for _ in range(n_res)])
    def forward(self, x, skip):
        x = self.act(self.norm(self.up(x)))
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = self.fuse(torch.cat([x, skip], dim=1))
        return self.blocks(x)

class ColorModulation(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels * 2), nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(channels * 2, channels), nn.Sigmoid(),
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        g = self.pool(x).view(b, c)
        return x * self.fc(g).view(b, c, 1, 1)

class RefinementNetV2(nn.Module):
    def __init__(self, base_ch=64):
        super().__init__()
        c = base_ch
        self.entry = nn.Sequential(
            nn.Conv2d(3, c, 3, padding=1, bias=False),
            nn.InstanceNorm2d(c, affine=True), nn.LeakyReLU(0.2, inplace=True), ResBlock(c),
        )
        self.down1 = DownBlock(c, c*2, n_res=2)
        self.down2 = DownBlock(c*2, c*4, n_res=2)
        self.bottleneck = nn.Sequential(ResBlock(c*4), ResBlock(c*4), CBAM(c*4))
        self.up2 = UpBlock(c*4, c*2, c*2, n_res=2)
        self.up1 = UpBlock(c*2, c, c, n_res=2)
        self.color_mod = ColorModulation(c)
        self.out_conv = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(c, 3, 1),
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
        nn.init.zeros_(self.out_conv[-1].weight)
        nn.init.zeros_(self.out_conv[-1].bias)
    def forward(self, x):
        e0 = self.entry(x)
        e1 = self.down1(e0)
        e2 = self.down2(e1)
        b = self.bottleneck(e2)
        d1 = self.up2(b, e1)
        d0 = self.up1(d1, e0)
        d0 = self.color_mod(d0)
        return (x + self.out_conv(d0)).clamp(0, 1)
    @torch.no_grad()
    def count_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

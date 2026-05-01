"""
RefinementNet v3 — 大容量图像增强网络 (~9M 参数)

相比 v2 (1.9M @ base_ch=32):
  - 48 base channels (v2 实际用 32)
  - 3级编解码器 (v2=2级), /8 下采样, 感受野翻倍
  - 瓶颈层 3 个 ResBlock + CBAM (v2=2个)
  - 增强色彩调制: 仿射变换 scale+shift (v2 仅 scale)
  - 总参数 ~9M

目标: 突破 MUSIQ 4.15 天花板, 达到 4.20+
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
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


class ColorModulationV2(nn.Module):
    """增强色彩调制: 仿射变换 scale + shift"""
    def __init__(self, channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(channels * 2, channels * 2),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        g = self.pool(x).view(b, c)
        g = self.fc(g)
        scale = torch.sigmoid(g[:, :c]).view(b, c, 1, 1)
        shift = (torch.tanh(g[:, c:]) * 0.1).view(b, c, 1, 1)
        return x * scale + shift


class RefinementNetV3(nn.Module):
    """
    U-Net v3 (~9M params)

    Entry (3->48) -> Down1 (48->96) -> Down2 (96->192) -> Down3 (192->192)
    -> Bottleneck (192, 3xResBlock + CBAM)
    -> Up3 (192->192) -> Up2 (192->96) -> Up1 (96->48)
    -> ColorModV2 -> Output (48->3)
    """

    def __init__(self, base_ch=48):
        super().__init__()
        c = base_ch

        self.entry = nn.Sequential(
            nn.Conv2d(3, c, 3, padding=1, bias=False),
            nn.InstanceNorm2d(c, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(c),
        )

        self.down1 = DownBlock(c, c * 2, n_res=2)
        self.down2 = DownBlock(c * 2, c * 4, n_res=2)
        self.down3 = DownBlock(c * 4, c * 4, n_res=2)

        self.bottleneck = nn.Sequential(
            ResBlock(c * 4),
            ResBlock(c * 4),
            ResBlock(c * 4),
            CBAM(c * 4),
        )

        self.up3 = UpBlock(c * 4, c * 4, c * 4, n_res=2)
        self.up2 = UpBlock(c * 4, c * 2, c * 2, n_res=2)
        self.up1 = UpBlock(c * 2, c, c, n_res=2)

        self.color_mod = ColorModulationV2(c)

        self.out_conv = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c, 3, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.2)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.zeros_(self.out_conv[-1].weight)
        nn.init.zeros_(self.out_conv[-1].bias)

    def forward(self, x):
        """x: (B, 3, H, W) [0,1] -> (B, 3, H, W) [0,1]"""
        e0 = self.entry(x)
        e1 = self.down1(e0)
        e2 = self.down2(e1)
        e3 = self.down3(e2)

        b = self.bottleneck(e3)

        d2 = self.up3(b, e2)
        d1 = self.up2(d2, e1)
        d0 = self.up1(d1, e0)

        d0 = self.color_mod(d0)
        residual = self.out_conv(d0)

        return (x + residual).clamp(0, 1)

    @torch.no_grad()
    def count_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

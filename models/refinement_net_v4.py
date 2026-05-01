"""RefinementNet V4: 双分支像素级图像精修网络。

在 diff_isp 粗渲染基础上进行像素级精修，突破全局 ISP 参数的天花板。

双分支架构:
- 全局色彩分支: 自适应 3×3 色彩矩阵 + per-channel gamma 曲线
- 局部细节分支: 3 级 U-Net 解码器 + 空间细节残差

参数量: ~16M
输入: (B, 3, H, W) diff_isp 粗渲染图
输出: (B, 3, H, W) 精修后高质量图
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CBAM(nn.Module):
    """Convolutional Block Attention Module (通道 + 空间注意力)。"""

    def __init__(self, channels, reduction=8):
        super().__init__()
        # 通道注意力
        mid = max(channels // reduction, 8)
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
        )
        self.channel_max = nn.AdaptiveMaxPool2d(1)
        self.channel_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
        )

        # 空间注意力
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # 通道注意力
        avg_out = self.channel_attn(x)
        max_out = self.channel_fc(self.channel_max(x).flatten(1))
        channel_w = torch.sigmoid(avg_out + max_out).unsqueeze(-1).unsqueeze(-1)
        x = x * channel_w

        # 空间注意力
        avg_pool = x.mean(dim=1, keepdim=True)
        max_pool = x.max(dim=1, keepdim=True)[0]
        spatial_w = self.spatial_attn(torch.cat([avg_pool, max_pool], dim=1))
        x = x * spatial_w

        return x


class ResBlock(nn.Module):
    """残差块: Conv + IN + GELU + Conv + IN + 残差。"""

    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x):
        return x + self.block(x)


class EncoderBlock(nn.Module):
    """编码器块: Conv 下采样 + ResBlock + CBAM。"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch),
            nn.GELU(),
        )
        self.res = ResBlock(out_ch)
        self.attn = CBAM(out_ch)

    def forward(self, x):
        x = self.down(x)
        x = self.res(x)
        x = self.attn(x)
        return x


class DecoderBlock(nn.Module):
    """解码器块: Upsample + Skip concat + Conv + ResBlock。"""

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear',
                              align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch),
            nn.GELU(),
        )
        self.res = ResBlock(out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # 处理尺寸不匹配
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, skip.shape[2:], mode='bilinear',
                              align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        x = self.res(x)
        return x


class GlobalColorBranch(nn.Module):
    """全局色彩分支: 自适应 3×3 色彩矩阵 + per-channel gamma。

    学习全局色彩变换:
        out_color = ColorMatrix @ in_color + bias
        out = out_color ^ gamma
    """

    def __init__(self, base_ch=64):
        super().__init__()
        # 从图像提取全局统计
        self.encoder = nn.Sequential(
            nn.Conv2d(3, base_ch, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(base_ch, base_ch * 2, 3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

        feat_dim = base_ch * 2

        # 预测 3×3 色彩矩阵 + bias
        self.color_matrix_head = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.GELU(),
            nn.Linear(64, 9 + 3),  # 3×3 matrix + 3 bias
        )

        # 预测 per-channel gamma
        self.gamma_head = nn.Sequential(
            nn.Linear(feat_dim, 32),
            nn.GELU(),
            nn.Linear(32, 3),
        )

        # 初始化为恒等变换
        nn.init.zeros_(self.color_matrix_head[-1].weight)
        nn.init.zeros_(self.color_matrix_head[-1].bias)
        # bias 初始为 identity matrix flattened + zero bias
        with torch.no_grad():
            self.color_matrix_head[-1].bias[:9] = torch.tensor(
                [1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=torch.float32
            )

        nn.init.zeros_(self.gamma_head[-1].weight)
        nn.init.ones_(self.gamma_head[-1].bias)  # gamma=1 → identity

    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) 输入图片
        Returns:
            (B, 3, H, W) 全局色彩变换后的图片
        """
        feat = self.encoder(x)

        # 色彩矩阵
        cm_params = self.color_matrix_head(feat)
        matrix = cm_params[:, :9].view(-1, 3, 3)  # (B, 3, 3)
        bias = cm_params[:, 9:].view(-1, 3, 1, 1)  # (B, 3, 1, 1)

        # 应用色彩矩阵: (B, 3, H, W) → reshape → matmul
        B, C, H, W = x.shape
        x_flat = x.view(B, 3, -1)  # (B, 3, H*W)
        color_out = torch.bmm(matrix, x_flat).view(B, 3, H, W) + bias

        # Gamma 曲线
        gamma = self.gamma_head(feat).view(-1, 3, 1, 1)
        gamma = gamma.clamp(0.2, 5.0)  # 安全范围
        color_out = color_out.clamp(1e-6, 1.0) ** gamma

        return color_out.clamp(0.0, 1.0)


class LocalDetailBranch(nn.Module):
    """局部细节分支: 3 级 U-Net 产生空间细节残差。"""

    def __init__(self, base_ch=64):
        super().__init__()
        # 编码器
        self.stem = nn.Sequential(
            nn.Conv2d(3, base_ch, 3, padding=1, bias=False),
            nn.InstanceNorm2d(base_ch),
            nn.GELU(),
        )
        self.enc1 = EncoderBlock(base_ch, base_ch * 2)      # /2
        self.enc2 = EncoderBlock(base_ch * 2, base_ch * 4)   # /4
        self.enc3 = EncoderBlock(base_ch * 4, base_ch * 8)   # /8

        # 瓶颈
        self.bottleneck = nn.Sequential(
            ResBlock(base_ch * 8),
            ResBlock(base_ch * 8),
            ResBlock(base_ch * 8),
        )

        # 解码器
        self.dec3 = DecoderBlock(base_ch * 8, base_ch * 4, base_ch * 4)
        self.dec2 = DecoderBlock(base_ch * 4, base_ch * 2, base_ch * 2)
        self.dec1 = DecoderBlock(base_ch * 2, base_ch, base_ch)

        # 输出: 空间残差
        self.out_conv = nn.Sequential(
            nn.Conv2d(base_ch, base_ch // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(base_ch // 2, 3, 1),
            nn.Tanh(),
        )

        # 残差强度 (初始为 0, 逐步学习)
        self.residual_scale = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) 输入图片
        Returns:
            (B, 3, H, W) 空间细节残差 (加到全局色彩分支输出上)
        """
        s0 = self.stem(x)       # (B, base_ch, H, W)
        s1 = self.enc1(s0)      # (B, base_ch*2, H/2, W/2)
        s2 = self.enc2(s1)      # (B, base_ch*4, H/4, W/4)
        s3 = self.enc3(s2)      # (B, base_ch*8, H/8, W/8)

        b = self.bottleneck(s3)

        d3 = self.dec3(b, s2)
        d2 = self.dec2(d3, s1)
        d1 = self.dec1(d2, s0)

        residual = self.out_conv(d1)  # (B, 3, H, W), [-1, 1]

        # 缩放残差
        scale = torch.sigmoid(self.residual_scale) * 0.3  # max ±0.3
        return residual * scale


class RefinementNetV4(nn.Module):
    """双分支图像精修网络。

    Pipeline:
        input → GlobalColorBranch → 全局色彩调整
             → LocalDetailBranch → 空间细节残差
        output = GlobalColor + LocalDetail

    Args:
        base_ch: 基础通道数 (影响模型大小)
    """

    def __init__(self, base_ch=64):
        super().__init__()
        self.global_branch = GlobalColorBranch(base_ch)
        self.local_branch = LocalDetailBranch(base_ch)

    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) diff_isp 粗渲染图 [0, 1]
        Returns:
            (B, 3, H, W) 精修后图片 [0, 1]
        """
        global_out = self.global_branch(x)
        local_residual = self.local_branch(x)
        output = (global_out + local_residual).clamp(0.0, 1.0)
        return output

    def count_params(self):
        """返回 (总参数量, 可训练参数量)。"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


if __name__ == '__main__':
    model = RefinementNetV4(base_ch=64)
    x = torch.randn(2, 3, 512, 512).sigmoid()
    out = model(x)
    print(f"Input: {x.shape} → Output: {out.shape}")
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params / 1e6:.2f}M")

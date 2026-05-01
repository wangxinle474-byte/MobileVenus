"""MobileViT-Small 视觉编码器，含 SE 注意力和 FPN 多尺度融合。

基于 Apple MobileViT (ICLR 2022) 重写，增加:
- 每个 stage 的 Squeeze-and-Excitation (SE) 通道注意力
- Feature Pyramid Network (FPN) 多尺度特征融合
- 支持返回多尺度中间特征 (用于蒸馏)

输入: (B, 3, 224, 224)
输出: (B, 384) 全局特征向量
参数量: ~5.6M
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ---------------------------------------------------------------------------
# 基础模块
# ---------------------------------------------------------------------------

class ConvBnAct(nn.Module):
    """Conv2d + BatchNorm + 激活函数。"""

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, groups=1,
                 act=nn.SiLU):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding,
                              groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = act(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class SqueezeExcitation(nn.Module):
    """SE 通道注意力模块。"""

    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid),
            nn.SiLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


# ---------------------------------------------------------------------------
# MobileNetV2 Inverted Residual Block
# ---------------------------------------------------------------------------

class MV2Block(nn.Module):
    """MobileNetV2 倒残差模块 (expand → depthwise → project)。"""

    def __init__(self, in_ch, out_ch, stride=1, expansion=4):
        super().__init__()
        mid = in_ch * expansion
        self.use_residual = (stride == 1 and in_ch == out_ch)

        layers = []
        # Expand
        if expansion != 1:
            layers.append(ConvBnAct(in_ch, mid, 1))
        # Depthwise
        layers.append(ConvBnAct(mid, mid, 3, stride=stride, groups=mid))
        # Project (no activation)
        layers.append(nn.Conv2d(mid, out_ch, 1, bias=False))
        layers.append(nn.BatchNorm2d(out_ch))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.block(x)
        if self.use_residual:
            out = out + x
        return out


# ---------------------------------------------------------------------------
# MobileViT Transformer Block
# ---------------------------------------------------------------------------

class MultiHeadSelfAttention(nn.Module):
    """多头自注意力。"""

    def __init__(self, dim, num_heads=1, qkv_bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


class TransformerBlock(nn.Module):
    """Transformer 编码器块: MHSA + FFN。"""

    def __init__(self, dim, num_heads=1, mlp_ratio=2.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.SiLU(inplace=True),
            nn.Linear(mlp_hidden, dim),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class MobileViTBlock(nn.Module):
    """MobileViT 块: 局部特征 → patch 化 → Transformer → 恢复空间。

    Args:
        in_ch: 输入通道数
        out_ch: 输出通道数 (也是 transformer dim)
        transformer_dim: Transformer 内部维度
        num_heads: 注意力头数
        num_layers: Transformer 层数
        patch_size: patch 大小 (h, w)
    """

    def __init__(self, in_ch, out_ch, transformer_dim, num_heads=1,
                 num_layers=2, patch_size=(2, 2)):
        super().__init__()
        self.patch_h, self.patch_w = patch_size

        # 局部特征提取
        self.local_rep = nn.Sequential(
            ConvBnAct(in_ch, in_ch, 3),
            ConvBnAct(in_ch, transformer_dim, 1),
        )

        # Transformer
        self.transformer = nn.Sequential(
            *[TransformerBlock(transformer_dim, num_heads) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(transformer_dim)

        # 投影回通道维度
        self.proj = ConvBnAct(transformer_dim, out_ch, 1)

        # 融合局部和全局
        self.fusion = ConvBnAct(out_ch + in_ch, out_ch, 1)

    def forward(self, x):
        _, _, H, W = x.shape
        ph, pw = self.patch_h, self.patch_w

        # 局部特征
        local_feat = self.local_rep(x)
        C = local_feat.shape[1]

        # Unfold: (B, C, H, W) → (B * num_patches, patch_size, C)
        num_patches_h = H // ph
        num_patches_w = W // pw
        num_patches = num_patches_h * num_patches_w

        # 重排为 patches
        patches = rearrange(local_feat,
                            'b c (nh ph) (nw pw) -> (b nh nw) (ph pw) c',
                            ph=ph, pw=pw)

        # Transformer
        patches = self.transformer(patches)
        patches = self.norm(patches)

        # Fold back
        global_feat = rearrange(patches,
                                '(b nh nw) (ph pw) c -> b c (nh ph) (nw pw)',
                                nh=num_patches_h, nw=num_patches_w,
                                ph=ph, pw=pw)

        global_feat = self.proj(global_feat)

        # 融合局部 + 全局
        out = self.fusion(torch.cat([x, global_feat], dim=1))
        return out


# ---------------------------------------------------------------------------
# Feature Pyramid Network (FPN) 融合
# ---------------------------------------------------------------------------

class FeaturePyramidFusion(nn.Module):
    """轻量 FPN: 将多尺度特征融合为统一维度。"""

    def __init__(self, in_channels_list, out_dim=384):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, out_dim, 1, bias=False),
                nn.BatchNorm2d(out_dim),
            )
            for in_ch in in_channels_list
        ])
        self.fusion_conv = ConvBnAct(out_dim, out_dim, 3)

    def forward(self, features):
        """
        Args:
            features: list of feature maps from different stages,
                      from high-res to low-res
        Returns:
            fused: (B, out_dim, H_last, W_last) 融合特征
        """
        # 投影到统一通道
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]

        # 自顶向下融合 (低分辨率 → 高分辨率)
        target_size = laterals[-1].shape[2:]
        fused = laterals[-1]
        for i in range(len(laterals) - 2, -1, -1):
            resized = F.adaptive_avg_pool2d(laterals[i], target_size)
            fused = fused + resized

        fused = self.fusion_conv(fused)
        return fused


# ---------------------------------------------------------------------------
# MobileViT-Small 主模型
# ---------------------------------------------------------------------------

class MobileViTSmall(nn.Module):
    """MobileViT-Small 视觉编码器。

    架构:
        Conv Stem (3→16, stride=2)
        → Stage 2: MV2(16→32) + SE
        → Stage 3: MV2(32→48) + MobileViT(48→64, T=96, 2层) + SE
        → Stage 4: MV2(64→64) + MobileViT(64→80, T=120, 4层) + SE
        → Stage 5: MV2(80→80) + MobileViT(80→96, T=144, 3层) + SE
        → FPN([64, 80, 96] → 384)
        → GlobalAvgPool → (B, 384)

    Args:
        image_size: 输入图片大小 (默认 224)
        num_classes: 分类头输出维度 (0 = 不加分类头)
        use_se: 是否使用 SE 注意力
        use_fpn: 是否使用 FPN 融合
        output_dim: 输出特征维度
    """

    def __init__(self, image_size=224, num_classes=0, use_se=True,
                 use_fpn=True, output_dim=384):
        super().__init__()
        self.use_se = use_se
        self.use_fpn = use_fpn
        self.output_dim = output_dim

        # Conv Stem
        self.conv_stem = nn.Sequential(
            ConvBnAct(3, 16, 3, stride=2),
        )

        # Stage 2: MV2 blocks (16 → 32), stride=2
        self.stage2 = nn.Sequential(
            MV2Block(16, 32, stride=2, expansion=4),
            MV2Block(32, 32, stride=1, expansion=4),
            MV2Block(32, 32, stride=1, expansion=4),
        )
        self.se2 = SqueezeExcitation(32) if use_se else nn.Identity()

        # Stage 3: MV2 → MobileViT (32 → 64), stride=2
        self.stage3_mv2 = MV2Block(32, 48, stride=2, expansion=4)
        self.stage3_vit = MobileViTBlock(
            in_ch=48, out_ch=64, transformer_dim=96,
            num_heads=1, num_layers=2, patch_size=(2, 2)
        )
        self.se3 = SqueezeExcitation(64) if use_se else nn.Identity()

        # Stage 4: MV2 → MobileViT (64 → 80), stride=2
        self.stage4_mv2 = MV2Block(64, 64, stride=2, expansion=4)
        self.stage4_vit = MobileViTBlock(
            in_ch=64, out_ch=80, transformer_dim=120,
            num_heads=2, num_layers=4, patch_size=(2, 2)
        )
        self.se4 = SqueezeExcitation(80) if use_se else nn.Identity()

        # Stage 5: MV2 → MobileViT (80 → 96), stride=2
        self.stage5_mv2 = MV2Block(80, 80, stride=2, expansion=4)
        self.stage5_vit = MobileViTBlock(
            in_ch=80, out_ch=96, transformer_dim=144,
            num_heads=3, num_layers=3, patch_size=(2, 2)
        )
        self.se5 = SqueezeExcitation(96) if use_se else nn.Identity()

        # FPN 多尺度融合
        if use_fpn:
            self.fpn = FeaturePyramidFusion(
                in_channels_list=[64, 80, 96],
                out_dim=output_dim,
            )
        else:
            self.final_conv = ConvBnAct(96, output_dim, 1)

        # 全局池化
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # 可选分类头
        self.classifier = nn.Linear(output_dim, num_classes) if num_classes > 0 else None

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward_features(self, x):
        """提取多尺度特征。"""
        x = self.conv_stem(x)          # (B, 16, 112, 112)

        # Stage 2
        x = self.stage2(x)             # (B, 32, 56, 56)
        x = self.se2(x)

        # Stage 3
        x = self.stage3_mv2(x)         # (B, 48, 28, 28)
        x = self.stage3_vit(x)         # (B, 64, 28, 28)
        f3 = self.se3(x)

        # Stage 4
        x = self.stage4_mv2(f3)        # (B, 64, 14, 14)
        x = self.stage4_vit(x)         # (B, 80, 14, 14)
        f4 = self.se4(x)

        # Stage 5
        x = self.stage5_mv2(f4)        # (B, 80, 7, 7)
        x = self.stage5_vit(x)         # (B, 96, 7, 7)
        f5 = self.se5(x)

        return f3, f4, f5

    def forward(self, x, return_multi_scale=False, return_spatial=False):
        """
        Args:
            x: (B, 3, 224, 224) 输入图片
            return_multi_scale: 是否返回多尺度特征 dict
            return_spatial: 是否返回空间特征 (不做 global pool)
        Returns:
            默认: (B, 384) 全局特征向量
        """
        f3, f4, f5 = self.forward_features(x)

        if self.use_fpn:
            feat = self.fpn([f3, f4, f5])  # (B, 384, 7, 7)
        else:
            feat = self.final_conv(f5)     # (B, 384, 7, 7)

        if return_spatial:
            return feat

        if return_multi_scale:
            return {
                'stage3': f3,
                'stage4': f4,
                'stage5': f5,
                'fused': feat,
                'global': self.global_pool(feat).flatten(1),
            }

        global_feat = self.global_pool(feat).flatten(1)  # (B, 384)

        if self.classifier is not None:
            return self.classifier(global_feat)

        return global_feat


def mobilevit_small(**kwargs):
    """便捷构造函数。"""
    return MobileViTSmall(**kwargs)


if __name__ == '__main__':
    model = MobileViTSmall(image_size=224, output_dim=384)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print(f"Output shape: {out.shape}")  # (2, 384)
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params / 1e6:.2f}M")

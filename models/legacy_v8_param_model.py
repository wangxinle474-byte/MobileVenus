from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class ConvBNAct(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )


class InvertedResidual(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, expansion=4):
        super().__init__()
        mid = in_ch * expansion
        self.use_residual = stride == 1 and in_ch == out_ch
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid, mid, 3, stride=stride, padding=1, groups=mid, bias=False),
            nn.BatchNorm2d(mid),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x):
        out = self.conv(x)
        if self.use_residual:
            out = out + x
        return out


class SqueezeExcitation(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.pool(x).flatten(1)
        w = self.excitation(w).view(b, c, 1, 1)
        return x * w


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(0.0),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x):
        y = self.norm1(x)
        y, _ = self.attn(y, y, y, need_weights=False)
        x = x + y
        x = x + self.mlp(self.norm2(x))
        return x


class LegacyMobileViTBlock(nn.Module):
    def __init__(self, in_ch, out_ch, transformer_dim, num_heads, num_layers, patch_size=(2, 2)):
        super().__init__()
        self.patch_h, self.patch_w = patch_size
        self.local_rep = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_ch, transformer_dim, 1, bias=False),
        )
        self.global_rep = nn.Sequential(*[TransformerBlock(transformer_dim, num_heads) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(transformer_dim)
        self.fusion = nn.Sequential(
            nn.Conv2d(transformer_dim, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )
        self.residual = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x):
        y = self.local_rep(x)
        b, c, h, w = y.shape
        ph, pw = self.patch_h, self.patch_w
        pad_h = (ph - h % ph) % ph
        pad_w = (pw - w % pw) % pw
        if pad_h or pad_w:
            y = F.pad(y, (0, pad_w, 0, pad_h), mode='replicate')
        hp, wp = y.shape[-2:]
        patches = rearrange(y, 'b c (nh ph) (nw pw) -> b (nh nw) (ph pw) c', ph=ph, pw=pw)
        patches = rearrange(patches, 'b n p c -> (b n) p c')
        patches = self.global_rep(patches)
        patches = self.norm(patches)
        y = rearrange(patches, '(b nh nw) (ph pw) c -> b c (nh ph) (nw pw)', b=b, nh=hp // ph, nw=wp // pw, ph=ph, pw=pw)
        y = y[:, :, :h, :w]
        return self.fusion(y) + self.residual(x)


class FeaturePyramidFusion(nn.Module):
    def __init__(self, in_channels_list, out_dim=384):
        super().__init__()
        self.scale_weights = nn.Parameter(torch.ones(len(in_channels_list)))
        self.lateral_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, out_dim, 1, bias=False),
                nn.BatchNorm2d(out_dim),
            )
            for in_ch in in_channels_list
        ])
        self.channel_attention = SqueezeExcitation(out_dim)

    def forward(self, features):
        target_size = features[-1].shape[-2:]
        weights = torch.softmax(self.scale_weights, dim=0)
        fused = 0.0
        for weight, conv, feat in zip(weights, self.lateral_convs, features):
            lat = conv(feat)
            if lat.shape[-2:] != target_size:
                lat = F.adaptive_avg_pool2d(lat, target_size)
            fused = fused + weight * lat
        return self.channel_attention(fused)


class LegacyMobileViTSmall(nn.Module):
    def __init__(self, image_size=224, num_classes=0, use_se=True, use_fpn=True, output_dim=384):
        super().__init__()
        self.use_se = use_se
        self.use_fpn = use_fpn
        self.output_dim = output_dim
        self.conv_stem = ConvBNAct(3, 16, 3, stride=2)
        self.stage2 = InvertedResidual(16, 32, stride=2, expansion=4)
        self.se2 = SqueezeExcitation(32) if use_se else nn.Identity()
        self.stage3 = nn.Sequential(
            InvertedResidual(32, 48, stride=2, expansion=4),
            LegacyMobileViTBlock(48, 64, transformer_dim=96, num_heads=4, num_layers=2),
        )
        self.se3 = SqueezeExcitation(64) if use_se else nn.Identity()
        self.stage4 = nn.Sequential(
            InvertedResidual(64, 64, stride=2, expansion=4),
            LegacyMobileViTBlock(64, 80, transformer_dim=120, num_heads=4, num_layers=4),
        )
        self.se4 = SqueezeExcitation(80) if use_se else nn.Identity()
        self.stage5 = nn.Sequential(
            InvertedResidual(80, 80, stride=2, expansion=4),
            LegacyMobileViTBlock(80, 96, transformer_dim=144, num_heads=4, num_layers=3),
        )
        self.se5 = SqueezeExcitation(96) if use_se else nn.Identity()
        self.fpn = FeaturePyramidFusion([64, 80, 96], out_dim=output_dim)
        self.final_conv = ConvBNAct(96, output_dim, 1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(output_dim, num_classes) if num_classes > 0 else None

    def forward_features(self, x):
        x = self.conv_stem(x)
        x = self.se2(self.stage2(x))
        f3 = self.se3(self.stage3(x))
        f4 = self.se4(self.stage4(f3))
        f5 = self.se5(self.stage5(f4))
        return f3, f4, f5

    def forward(self, x, return_multi_scale=False, return_spatial=False):
        f3, f4, f5 = self.forward_features(x)
        feat = self.fpn([f3, f4, f5]) if self.use_fpn else self.final_conv(f5)
        if return_spatial:
            return feat
        pooled = self.global_pool(feat).flatten(1)
        if return_multi_scale:
            return {'stage3': f3, 'stage4': f4, 'stage5': f5, 'fused': feat, 'global': pooled}
        if self.classifier is not None:
            return self.classifier(pooled)
        return pooled


class LegacySemanticProjector(nn.Module):
    def __init__(self, visual_dim=384, semantic_dim=256, hidden_dim=None, dropout=0.1):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = (visual_dim + semantic_dim) // 2
        self.projector = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, semantic_dim),
            nn.LayerNorm(semantic_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, visual_feat):
        return F.normalize(self.projector(visual_feat), p=2, dim=-1)


class LegacyLightroomDecoder(nn.Module):
    def __init__(self, semantic_dim=256, hidden_dim=256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(semantic_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.ev_head = nn.Sequential(nn.Linear(hidden_dim, 64), nn.GELU(), nn.Linear(64, 1))
        self.wb_head = nn.Sequential(nn.Linear(hidden_dim, 64), nn.GELU(), nn.Linear(64, 1))
        self.range100_heads = nn.ModuleDict({
            'contrast': nn.Sequential(nn.Linear(hidden_dim, 64), nn.GELU(), nn.Linear(64, 1)),
            'shadows': nn.Sequential(nn.Linear(hidden_dim, 64), nn.GELU(), nn.Linear(64, 1)),
            'highlights': nn.Sequential(nn.Linear(hidden_dim, 64), nn.GELU(), nn.Linear(64, 1)),
            'saturation': nn.Sequential(nn.Linear(hidden_dim, 64), nn.GELU(), nn.Linear(64, 1)),
        })
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, 6),
            nn.Sigmoid(),
        )

    def forward(self, semantic_emb):
        h = self.shared(semantic_emb)
        ev = torch.tanh(self.ev_head(h))
        wb = torch.tanh(self.wb_head(h))
        contrast = torch.tanh(self.range100_heads['contrast'](h))
        shadows = torch.tanh(self.range100_heads['shadows'](h))
        highlights = torch.tanh(self.range100_heads['highlights'](h))
        saturation = torch.tanh(self.range100_heads['saturation'](h))
        norm_params = torch.cat([ev, wb, contrast, shadows, highlights, saturation], dim=-1)
        raw_params = {
            'ev_compensation': ev * 3.0,
            'white_balance': (wb + 1.0) * 4000.0 + 2000.0,
            'contrast': contrast * 100.0,
            'shadows': shadows * 100.0,
            'highlights': highlights * 100.0,
            'saturation': saturation * 100.0,
        }
        return {'raw_params': raw_params, 'norm_params': norm_params, 'confidence': self.confidence_head(h)}


class LegacyDistillParamModel(nn.Module):
    def __init__(self, image_size=224, visual_dim=384, semantic_dim=256, decoder_hidden=256):
        super().__init__()
        self.vision_encoder = LegacyMobileViTSmall(image_size=image_size, output_dim=visual_dim)
        self.semantic_projector = LegacySemanticProjector(visual_dim=visual_dim, semantic_dim=semantic_dim)
        self.decoder = LegacyLightroomDecoder(semantic_dim=semantic_dim, hidden_dim=decoder_hidden)

    def forward(self, images, return_embedding=False):
        visual_feat = self.vision_encoder(images)
        semantic_emb = self.semantic_projector(visual_feat)
        outputs = self.decoder(semantic_emb)
        if return_embedding:
            outputs['semantic_emb'] = semantic_emb
        return outputs

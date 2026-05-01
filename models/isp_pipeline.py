"""Gamma-aware Lightroom ISP 渲染管线。

非可微版本，用于推理和评估。可微版本见 diff_isp.py。

Pipeline:
    sRGB → Linear → WB → EV → Contrast → Shadows/Highlights → Saturation → sRGB
"""

import torch
import torch.nn.functional as F
import numpy as np


def srgb_to_linear(x):
    """sRGB → Linear (gamma 解码)。"""
    x = x.clamp(0.0, 1.0)
    linear = torch.where(
        x <= 0.04045,
        x / 12.92,
        ((x + 0.055) / 1.055) ** 2.4,
    )
    return linear.clamp(0.0, 1.0)


def linear_to_srgb(x):
    """Linear → sRGB (gamma 编码)。"""
    x = x.clamp(0.0, 1.0)
    srgb = torch.where(
        x <= 0.0031308,
        x * 12.92,
        1.055 * x ** (1.0 / 2.4) - 0.055,
    )
    return srgb.clamp(0.0, 1.0)


def apply_white_balance(img_linear, temperature_k):
    """应用白平衡 (色温调整)。

    Args:
        img_linear: (B, 3, H, W) Linear 空间图片
        temperature_k: (B, 1) 色温 (Kelvin), 6500K 为中性
    Returns:
        (B, 3, H, W) 白平衡后的图片
    """
    # 简化的色温 → RGB gains 映射
    # 基于 Planckian locus 近似
    t = temperature_k.view(-1, 1, 1, 1) / 6500.0  # 归一化到 D65

    r_gain = 1.0 / t.clamp(0.5, 2.0)
    b_gain = t.clamp(0.5, 2.0)
    g_gain = torch.ones_like(r_gain)

    gains = torch.cat([r_gain, g_gain, b_gain], dim=1)  # (B, 3, 1, 1)
    result = img_linear * gains
    return result.clamp(0.0, 1.0)


def apply_ev(img_linear, ev_compensation):
    """应用曝光补偿。

    Args:
        img_linear: (B, 3, H, W) Linear 空间
        ev_compensation: (B, 1) EV 补偿值 [-3, +3]
    """
    factor = (2.0 ** ev_compensation).view(-1, 1, 1, 1)
    return (img_linear * factor).clamp(0.0, 1.0)


def apply_contrast(img_srgb, contrast):
    """应用对比度调整。

    Args:
        img_srgb: (B, 3, H, W) sRGB 空间
        contrast: (B, 1) 对比度 [-100, 100]
    """
    factor = (contrast / 100.0 + 1.0).view(-1, 1, 1, 1)
    mid = 0.5
    result = mid + (img_srgb - mid) * factor
    return result.clamp(0.0, 1.0)


def apply_shadows_highlights(img_srgb, shadows, highlights):
    """应用阴影/高光调整。

    Args:
        img_srgb: (B, 3, H, W) sRGB 空间
        shadows: (B, 1) 阴影 [-100, 100]
        highlights: (B, 1) 高光 [-100, 100]
    """
    # 计算亮度
    lum = 0.299 * img_srgb[:, 0:1] + 0.587 * img_srgb[:, 1:2] + \
          0.114 * img_srgb[:, 2:3]

    # 平滑 sigmoid mask
    shadow_mask = torch.sigmoid((0.3 - lum) * 10.0)    # 暗区
    highlight_mask = torch.sigmoid((lum - 0.7) * 10.0)  # 亮区

    shadow_adj = (shadows / 100.0).view(-1, 1, 1, 1) * shadow_mask * 0.3
    highlight_adj = (highlights / 100.0).view(-1, 1, 1, 1) * highlight_mask * 0.3

    # 保持色彩比例
    ratio = img_srgb / (lum + 1e-6)
    new_lum = (lum + shadow_adj + highlight_adj).clamp(0.0, 1.0)
    result = ratio * new_lum

    return result.clamp(0.0, 1.0)


def apply_saturation(img_srgb, saturation):
    """应用饱和度调整。

    Args:
        img_srgb: (B, 3, H, W) sRGB 空间
        saturation: (B, 1) 饱和度 [-100, 100]
    """
    factor = (saturation / 100.0 + 1.0).view(-1, 1, 1, 1)
    gray = 0.299 * img_srgb[:, 0:1] + 0.587 * img_srgb[:, 1:2] + \
           0.114 * img_srgb[:, 2:3]
    result = gray + (img_srgb - gray) * factor
    return result.clamp(0.0, 1.0)


def render_params(img_srgb, params):
    """完整 ISP 渲染: 应用 6 个参数。

    Args:
        img_srgb: (B, 3, H, W) sRGB 输入图片
        params: dict with keys: ev_compensation, white_balance,
                contrast, shadows, highlights, saturation
    Returns:
        (B, 3, H, W) 渲染后的 sRGB 图片
    """
    # sRGB → Linear
    img = srgb_to_linear(img_srgb)

    # 白平衡
    img = apply_white_balance(img, params['white_balance'])

    # 曝光
    img = apply_ev(img, params['ev_compensation'])

    # Linear → sRGB
    img = linear_to_srgb(img)

    # 对比度
    img = apply_contrast(img, params['contrast'])

    # 阴影/高光
    img = apply_shadows_highlights(img, params['shadows'], params['highlights'])

    # 饱和度
    img = apply_saturation(img, params['saturation'])

    return img

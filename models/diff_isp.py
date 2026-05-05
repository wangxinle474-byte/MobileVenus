"""
可微分 Lightroom-style ISP v2 — 基于 PyTorch，支持反向传播

增强版:
  - Tone Curve: 参数化三次 S 曲线，保留中间调细节
  - Clarity:   局部对比度增强 (unsharp mask on luminance)
  - Shadows/Highlights: 平滑 sigmoid mask + 色彩比率保持

Input:  (B, 3, H, W)  [0, 1] sRGB tensor
Output: (B, 3, H, W)  [0, 1] sRGB tensor
"""
import torch
import torch.nn.functional as F
from typing import Dict


# ─────────────────────────── 色彩空间转换 ───────────────────────────

def srgb_to_linear(img: torch.Tensor) -> torch.Tensor:
    """sRGB gamma → linear light.  img: (B,C,H,W) in [0,1]"""
    img = img.clamp(0.0, 1.0)
    return torch.where(
        img <= 0.04045,
        img / 12.92,
        ((img.clamp(min=0.04045) + 0.055) / 1.055) ** 2.4,
    )


def linear_to_srgb(img: torch.Tensor) -> torch.Tensor:
    """linear light → sRGB gamma.  img: (B,C,H,W)"""
    img = img.clamp(0.0, 1.0)
    return torch.where(
        img <= 0.0031308,
        img * 12.92,
        1.055 * img.clamp(min=0.0031308) ** (1.0 / 2.4) - 0.055,
    )


def rgb_to_luminance(img: torch.Tensor) -> torch.Tensor:
    """Rec. 709 luminance.  img: (B,C,H,W) → (B,1,H,W)"""
    return (0.2126 * img[:, 0:1]
            + 0.7152 * img[:, 1:2]
            + 0.0722 * img[:, 2:3])


# ─────────────────────────── 色温 → RGB 增益 ───────────────────────────

def color_temp_to_rgb_gains(wb_k: torch.Tensor) -> torch.Tensor:
    """
    白平衡色温 → RGB 增益 (可微版本，基于 Planckian locus 近似)
    wb_k: (B,) Kelvin 值
    Returns: (B, 3) gains，绿色通道归一化为 1
    """
    t = wb_k / 100.0  # 归一化到 [20, 100] 区间

    # ── Red ──────────────────────────────────────
    r_low  = torch.full_like(t, 255.0)
    r_high = 329.698727446 * (t - 60).clamp(min=1e-3) ** (-0.1332047592)
    r = torch.where(t <= 66.0, r_low, r_high)

    # ── Green ──────────────────────────────────────
    g_low  = 99.4708025861  * torch.log(t.clamp(min=1.0)) - 161.1195681661
    g_high = 288.1221695283 * (t - 60).clamp(min=1e-3) ** (-0.0755148492)
    g = torch.where(t <= 66.0, g_low, g_high)

    # ── Blue ──────────────────────────────────────
    b_high = torch.full_like(t, 255.0)
    b_zero = torch.zeros_like(t)
    b_mid  = 138.5177312231 * torch.log((t - 10).clamp(min=1e-3)) - 305.0447927307
    b = torch.where(t >= 66.0, b_high,
        torch.where(t <= 19.0, b_zero, b_mid))

    rgb = torch.stack([r, g, b], dim=1).clamp(0.0, 255.0) / 255.0  # (B, 3)

    # 参考色温 5500K 的增益 (ref_t=55)
    ref_t = 55.0
    ref_r = 255.0
    ref_g = 99.4708025861 * torch.log(torch.tensor(ref_t)) - 161.1195681661
    ref_b = 138.5177312231 * torch.log(torch.tensor(ref_t - 10.0)) - 305.0447927307
    ref = torch.tensor([ref_r, ref_g.item(), ref_b.item()],
                       device=wb_k.device, dtype=wb_k.dtype).clamp(0, 255) / 255.0

    gains = ref.unsqueeze(0) / (rgb + 1e-8)   # (B, 3)
    gains = gains / gains[:, 1:2]              # 绿色通道归一化为 1
    return gains


# ─────────────────────────── SSIM 损失 ───────────────────────────

def ssim_loss(pred: torch.Tensor, target: torch.Tensor,
              window_size: int = 11) -> torch.Tensor:
    """
    可微 SSIM 损失 = 1 - SSIM.
    pred, target: (B, C, H, W) in [0, 1]
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    C = pred.shape[1]
    pad = window_size // 2

    kernel = torch.ones(C, 1, window_size, window_size,
                        device=pred.device, dtype=pred.dtype) / (window_size ** 2)

    mu1 = F.conv2d(pred,   kernel, padding=pad, groups=C)
    mu2 = F.conv2d(target, kernel, padding=pad, groups=C)

    mu1_sq  = mu1 ** 2
    mu2_sq  = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sig1_sq = F.conv2d(pred   ** 2, kernel, padding=pad, groups=C) - mu1_sq
    sig2_sq = F.conv2d(target ** 2, kernel, padding=pad, groups=C) - mu2_sq
    sig12   = F.conv2d(pred * target, kernel, padding=pad, groups=C) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sig12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sig1_sq + sig2_sq + C2) + 1e-8)

    return 1.0 - ssim_map.mean()


# ─────────────────── Tone Curve ───────────────────

def _tone_curve(x: torch.Tensor, strength: torch.Tensor) -> torch.Tensor:
    """
    参数化三次 S 曲线，保留中间调细节。
    x: (B,C,H,W) in [0,1] 感知空间
    strength: (B,1,1,1) 对比度强度，[-1, 1] 约对应 contrast/100

    曲线: y = x + s * x * (1-x) * (2x-1)
    s=0  → identity
    s>0  → S-curve (增强对比)
    s<0  → inverse-S (减弱对比)
    始终保证 y(0)=0, y(0.5)=0.5, y(1)=1
    """
    s = strength.clamp(-0.99, 0.99)
    y = x + s * x * (1.0 - x) * (2.0 * x - 1.0)
    return y.clamp(1e-4, 1.0)


# ─────────────────── Clarity (局部对比度) ───────────────────

def _clarity(img: torch.Tensor, lum: torch.Tensor,
            strength, kernel_size: int = 15) -> torch.Tensor:
    """
    局部对比度增强/减弱: unsharp mask on luminance, 应用到 RGB。
    img:       (B,3,H,W) linear
    lum:       (B,1,H,W) luminance
    strength:  scalar 或 (B,1,1,1) tensor, 可正可负
               +: 增强细节 (锐化); -: 柔化; 0: 不变
    """
    if kernel_size < 3:
        return img
    pad = kernel_size // 2
    k = torch.ones(1, 1, kernel_size, kernel_size,
                   device=lum.device, dtype=lum.dtype) / (kernel_size ** 2)
    lum_blur = F.conv2d(lum, k, padding=pad)
    detail = lum - lum_blur  # 局部细节
    boost = 1.0 + strength * detail / (lum.abs() + 0.05)
    return (img * boost).clamp(0.0, 4.0)


# ───────────── 平滑 Shadows / Highlights mask ─────────────

def _smooth_shadow_mask(lum: torch.Tensor,
                       center: float = 0.15, width: float = 0.12) -> torch.Tensor:
    """平滑 sigmoid 暗部 mask: 1 at dark, 0 at bright, 平滑过渡"""
    return torch.sigmoid(-(lum - center) / width)


def _smooth_highlight_mask(lum: torch.Tensor,
                          center: float = 0.65, width: float = 0.15) -> torch.Tensor:
    """平滑 sigmoid 亮部 mask: 0 at dark, 1 at bright, 平滑过渡"""
    return torch.sigmoid((lum - center) / width)


# ─────────────────────────── 主 ISP 函数 ───────────────────────────

def apply_diff_isp(img: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    可微 ISP v2：将 Lightroom 参数应用到图像上（增强渲染）。

    Args:
        img:    (B, 3, H, W)  [0, 1] sRGB tensor
        params: dict  param_name → (B,) 真实物理值（未归一化）
                键: ev_compensation, white_balance, contrast,
                    shadows, highlights, saturation
                    (brightness, vibrance 可选，缺省为0)

    Returns:
        (B, 3, H, W)  [0, 1] sRGB tensor
    """
    B = img.shape[0]
    linear = srgb_to_linear(img)

    # 1. White Balance
    wb = params['white_balance']
    gains = color_temp_to_rgb_gains(wb).clamp(0.0, 8.0)
    linear = (linear * gains.view(B, 3, 1, 1)).clamp(0.0, 4.0)

    # 2. EV Compensation (可选, 新 pipeline 建议用 brightness 代替; 此 op 保留做向后兼容)
    ev = params.get('ev_compensation', torch.zeros(B, device=img.device)).view(B, 1, 1, 1)
    linear = (linear * (2.0 ** ev)).clamp(0.0, 4.0)

    # 3. Brightness (gamma shift, optional)
    brightness = params.get('brightness', torch.zeros(B, device=img.device))
    gamma_shift = (1.0 - brightness / 200.0).clamp(0.25, 4.0).view(B, 1, 1, 1)
    linear = linear.clamp(min=1e-3) ** gamma_shift

    # 4. Tone Curve + Contrast (参数化 S 曲线，代替简单对比度)
    contrast = params['contrast']
    strength = (contrast / 100.0).view(B, 1, 1, 1)
    perceptual = linear.clamp(1e-3, 1.0) ** (1.0 / 2.2)
    perceptual = _tone_curve(perceptual, strength)
    linear = perceptual ** 2.2

    # 5. Shadows (暗部提亮 — 平滑 sigmoid mask + 色彩比率保持)
    shadows = params['shadows']
    lum = rgb_to_luminance(linear.clamp(0, 1))
    shadow_mask = _smooth_shadow_mask(lum)
    lift = (shadows / 100.0 * 0.2).view(B, 1, 1, 1)
    ratio = linear / (lum + 1e-6)
    lum_lifted = lum + shadow_mask * lift
    linear = ratio * lum_lifted

    # 6. Highlights (亮部压暗 — 平滑 sigmoid mask + 色彩比率保持)
    highlights = params['highlights']
    lum = rgb_to_luminance(linear.clamp(0, 1))
    hi_mask = _smooth_highlight_mask(lum)
    pull = (-highlights / 100.0 * 0.25).view(B, 1, 1, 1)
    ratio = linear / (lum + 1e-6)
    lum_pulled = lum + hi_mask * pull
    linear = ratio * lum_pulled.clamp(min=0)

    # 7. Clarity (局部对比度, 参数化强度)
    #    clarity ∈ [-100, 100] → strength ∈ [-0.5, 0.5]; 默认 0 = 不变
    clarity = params.get('clarity', torch.zeros(B, device=img.device))
    clarity_strength = (clarity / 100.0 * 0.5).view(B, 1, 1, 1)
    lum = rgb_to_luminance(linear.clamp(0, 1))
    linear = _clarity(linear, lum, strength=clarity_strength, kernel_size=15)

    # 8. Saturation
    saturation = params['saturation']
    lum = rgb_to_luminance(linear.clamp(0, 1))
    sat_factor = (1.0 + saturation / 100.0).view(B, 1, 1, 1)
    linear = lum + (linear - lum) * sat_factor

    # 9. Vibrance (optional)
    vibrance = params.get('vibrance', torch.zeros(B, device=img.device))
    lum = rgb_to_luminance(linear.clamp(0, 1))
    chroma = ((linear - lum) ** 2).mean(dim=1, keepdim=True).sqrt() + 1e-8
    chroma_ref = chroma.flatten(1).mean(dim=1).view(B, 1, 1, 1).clamp(min=1e-8) * 3.0
    sat_ratio = (chroma / chroma_ref).clamp(0, 1)
    weight = (1.0 - sat_ratio).clamp(min=1e-4) ** 1.5
    vib_factor = 1.0 + (vibrance / 100.0).view(B, 1, 1, 1) * weight
    linear = lum + (linear - lum) * vib_factor

    return linear_to_srgb(linear)


def image_reconstruction_loss(img: torch.Tensor,
                               pred_params: Dict[str, torch.Tensor],
                               gt_params: Dict[str, torch.Tensor],
                               l1_weight: float = 1.0,
                               ssim_weight: float = 1.0) -> torch.Tensor:
    """
    图像重建损失: diff_ISP(img, pred_params) vs diff_ISP(img, gt_params)

    Args:
        img:         (B, 3, H, W) [0, 1] 原始图像
        pred_params: 预测参数 dict (denormalized)
        gt_params:   GT 参数 dict (denormalized)

    Returns:
        scalar loss
    """
    pred_rendered = apply_diff_isp(img, pred_params)
    gt_rendered   = apply_diff_isp(img, gt_params)

    loss = torch.tensor(0.0, device=img.device)
    if l1_weight > 0:
        loss = loss + l1_weight * F.l1_loss(pred_rendered, gt_rendered)
    if ssim_weight > 0:
        loss = loss + ssim_weight * ssim_loss(pred_rendered, gt_rendered)
    return loss

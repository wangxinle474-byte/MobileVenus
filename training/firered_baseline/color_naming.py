"""Color naming module for v9 NamedCurves-style models.

Decomposes RGB images into N color probability maps using soft HSV-based
classification. This is a simplified, fully differentiable alternative to
Van de Weijer et al. (2007) LUT-based color naming, which requires a 32^3
RGB lookup table that's not easily accessible from all regions.

Supported groupings:
  - 'full6' (default, matches NamedCurves paper):
      red / blue / green / yellow-orange-brown / pink-purple / achromatic
  - 'compact3' (for v9a simplified):
      warm (red+yellow+orange+brown) / cool (green+blue+purple+pink) / neutral
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


# ============================================================
# Hue centers (degrees) for soft Gaussian naming
# ============================================================
# Each color is defined by a hue center and standard deviation (in degrees).
# Achromatic (low saturation) is handled separately via saturation gate.
HUE_CENTERS_DEG = {
    'red':    0.0,    # wraps around 360
    'orange': 30.0,
    'yellow': 60.0,
    'green':  120.0,
    'cyan':   180.0,
    'blue':   220.0,
    'purple': 280.0,
    'pink':   330.0,
}
HUE_SIGMA_DEG = 25.0  # soft assignment width


# ============================================================
# Color naming forward
# ============================================================
def rgb_to_hsv(rgb: torch.Tensor) -> torch.Tensor:
    """Differentiable RGB -> HSV. Input/output in [0, 1] except H in [0, 1].

    Args:
        rgb: (B, 3, H, W) in [0, 1]
    Returns:
        hsv: (B, 3, H, W) where H, S, V all in [0, 1]
    """
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    maxc = rgb.max(dim=1).values
    minc = rgb.min(dim=1).values
    v = maxc
    diff = maxc - minc
    s = diff / (maxc + 1e-8)

    # Hue computation (avoid NaN by adding small eps where diff is 0)
    rc = (maxc - r) / (diff + 1e-8)
    gc = (maxc - g) / (diff + 1e-8)
    bc = (maxc - b) / (diff + 1e-8)

    h_r = (bc - gc)              # r is max
    h_g = 2.0 + (rc - bc)         # g is max
    h_b = 4.0 + (gc - rc)         # b is max

    h = torch.where(maxc == r, h_r,
        torch.where(maxc == g, h_g, h_b))
    h = (h / 6.0) % 1.0           # to [0, 1]

    # Where diff == 0, hue is undefined; set to 0
    h = torch.where(diff < 1e-6, torch.zeros_like(h), h)

    return torch.stack([h, s, v], dim=1)


def _hue_likelihood(h_deg: torch.Tensor, center_deg: float,
                    sigma_deg: float) -> torch.Tensor:
    """Gaussian-shaped likelihood on a circular hue axis (degrees).

    h_deg, center_deg in degrees. Handles wraparound via shortest arc.
    """
    diff = (h_deg - center_deg + 180.0) % 360.0 - 180.0
    return torch.exp(-0.5 * (diff / sigma_deg) ** 2)


def compute_color_naming_maps(
    rgb: torch.Tensor,
    grouping: str = 'full6',
    achromatic_sat_threshold: float = 0.15,
    achromatic_sat_sigma: float = 0.08,
) -> torch.Tensor:
    """Compute soft color naming probability maps for an RGB image.

    Args:
        rgb: (B, 3, H, W) in [0, 1]
        grouping: 'full6' (red/blue/green/yob/pp/achromatic) or
                  'compact3' (warm/cool/neutral)
        achromatic_sat_threshold: saturation below this gets weighted toward
                                  achromatic (in [0,1] HSV S space)
        achromatic_sat_sigma: smoothness of the saturation gate
    Returns:
        maps: (B, N, H, W) where N=6 or 3, each pixel sums to ~1 across N
    """
    assert grouping in ('full6', 'compact3'), f'Unknown grouping: {grouping}'

    hsv = rgb_to_hsv(rgb)               # (B, 3, H, W)
    h_deg = hsv[:, 0] * 360.0             # (B, H, W)
    s = hsv[:, 1]                          # (B, H, W)

    # Achromatic gate: sigmoid on (s - threshold)
    # High when s < threshold (achromatic), low when s > threshold (chromatic)
    achromatic_gate = torch.sigmoid(
        -(s - achromatic_sat_threshold) / achromatic_sat_sigma)

    # Per-color hue likelihoods (B, H, W) for each of 8 base hues
    hue_likes = {}
    for name, center in HUE_CENTERS_DEG.items():
        hue_likes[name] = _hue_likelihood(h_deg, center, HUE_SIGMA_DEG)

    if grouping == 'full6':
        # Paper grouping: red / blue / green / yob / pp / achromatic
        red_like    = hue_likes['red']
        green_like  = hue_likes['green']
        blue_like   = hue_likes['blue'] + 0.5 * hue_likes['cyan']
        yob_like    = (hue_likes['yellow'] + hue_likes['orange']
                       + 0.3 * hue_likes['red'])     # brown lives near red+orange dark
        pp_like     = hue_likes['pink'] + hue_likes['purple']

        # Stack chromatic likelihoods; weight by (1 - achromatic_gate)
        chrom = torch.stack([red_like, green_like, blue_like,
                             yob_like, pp_like], dim=1)  # (B, 5, H, W)
        chrom = chrom * (1.0 - achromatic_gate.unsqueeze(1))

        achrom = achromatic_gate.unsqueeze(1)             # (B, 1, H, W)
        maps = torch.cat([chrom, achrom], dim=1)          # (B, 6, H, W)

    else:  # compact3: warm / cool / neutral
        warm_like = (hue_likes['red'] + hue_likes['orange']
                     + hue_likes['yellow'] + 0.5 * hue_likes['pink'])
        cool_like = (hue_likes['green'] + hue_likes['cyan']
                     + hue_likes['blue'] + 0.5 * hue_likes['purple'])

        chrom = torch.stack([warm_like, cool_like], dim=1)  # (B, 2, H, W)
        chrom = chrom * (1.0 - achromatic_gate.unsqueeze(1))

        neutral = achromatic_gate.unsqueeze(1)               # (B, 1, H, W)
        maps = torch.cat([chrom, neutral], dim=1)            # (B, 3, H, W)

    # Normalize so each pixel sums to 1
    maps = maps / (maps.sum(dim=1, keepdim=True) + 1e-8)
    return maps


# ============================================================
# Sanity check
# ============================================================
if __name__ == '__main__':
    # Test on a tiny image with known colors
    rgb = torch.tensor([
        [[[1.0, 0.0, 0.5, 0.5]],  # row R: red, ?, mid, mid
         [[0.0, 1.0, 0.5, 0.5]],  # row G: ?, green, mid, mid
         [[0.0, 0.0, 0.5, 0.0]]],  # row B: black, ?, neutral, ?
    ])  # (1, 3, 1, 4)
    maps6 = compute_color_naming_maps(rgb, grouping='full6')
    maps3 = compute_color_naming_maps(rgb, grouping='compact3')
    print('full6 maps shape:', maps6.shape, 'sum per pixel:',
          maps6.sum(dim=1)[0, 0].tolist())
    print('compact3 maps shape:', maps3.shape, 'sum per pixel:',
          maps3.sum(dim=1)[0, 0].tolist())
    print('full6 [red, green, blue, yob, pp, achromatic]:')
    print(maps6[0, :, 0, :].tolist())

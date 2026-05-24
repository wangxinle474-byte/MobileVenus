"""Bezier tone curve module for v9 NamedCurves-style models.

Implements 1D Bezier curves parameterized by control point Y-values.
The X-coordinates are implicitly evenly distributed; the curve is evaluated
in Bernstein form: B(t) = Σ_m C(n, m) * t^m * (1-t)^(n-m) * P_m
with the first control point P_0 fixed at 0 (black → black guarantee).
"""
from __future__ import annotations

import math
import torch


def bernstein_basis(t: torch.Tensor, n: int) -> torch.Tensor:
    """Compute Bernstein basis values B_{n,m}(t) for m = 0..n.

    Args:
        t: (...) tensor with values in [0, 1]
        n: polynomial degree (= num_control_points - 1)
    Returns:
        basis: (..., n+1) tensor
    """
    basis = []
    one_minus_t = 1.0 - t
    for m in range(n + 1):
        coef = math.comb(n, m)
        # Use stable power: t^m * (1-t)^(n-m)
        # For t exactly 0 with m=0, 0^0 is defined as 1 in our convention.
        if m == 0:
            t_pow = torch.ones_like(t)
        else:
            t_pow = t ** m
        if n - m == 0:
            one_pow = torch.ones_like(t)
        else:
            one_pow = one_minus_t ** (n - m)
        basis.append(coef * t_pow * one_pow)
    return torch.stack(basis, dim=-1)


def apply_bezier_1d(x: torch.Tensor, cp: torch.Tensor,
                    fix_first_zero: bool = True) -> torch.Tensor:
    """Apply a 1D Bezier tone curve per-batch.

    Args:
        x: (B, ...) pixel values in [0, 1], where ... can include H, W
        cp: (B, M) control point Y-values (M = num_control_points)
                  If fix_first_zero is True, only the last M-1 values are
                  learned and P_0 is forced to 0.
        fix_first_zero: whether to clamp P_0 to 0 (black -> black)
    Returns:
        y: same shape as x, output of curve at each pixel
    """
    if fix_first_zero:
        zeros = torch.zeros_like(cp[..., :1])
        cp = torch.cat([zeros, cp[..., 1:]], dim=-1)

    M = cp.shape[-1]
    n = M - 1

    # Clamp x to [0, 1] to avoid edge instability
    x_safe = x.clamp(1e-6, 1.0 - 1e-6)

    basis = bernstein_basis(x_safe, n)  # (B, ..., M)

    # Reshape cp from (B, M) to broadcast over spatial dims:
    # If x has shape (B, H, W), basis has shape (B, H, W, M), cp -> (B, 1, 1, M)
    while cp.dim() < basis.dim():
        cp = cp.unsqueeze(-2)

    return (basis * cp).sum(dim=-1)


def apply_per_channel_bezier(rgb: torch.Tensor, cp: torch.Tensor,
                              fix_first_zero: bool = True) -> torch.Tensor:
    """Apply 3 independent Bezier curves to R/G/B channels.

    Args:
        rgb: (B, 3, H, W) input image in [0, 1]
        cp: (B, 3, M) control points, one set per channel
        fix_first_zero: clamp P_0 to 0 for each channel
    Returns:
        out: (B, 3, H, W) curve output
    """
    outs = []
    for c in range(3):
        x_c = rgb[:, c]                 # (B, H, W)
        cp_c = cp[:, c]                  # (B, M)
        y_c = apply_bezier_1d(x_c, cp_c, fix_first_zero=fix_first_zero)
        outs.append(y_c)
    return torch.stack(outs, dim=1)      # (B, 3, H, W)


def apply_per_color_per_channel_bezier(
    rgb: torch.Tensor, cp: torch.Tensor,
    fix_first_zero: bool = True,
) -> torch.Tensor:
    """Apply N×3 Bezier curves (one per color group, per channel).

    Args:
        rgb: (B, 3, H, W) standardized image in [0, 1]
        cp: (B, N, 3, M) control points for N color groups × 3 channels
        fix_first_zero: clamp P_0 to 0 for every curve
    Returns:
        out: (B, N, 3, H, W) one enhanced image per color group
    """
    B, N, _, M = cp.shape
    outs = []
    for n_idx in range(N):
        cp_n = cp[:, n_idx]                                  # (B, 3, M)
        y_n = apply_per_channel_bezier(rgb, cp_n, fix_first_zero=fix_first_zero)
        outs.append(y_n)                                      # (B, 3, H, W)
    return torch.stack(outs, dim=1)                          # (B, N, 3, H, W)


# ============================================================
# Sanity check
# ============================================================
if __name__ == '__main__':
    # Identity curve: control points = evenly spaced [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    # Bezier with these CPs should give the identity y = x
    M = 7
    identity_cp_y = torch.linspace(0, 1, M).unsqueeze(0)  # (1, M)
    x = torch.linspace(0, 1, 11).unsqueeze(0)              # (1, 11)
    y = apply_bezier_1d(x, identity_cp_y, fix_first_zero=True)
    print(f'identity test (M={M}):')
    print(f'  x = {x[0].tolist()}')
    print(f'  y = {[round(v, 3) for v in y[0].tolist()]}')
    # NOTE: Bezier with linear control points is NOT exactly identity due to
    # Bernstein basis weights, but should be close to a smooth monotone curve.

    # S-curve: low end pulled down, high end pulled up
    s_cp = torch.tensor([[0.0, 0.05, 0.1, 0.4, 0.8, 0.95, 1.0]])  # (1, 7)
    y_s = apply_bezier_1d(x, s_cp, fix_first_zero=True)
    print(f'\nS-curve test:')
    print(f'  cp = {s_cp[0].tolist()}')
    print(f'  y  = {[round(v, 3) for v in y_s[0].tolist()]}')

    # Test per-channel
    rgb = torch.rand(2, 3, 4, 4)
    cp_3 = torch.linspace(0, 1, M).unsqueeze(0).unsqueeze(0).expand(2, 3, M).contiguous()
    out = apply_per_channel_bezier(rgb, cp_3)
    print(f'\nper-channel output shape: {out.shape}')

    # Test per-color
    rgb = torch.rand(2, 3, 4, 4)
    cp_N3 = torch.linspace(0, 1, M).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(2, 6, 3, M).contiguous()
    out = apply_per_color_per_channel_bezier(rgb, cp_N3)
    print(f'per-color output shape: {out.shape}')

    # Gradient check
    cp_g = torch.linspace(0, 1, M).unsqueeze(0).requires_grad_(True)
    x_g = torch.tensor([[0.3, 0.5, 0.7]], requires_grad=True)
    y_g = apply_bezier_1d(x_g, cp_g, fix_first_zero=True)
    loss = y_g.sum()
    loss.backward()
    print(f'\ngradient w.r.t. cp: {cp_g.grad}')
    print(f'gradient w.r.t. x: {x_g.grad}')
    assert torch.isfinite(cp_g.grad).all(), 'NaN/Inf in cp grad'
    assert torch.isfinite(x_g.grad).all(), 'NaN/Inf in x grad'
    print('  all gradients finite ✓')

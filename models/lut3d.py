"""Image-Adaptive 3D LUT — 可微三线性插值实现.

参考: Zeng et al., "Learning Image-Adaptive 3D Lookup Tables for High
Performance Photo Enhancement in Real-Time", CVPR 2020 / TPAMI 2022.

核心思路:
  - 学习 N 个 basis 3D LUT (D×D×D×3), 其中 D=33 (标准)
  - 对每张图, encoder 预测 N 个融合权重 w_i (softmax)
  - 最终 LUT = sum(w_i * LUT_i)
  - 通过三线性插值将 LUT 应用到图像上

优势: 3D LUT 有 ~107K 自由参数, 远超 7D ISP 的表达力.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TrilinearInterpolation(nn.Module):
    """可微三线性插值: 将 3D LUT 应用到 RGB 图像.

    LUT shape: (B, 3, D, D, D)  —— 每个 output channel 一个 D^3 表
    img shape: (B, 3, H, W)    —— [0, 1] sRGB

    输出:     (B, 3, H, W)    —— [0, 1] 变换后
    """

    def __init__(self):
        super().__init__()

    def forward(self, lut: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            lut: (B, 3, D, D, D)  3D LUT
            img: (B, 3, H, W)     input image [0, 1]
        Returns:
            (B, 3, H, W) transformed image
        """
        B, C, H, W = img.shape
        D = lut.shape[2]

        # img [0,1] → 索引空间 [0, D-1]
        img_c = img.clamp(0, 1) * (D - 1)  # (B, 3, H, W)

        # 分离 R, G, B 通道作为 LUT 的坐标
        r = img_c[:, 0:1]  # (B, 1, H, W)
        g = img_c[:, 1:2]
        b = img_c[:, 2:3]

        # floor / ceil 索引
        r0 = r.floor().long().clamp(0, D - 2)
        g0 = g.floor().long().clamp(0, D - 2)
        b0 = b.floor().long().clamp(0, D - 2)
        r1 = r0 + 1
        g1 = g0 + 1
        b1 = b0 + 1

        # 小数部分 (插值权重)
        fr = (r - r0.float()).clamp(0, 1)  # (B, 1, H, W)
        fg = (g - g0.float()).clamp(0, 1)
        fb = (b - b0.float()).clamp(0, 1)

        # 展平空间维度以便 gather
        r0 = r0.squeeze(1).reshape(B, -1)  # (B, H*W)
        g0 = g0.squeeze(1).reshape(B, -1)
        b0 = b0.squeeze(1).reshape(B, -1)
        r1 = r1.squeeze(1).reshape(B, -1)
        g1 = g1.squeeze(1).reshape(B, -1)
        b1 = b1.squeeze(1).reshape(B, -1)

        # 从 LUT 采样 8 个角点
        def sample_lut(ri, gi, bi):
            """从 lut (B, 3, D, D, D) 中采样, 返回 (B, 3, H*W)."""
            # 线性化索引: idx = ri * D * D + gi * D + bi
            idx = ri * D * D + gi * D + bi  # (B, H*W)
            idx = idx.unsqueeze(1).expand(-1, 3, -1)  # (B, 3, H*W)
            lut_flat = lut.reshape(B, 3, -1)  # (B, 3, D^3)
            return torch.gather(lut_flat, 2, idx)  # (B, 3, H*W)

        # 8 个角点
        c000 = sample_lut(r0, g0, b0)
        c001 = sample_lut(r0, g0, b1)
        c010 = sample_lut(r0, g1, b0)
        c011 = sample_lut(r0, g1, b1)
        c100 = sample_lut(r1, g0, b0)
        c101 = sample_lut(r1, g0, b1)
        c110 = sample_lut(r1, g1, b0)
        c111 = sample_lut(r1, g1, b1)

        # 插值权重 reshape
        fr = fr.reshape(B, 1, -1)  # (B, 1, H*W)
        fg = fg.reshape(B, 1, -1)
        fb = fb.reshape(B, 1, -1)

        # 三线性插值
        c00 = c000 * (1 - fb) + c001 * fb
        c01 = c010 * (1 - fb) + c011 * fb
        c10 = c100 * (1 - fb) + c101 * fb
        c11 = c110 * (1 - fb) + c111 * fb

        c0 = c00 * (1 - fg) + c01 * fg
        c1 = c10 * (1 - fg) + c11 * fg

        out = c0 * (1 - fr) + c1 * fr  # (B, 3, H*W)
        return out.reshape(B, 3, H, W).clamp(0, 1)


class Basis3DLUT(nn.Module):
    """N 个可学习的 basis 3D LUT.

    Args:
        n_luts: basis LUT 数量 (默认 3)
        dim: LUT 网格维度 (默认 33)
    """

    def __init__(self, n_luts: int = 3, dim: int = 33):
        super().__init__()
        self.n_luts = n_luts
        self.dim = dim
        self.interp = TrilinearInterpolation()

        # 初始化: 第一个 LUT = identity, 其余 = identity + 小噪声
        for i in range(n_luts):
            lut = self._identity_lut(dim)
            if i > 0:
                lut = lut + torch.randn_like(lut) * 0.01
            self.register_parameter(
                f'lut_{i}', nn.Parameter(lut))

    def _identity_lut(self, dim: int) -> torch.Tensor:
        """生成恒等 3D LUT: LUT[r,g,b] = (r,g,b) / (dim-1).

        Returns: (3, D, D, D)
        """
        coords = torch.linspace(0, 1, dim)
        # r 变化最快的维度
        r = coords.view(dim, 1, 1).expand(dim, dim, dim)
        g = coords.view(1, dim, 1).expand(dim, dim, dim)
        b = coords.view(1, 1, dim).expand(dim, dim, dim)
        return torch.stack([r, g, b], dim=0)  # (3, D, D, D)

    def forward(self, weights: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            weights: (B, n_luts)  softmax 融合权重
            img:     (B, 3, H, W) 输入图像 [0, 1]
        Returns:
            (B, 3, H, W) 变换后图像
        """
        B = img.shape[0]

        # 加权融合 basis LUTs
        luts = [getattr(self, f'lut_{i}') for i in range(self.n_luts)]
        # stack: (n_luts, 3, D, D, D)
        lut_stack = torch.stack(luts, dim=0)
        # weights: (B, n_luts) → (B, n_luts, 1, 1, 1, 1)
        w = weights.view(B, self.n_luts, 1, 1, 1, 1)
        # fused: (B, 3, D, D, D)
        fused_lut = (w * lut_stack.unsqueeze(0)).sum(dim=1)

        return self.interp(fused_lut, img)


def lut_smoothness_loss(lut_module: Basis3DLUT) -> torch.Tensor:
    """3D LUT 平滑性正则: 相邻格点差异应小 (避免色彩跳变).

    对每个 basis LUT, 计算相邻格点的 L2 差异之和.
    """
    total = torch.tensor(0.0, device=next(lut_module.parameters()).device)
    for i in range(lut_module.n_luts):
        lut = getattr(lut_module, f'lut_{i}')  # (3, D, D, D)
        # 三个方向的差分
        dr = (lut[:, 1:, :, :] - lut[:, :-1, :, :]) ** 2
        dg = (lut[:, :, 1:, :] - lut[:, :, :-1, :]) ** 2
        db = (lut[:, :, :, 1:] - lut[:, :, :, :-1]) ** 2
        total = total + dr.mean() + dg.mean() + db.mean()
    return total / lut_module.n_luts


def lut_monotonicity_loss(lut_module: Basis3DLUT) -> torch.Tensor:
    """单调性正则: 鼓励 LUT 沿对角线方向单调递增 (亮的输入产生亮的输出).

    具体: 对 R channel, LUT[r+1,g,b].R >= LUT[r,g,b].R (同理 G, B).
    违反单调性的部分给予惩罚.
    """
    total = torch.tensor(0.0, device=next(lut_module.parameters()).device)
    for i in range(lut_module.n_luts):
        lut = getattr(lut_module, f'lut_{i}')  # (3, D, D, D)
        # R channel 沿 r 轴
        dr_r = lut[0, 1:, :, :] - lut[0, :-1, :, :]
        # G channel 沿 g 轴
        dg_g = lut[1, :, 1:, :] - lut[1, :, :-1, :]
        # B channel 沿 b 轴
        db_b = lut[2, :, :, 1:] - lut[2, :, :, :-1]
        # 惩罚递减部分 (relu(-diff))
        total = total + F.relu(-dr_r).mean()
        total = total + F.relu(-dg_g).mean()
        total = total + F.relu(-db_b).mean()
    return total / lut_module.n_luts


if __name__ == '__main__':
    # 快速测试
    luts = Basis3DLUT(n_luts=3, dim=33)
    img = torch.rand(2, 3, 64, 64)
    weights = torch.softmax(torch.randn(2, 3), dim=1)
    out = luts(weights, img)
    print(f'Input:  {img.shape}')
    print(f'Output: {out.shape}')
    print(f'LUT params: {sum(p.numel() for p in luts.parameters()) / 1e3:.1f}K')
    print(f'Smooth loss: {lut_smoothness_loss(luts):.6f}')
    print(f'Mono   loss: {lut_monotonicity_loss(luts):.6f}')

    # identity test: 权重全给第一个 LUT (identity), 输出应≈输入
    w_id = torch.zeros(2, 3)
    w_id[:, 0] = 1.0
    out_id = luts(w_id, img)
    diff = (out_id - img).abs().mean()
    print(f'Identity diff: {diff:.6f} (should be ~0)')

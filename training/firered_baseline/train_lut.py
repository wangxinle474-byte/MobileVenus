"""方案A: Image-Adaptive 3D LUT 训练 — 突破 7D ISP 参数天花板.

思路:
  - MobileViTSmall encoder + action embedding → 预测 N 个 basis LUT 融合权重
  - N 个可学习 3D LUT (33×33×33×3) 通过三线性插值应用到原图
  - 直接用像素级 L1 + SSIM loss 对齐 FireRed 编辑结果
  - 不经过 7D ISP 参数空间, 避免 ceiling 限制

vs 7D baseline:
  - 7D ISP ceiling: ~23.94 dB PSNR
  - 3D LUT 文献报告: ~25+ dB PSNR (FiveK Expert C)
  - 本方案目标: 验证 LUT 是否能显著超越 7D ceiling
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

# Set up sys.path BEFORE any local-package imports so that `training.*` and
# `models.*` resolve when this script is invoked as `python <path>/train_lut.py`.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as TF

from models.vision_encoder import MobileViTSmall  # noqa: E402
from models.lut3d import Basis3DLUT, lut_smoothness_loss, lut_monotonicity_loss  # noqa: E402
from models.diff_isp import ssim_loss, apply_diff_isp  # noqa: E402
from models.nilut import NILUT  # noqa: E402  (v10b residual head)
from models.vera_renderer import VeraRenderer  # noqa: E402  (v10c full renderer)
from training.firered_baseline.implicit_head import ImplicitResidualHead  # noqa: E402
from training.firered_baseline.color_naming import compute_color_naming_maps  # noqa: E402
from training.firered_baseline.bezier import apply_per_color_per_channel_bezier  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)

# ============================================================
# Configuration
# ============================================================
ACTIONS = ['contrast', 'saturation', 'shadows', 'highlights', 'wb']
ACTION_TO_IDX = {a: i for i, a in enumerate(ACTIONS)}

TIER_WEIGHTS = {
    'A excellent': 1.2,
    'B good':      1.0,
    'C acceptable': 0.5,
}

# 7D ISP 参数配置 (用于 v8 param+residual LUT 混合模型)
PARAM_NAMES_7D = [
    'white_balance', 'brightness', 'contrast',
    'shadows', 'highlights', 'saturation', 'clarity',
]
PARAM_NORM_7D = {
    'white_balance': {'center': 6000.0, 'scale': 4000.0,
                      'clip': (2000.0, 10000.0)},
    'brightness':    {'center': 0.0,    'scale': 100.0,
                      'clip': (-150.0, 150.0)},
    'contrast':      {'center': 0.0,    'scale': 100.0,
                      'clip': (-100.0, 100.0)},
    'shadows':       {'center': 0.0,    'scale': 100.0,
                      'clip': (-100.0, 100.0)},
    'highlights':    {'center': 0.0,    'scale': 100.0,
                      'clip': (-100.0, 100.0)},
    'saturation':    {'center': 0.0,    'scale': 100.0,
                      'clip': (-100.0, 100.0)},
    'clarity':       {'center': 0.0,    'scale': 100.0,
                      'clip': (-100.0, 100.0)},
}


def normalize_param_7d(name: str, value: float) -> float:
    cfg = PARAM_NORM_7D[name]
    v = max(cfg['clip'][0], min(cfg['clip'][1], value))
    return (v - cfg['center']) / cfg['scale']


# action_idx (in ACTIONS order) → param_7d_idx (in PARAM_NAMES_7D order)
# Used by parameter-perturbation training to inject noise only on the
# active action's parameter. Other 7D slots are always 0 in per-action
# data and should not receive noise.
ACTION_TO_PARAM_NAME = {
    'contrast': 'contrast',
    'saturation': 'saturation',
    'shadows': 'shadows',
    'highlights': 'highlights',
    'wb': 'white_balance',
    'brightness': 'brightness',
    'clarity': 'clarity',
}
ACTION_TO_PARAM_7D_INDICES = [
    PARAM_NAMES_7D.index(ACTION_TO_PARAM_NAME[a]) for a in ACTIONS
]


def set_actions(actions: list[str]):
    global ACTIONS, ACTION_TO_IDX, ACTION_TO_PARAM_7D_INDICES
    unknown = [a for a in actions if a not in ACTION_TO_PARAM_NAME]
    if unknown:
        raise ValueError(f'unknown actions: {unknown}')
    ACTIONS[:] = list(actions)
    ACTION_TO_IDX.clear()
    ACTION_TO_IDX.update({a: i for i, a in enumerate(ACTIONS)})
    ACTION_TO_PARAM_7D_INDICES[:] = [
        PARAM_NAMES_7D.index(ACTION_TO_PARAM_NAME[a]) for a in ACTIONS
    ]


def denormalize_params_7d(params_norm: torch.Tensor) -> dict:
    """Standalone (B, 7) [-1,1] → dict of (B,) physical params.

    Mirrors NamedCurvesPredictor._denorm_params but usable outside the
    model (e.g., in the training loop for on-the-fly target re-render).
    """
    out = {}
    for i, name in enumerate(PARAM_NAMES_7D):
        cfg = PARAM_NORM_7D[name]
        out[name] = params_norm[:, i] * cfg['scale'] + cfg['center']
    return out


# ============================================================
# Dataset
# ============================================================
class LUTDataset(Dataset):
    """返回 (orig_image, target_image, action_onehot, tier_weight).

    orig_image:   [0, 1] 原图 (用于 LUT 输入)
    target_image: [0, 1] FireRed 编辑结果 (像素监督目标)
    """

    def __init__(self, samples: list, image_size: int = 256,
                 is_train: bool = True, color_aug: bool = False,
                 random_crop: bool = False):
        self.samples = samples
        self.image_size = image_size
        self.is_train = is_train
        self.color_aug = color_aug
        self.random_crop = random_crop
        self.resize = T.Resize((image_size, image_size))
        # random_crop: resize to slightly larger, then crop
        self.resize_for_crop = T.Resize((int(image_size * 1.15), int(image_size * 1.15)))
        self.to_tensor = T.ToTensor()
        # encoder 输入需要 normalize
        self.norm = T.Normalize(
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        # 加载原图
        try:
            orig = Image.open(s['orig_path']).convert('RGB')
        except Exception as e:
            logger.warning(f'orig load fail {s["orig_path"]}: {e}')
            orig = Image.new('RGB', (self.image_size, self.image_size))

        # 加载 target (FireRed edit)
        target_path = s['target_path']
        # target_path 可能是相对路径
        if not Path(target_path).is_absolute():
            target_path = PROJECT_ROOT / target_path
        try:
            target = Image.open(target_path).convert('RGB')
        except Exception as e:
            logger.warning(f'target load fail {target_path}: {e}')
            target = orig.copy()

        # 随机水平翻转 (两张图同步)
        do_flip = self.is_train and random.random() < 0.5
        if do_flip:
            orig = orig.transpose(Image.FLIP_LEFT_RIGHT)
            target = target.transpose(Image.FLIP_LEFT_RIGHT)

        # resize (+ optional random crop)
        if self.random_crop and self.is_train:
            orig = self.resize_for_crop(orig)
            target = self.resize_for_crop(target)
            i, j, h, w = T.RandomCrop.get_params(
                orig, (self.image_size, self.image_size))
            orig = TF.crop(orig, i, j, h, w)
            target = TF.crop(target, i, j, h, w)
        else:
            orig = self.resize(orig)
            target = self.resize(target)

        # synchronized color augmentation (brightness + contrast jitter)
        if self.color_aug and self.is_train:
            bf = random.uniform(0.85, 1.15)
            cf = random.uniform(0.85, 1.15)
            orig = TF.adjust_brightness(orig, bf)
            target = TF.adjust_brightness(target, bf)
            orig = TF.adjust_contrast(orig, cf)
            target = TF.adjust_contrast(target, cf)

        # [0, 1] tensor (LUT 输入 + 像素监督)
        orig_t = self.to_tensor(orig)      # (3, H, W)
        target_t = self.to_tensor(target)  # (3, H, W)

        # encoder 输入 (normalized)
        enc_input = self.norm(orig_t)

        action_idx = ACTION_TO_IDX[s['action']]
        action_oh = torch.zeros(len(ACTIONS), dtype=torch.float32)
        action_oh[action_idx] = 1.0

        tier_w = TIER_WEIGHTS.get(s.get('quality_tier', ''), 0.5)

        # 7D ISP params (for v8 param supervision)
        P = s.get('P_inferred', {})
        params_norm_7d = torch.tensor(
            [normalize_param_7d(p, P.get(p) or 0.0) for p in PARAM_NAMES_7D],
            dtype=torch.float32)

        return {
            'enc_input': enc_input,      # (3, H, W) normalized for encoder
            'orig': orig_t,              # (3, H, W) [0,1] for LUT input
            'target': target_t,          # (3, H, W) [0,1] pixel target
            'action_onehot': action_oh,
            'action_idx': action_idx,
            'action': s['action'],
            'tier_weight': torch.tensor(tier_w, dtype=torch.float32),
            'source_image': s.get('source_image', ''),
            'params_norm_7d': params_norm_7d,  # (7,) for v8
        }


# ============================================================
# Model
# ============================================================
class LUTPredictor(nn.Module):
    """MobileViTSmall + action embedding → N 个 LUT 融合权重.

    总参数: encoder (~5.6M) + LUT basis (~323K) + head (~少量)
    """

    def __init__(self, n_luts: int = 3, lut_dim: int = 33,
                 image_size: int = 256, visual_dim: int = 384,
                 n_actions: int = 5, dropout: float = 0.3):
        super().__init__()
        self.n_luts = n_luts

        # Visual encoder
        self.encoder = MobileViTSmall(
            image_size=image_size,
            output_dim=visual_dim,
            use_se=True,
            use_fpn=True,
        )

        # Action embedding
        self.action_emb = nn.Linear(n_actions, 32)

        # Weight prediction head
        fused_dim = visual_dim + 32
        self.weight_head = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Dropout(dropout),
            nn.Linear(fused_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_luts),
            # 不加 softmax, forward 里做
        )

        # Learnable basis 3D LUTs
        self.lut_basis = Basis3DLUT(n_luts=n_luts, dim=lut_dim)

    def forward(self, enc_input: torch.Tensor, orig: torch.Tensor,
                action_onehot: torch.Tensor):
        """
        Args:
            enc_input:      (B, 3, H, W) normalized encoder input
            orig:           (B, 3, H, W) [0, 1] original for LUT
            action_onehot:  (B, 5) action one-hot
        Returns:
            output:  (B, 3, H, W) [0, 1] LUT-transformed image
            weights: (B, n_luts) softmax weights (for logging)
        """
        feat = self.encoder(enc_input)         # (B, 384)
        a = self.action_emb(action_onehot)     # (B, 32)
        fused = torch.cat([feat, a], dim=1)    # (B, 416)

        logits = self.weight_head(fused)       # (B, n_luts)
        weights = F.softmax(logits, dim=1)     # (B, n_luts)

        output = self.lut_basis(weights, orig) # (B, 3, H, W)
        return output, weights


class PerActionLUTPredictor(nn.Module):
    """每个 action 拥有独立的 N 个 basis LUT.

    思路: wb/saturation 等 action 的色彩变换差异巨大,
    共享 basis LUT 会互相干扰. 分 action 学专用 LUT 集合.

    参数量: encoder (~2.9M) + 5 action × N × LUT_dim³ × 3
    """

    def __init__(self, n_luts: int = 3, lut_dim: int = 33,
                 image_size: int = 256, visual_dim: int = 384,
                 n_actions: int = 5, dropout: float = 0.3):
        super().__init__()
        self.n_luts = n_luts
        self.n_actions = n_actions

        self.encoder = MobileViTSmall(
            image_size=image_size, output_dim=visual_dim,
            use_se=True, use_fpn=True,
        )

        # 每个 action 一组独立 basis LUTs
        self.action_luts = nn.ModuleList([
            Basis3DLUT(n_luts=n_luts, dim=lut_dim)
            for _ in range(n_actions)
        ])

        # 共享 encoder, 每 action 一个 weight head
        self.weight_heads = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(visual_dim),
                nn.Dropout(dropout),
                nn.Linear(visual_dim, 32),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(32, n_luts),
            )
            for _ in range(n_actions)
        ])

    def forward(self, enc_input, orig, action_onehot):
        feat = self.encoder(enc_input)           # (B, 384)
        action_idx = action_onehot.argmax(dim=1) # (B,)
        B = orig.shape[0]

        outputs = []
        all_weights = []
        for b in range(B):
            ai = action_idx[b].item()
            logits = self.weight_heads[ai](feat[b:b+1])     # (1, n_luts)
            w = F.softmax(logits, dim=1)                     # (1, n_luts)
            out = self.action_luts[ai](w, orig[b:b+1])      # (1, 3, H, W)
            outputs.append(out)
            all_weights.append(w)

        return torch.cat(outputs, dim=0), torch.cat(all_weights, dim=0)

    @property
    def lut_basis(self):
        """兼容 smoothness/monotonicity loss (聚合所有 action LUTs)."""
        return self.action_luts


def lut_reg_loss_per_action(action_luts):
    """对 PerActionLUT 的所有 action LUT 计算 smooth + mono.

    支持 nn.ModuleList 和 nn.ModuleDict.
    """
    s_total, m_total = 0.0, 0.0
    if isinstance(action_luts, nn.ModuleDict):
        modules = list(action_luts.values())
    else:
        modules = list(action_luts)
    for lut_module in modules:
        s_total += lut_smoothness_loss(lut_module)
        m_total += lut_monotonicity_loss(lut_module)
    n = max(len(modules), 1)
    return s_total / n, m_total / n


class HybridLUTPredictor(nn.Module):
    """混合模型: wb 走 explicit 3-gain 分支, 其他 action 走 per-action LUT.

    动机 (基于 v6 诊断):
      - shadows/contrast 是 1D tone curve, LUT 表达完美 (26.03/25.01 dB)
      - wb 是全局乘性变换 R*=g_R, G*=g_G, B*=g_B
        17³ LUT 难精确表达, v6 仅 21.97 dB
      - 直接预测 (g_R, g_G, g_B) 解析形式, 无 LUT 离散误差

    架构:
      encoder (shared) → branch by action
        if action == 'wb':  → 3-gain head → orig * gains
        else:               → per-action LUT (same as v6)
    """

    def __init__(self, n_luts: int = 3, lut_dim: int = 17,
                 image_size: int = 256, visual_dim: int = 384,
                 dropout: float = 0.5):
        super().__init__()
        self.n_luts = n_luts
        self.wb_idx = ACTIONS.index('wb')
        self.non_wb_actions = [a for a in ACTIONS if a != 'wb']

        self.encoder = MobileViTSmall(
            image_size=image_size, output_dim=visual_dim,
            use_se=True, use_fpn=True,
        )

        # wb-specific 3-gain head (输出 raw, 后续 exp 保证 positive)
        self.wb_head = nn.Sequential(
            nn.LayerNorm(visual_dim),
            nn.Dropout(dropout),
            nn.Linear(visual_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 3),
        )

        # 非 wb action: 各一组 basis LUTs + weight head
        self.action_luts = nn.ModuleDict({
            a: Basis3DLUT(n_luts=n_luts, dim=lut_dim)
            for a in self.non_wb_actions
        })
        self.weight_heads = nn.ModuleDict({
            a: nn.Sequential(
                nn.LayerNorm(visual_dim),
                nn.Dropout(dropout),
                nn.Linear(visual_dim, 32),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(32, n_luts),
            )
            for a in self.non_wb_actions
        })

    def forward(self, enc_input, orig, action_onehot):
        feat = self.encoder(enc_input)            # (B, 384)
        action_idx = action_onehot.argmax(dim=1)  # (B,)
        B = orig.shape[0]

        outputs = []
        weights_log = []
        for b in range(B):
            ai = action_idx[b].item()
            if ai == self.wb_idx:
                # 3-gain branch: gain = exp(raw * 0.3), neutral ~1.0
                gains_raw = self.wb_head(feat[b:b+1])           # (1, 3)
                gains = torch.exp(gains_raw * 0.3)              # (1, 3)
                out = (orig[b:b+1] *
                       gains.view(1, 3, 1, 1)).clamp(0.0, 1.0)
                outputs.append(out)
                # 用 gains 占位 (与 n_luts 对齐, 不一定相同长度时 pad)
                w = torch.zeros(1, self.n_luts, device=feat.device)
                w[0, :3] = gains.squeeze(0)[:self.n_luts]
                weights_log.append(w)
            else:
                action_name = ACTIONS[ai]
                logits = self.weight_heads[action_name](feat[b:b+1])
                w = F.softmax(logits, dim=1)
                out = self.action_luts[action_name](w, orig[b:b+1])
                outputs.append(out)
                weights_log.append(w)

        return torch.cat(outputs, dim=0), torch.cat(weights_log, dim=0)


class ParamResidualLUTPredictor(nn.Module):
    """v8: 7D ISP 参数 + 残差 LUT 混合模型.

    Pipeline:
      orig → encoder(shared) → feat (384-dim)
        feat + action_emb → param_head → 7D ISP params (可解释)
        apply_diff_isp(orig, params) → coarse (粗结果, ~23.94 dB ceiling)
        feat → weight_head[action] → residual LUT(coarse) → refined (精修)

    优势:
      - 输出可解释的 ISP 参数 (wb, contrast, shadows, ...)
      - 残差 LUT 补偿 diff_isp 表达力不足, 突破 23.94 dB ceiling
      - 联合端到端训练, 参数分支和 LUT 分支互补
    """

    def __init__(self, n_luts: int = 3, lut_dim: int = 17,
                 image_size: int = 256, visual_dim: int = 384,
                 n_actions: int = 5, dropout: float = 0.5):
        super().__init__()
        self.n_luts = n_luts
        self.n_actions = n_actions

        # Shared visual encoder
        self.encoder = MobileViTSmall(
            image_size=image_size, output_dim=visual_dim,
            use_se=True, use_fpn=True,
        )

        # Action embedding (shared)
        self.action_emb = nn.Linear(n_actions, 32)
        fused_dim = visual_dim + 32

        # 7D ISP parameter prediction head
        self.param_head = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Dropout(dropout),
            nn.Linear(fused_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, len(PARAM_NAMES_7D)),
            nn.Tanh(),  # output in [-1, 1] normalized
        )
        # Zero-init final Linear so Tanh→0 at start (params at centers
        # → near-identity ISP → stable first-batch gradients).
        nn.init.zeros_(self.param_head[-2].weight)
        nn.init.zeros_(self.param_head[-2].bias)

        # Per-action residual LUTs (applied to coarse ISP output)
        self.action_luts = nn.ModuleList([
            Basis3DLUT(n_luts=n_luts, dim=lut_dim)
            for _ in range(n_actions)
        ])

        # Per-action LUT weight prediction heads
        self.weight_heads = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(visual_dim),
                nn.Dropout(dropout),
                nn.Linear(visual_dim, 32),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(32, n_luts),
            )
            for _ in range(n_actions)
        ])

    def _denorm_params(self, params_norm: torch.Tensor) -> dict:
        """(B, 7) normalized [-1,1] → dict of (B,) physical values."""
        params = {}
        for i, name in enumerate(PARAM_NAMES_7D):
            cfg = PARAM_NORM_7D[name]
            params[name] = params_norm[:, i] * cfg['scale'] + cfg['center']
        return params

    def forward(self, enc_input, orig, action_onehot):
        """
        Returns:
            refined:     (B, 3, H, W) final output (ISP coarse + LUT correction)
            weights:     (B, n_luts) LUT fusion weights
            coarse:      (B, 3, H, W) ISP-rendered intermediate
            params_norm: (B, 7) predicted ISP params in [-1, 1]
        """
        feat = self.encoder(enc_input)            # (B, 384)
        action_idx = action_onehot.argmax(dim=1)  # (B,)
        B = orig.shape[0]

        # --- Param branch (batch) ---
        a_emb = self.action_emb(action_onehot)    # (B, 32)
        fused = torch.cat([feat, a_emb], dim=1)   # (B, 416)
        params_norm = self.param_head(fused)       # (B, 7) in [-1, 1]

        # Denormalize → physical values → ISP render
        # NOTE: apply_diff_isp backward is numerically unstable (shadows
        # ratio `linear/(lum+1e-6)` → ~1e6 gradient on dark pixels;
        # WB `where` branch + pow boundary issues). We DETACH coarse from
        # the graph so LUT branch trains on a clean feature, and param
        # branch is supervised directly via MSE to P_inferred. coarse
        # remains a useful *input feature* for the LUT (action-conditioned
        # pseudo-rendering) without polluting gradients.
        params_phys = self._denorm_params(params_norm)
        with torch.no_grad():
            coarse = apply_diff_isp(orig, {k: v.detach()
                                            for k, v in params_phys.items()})
            coarse = torch.nan_to_num(coarse, nan=0.5,
                                       posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

        # --- Residual LUT branch (per-sample, action-specific) ---
        outputs = []
        all_weights = []
        for b in range(B):
            ai = action_idx[b].item()
            logits = self.weight_heads[ai](feat[b:b+1])  # (1, n_luts)
            w = F.softmax(logits, dim=1)                  # (1, n_luts)
            out = self.action_luts[ai](w, coarse[b:b+1])  # LUT applied to coarse
            outputs.append(out)
            all_weights.append(w)

        refined = torch.cat(outputs, dim=0)
        weights = torch.cat(all_weights, dim=0)

        return refined, weights, coarse, params_norm


# ============================================================
# v9: NamedCurves-inspired (color naming + Bezier curves + optional attn)
# ============================================================
class _SimpleCrossAttention(nn.Module):
    """Lightweight conv-based local refinement.

    Given y_b (reference image) and an adjusted image, produces a small
    residual that locally refines the adjusted image. Mimics the spatial
    awareness of NamedCurves cross-attention without full attention compute.
    """

    def __init__(self, dim: int = 16):
        super().__init__()
        self.q_proj = nn.Sequential(
            nn.Conv2d(3, dim, 3, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, 2, 1),
        )
        self.k_proj = nn.Sequential(
            nn.Conv2d(3, dim, 3, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, 2, 1),
        )
        self.out = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(dim, 3, 3, 1, 1),
        )
        # Zero-init last conv so refinement starts at 0 (identity-passthrough).
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, y_b: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
        H, W = y_b.shape[-2:]
        Q = self.q_proj(y_b)                 # (B, dim, H/8, W/8)
        K = self.k_proj(img)                 # (B, dim, H/8, W/8)
        feat = Q * K                          # element-wise modulation
        delta_small = self.out(feat)         # (B, 3, H/8, W/8)
        delta = F.interpolate(delta_small, size=(H, W),
                              mode='bilinear', align_corners=False)
        return (img + 0.2 * torch.tanh(delta)).clamp(0.0, 1.0)


class _ContextHead(nn.Module):
    """Predicts a per-pixel spatial context map c(x, y) in [0, 1].

    Inspired by SA-LUT (2024). The context map provides a 4th axis to the
    LUT/Bezier lookup so that the same color in different spatial regions
    can map to different output values, enabling region-aware editing
    (e.g., FireRed's localized WB / saturation adjustments).

    Output: (B, 1, H, W) sigmoid-bounded mask.
    Zero-init last conv → starts at 0.5 (sigmoid(0)=0.5) for stable train.
    """

    def __init__(self, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, hidden, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, 3, 1, 1),
        )
        # Zero-init last conv → ctx_map ≈ 0.5 uniform at start. This is OK
        # because the BCPE branch has asymmetric init for cp_low/cp_high
        # (see NamedCurvesPredictor.__init__), so adj_low ≠ adj_high and
        # gradients flow through ctx_map via the blending formula.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """img: (B, 3, H, W) → context map (B, 1, H, W) in [0, 1]"""
        return torch.sigmoid(self.net(img))


class _ActionContextHead(nn.Module):
    def __init__(self, fused_dim: int, hidden: int = 16):
        super().__init__()
        self.spatial = nn.Sequential(
            nn.Conv2d(3, hidden, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, 3, 1, 1),
        )
        self.mod = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2),
        )
        nn.init.zeros_(self.spatial[-1].weight)
        nn.init.zeros_(self.spatial[-1].bias)
        nn.init.zeros_(self.mod[-1].weight)
        nn.init.zeros_(self.mod[-1].bias)

    def forward(self, img: torch.Tensor, fused: torch.Tensor) -> torch.Tensor:
        logits = self.spatial(img)
        scale, bias = self.mod(fused).chunk(2, dim=1)
        logits = logits * (1.0 + 0.5 * torch.tanh(scale).view(-1, 1, 1, 1))
        logits = logits + 0.5 * torch.tanh(bias).view(-1, 1, 1, 1)
        return torch.sigmoid(logits)


class _WBHead(nn.Module):
    """v9f: Dedicated 3-gain RGB white-balance head.

    Inspired by Deep White-Balance Editing (Afifi & Brown, CVPR 2020),
    which shows that sRGB WB editing is fundamentally a low-dim 3-gain
    multiplicative scaling problem. Our Bezier curves are too general
    to learn this clean linear structure from only 28 wb training samples.

    Architecture:
        fused (feat + action_emb) → MLP → 3 log-gains (Tanh-bounded)
        gains = exp(log_gains * gain_range)  → bounded multiplicative scale
        wb_corrected = (orig * gains).clamp(0, 1)

    Zero-init last layer → log_gains=0 → gains=1 → identity at start.
    Through action_emb conditioning, wb_head learns to be:
        - non-identity for wb action (do actual color shift)
        - identity for other actions (let Bezier handle them)
    """

    def __init__(self, fused_dim: int, gain_range: float = 0.4,
                 dropout: float = 0.5, hidden: int = 64):
        super().__init__()
        self.gain_range = gain_range  # max |log_gain|, e.g. 0.4 → gains in [e^-0.4, e^0.4] ≈ [0.67, 1.49]
        self.head = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Dropout(dropout),
            nn.Linear(fused_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 3),
            nn.Tanh(),  # bounded ±1, scaled by gain_range
        )
        # Zero-init final Linear → log_gains=0 → gains=1 at start
        nn.init.zeros_(self.head[-2].weight)
        nn.init.zeros_(self.head[-2].bias)

    def forward(self, fused: torch.Tensor, orig: torch.Tensor):
        """fused: (B, fused_dim), orig: (B, 3, H, W).
        Returns: wb_corrected (B, 3, H, W), log_gains (B, 3) for viz.
        """
        log_gains = self.head(fused) * self.gain_range          # (B, 3) in [-gain_range, gain_range]
        gains = torch.exp(log_gains)                              # (B, 3) > 0
        wb_corrected = (orig * gains.view(-1, 3, 1, 1)).clamp(0.0, 1.0)
        return wb_corrected, log_gains


class _LearnableColorNaming(nn.Module):
    """v9e: Semantic-aware color naming via HSV warm-start + learned residual.

    Architecture:
        log(HSV_CN_maps)  + small_conv(orig)  →  logits  →  softmax
                          ↑
                  zero-init at start
                  → output ≈ HSV CN (interpretable warm-start)
                  Over training, residual learns semantic deviations
                  (e.g., "blue but for sky", "warm but for face").

    Designed to address wb/saturation actions where the model needs
    pixel-level semantic understanding beyond hue-based grouping.
    """

    def __init__(self, n_colors: int = 3, hidden: int = 16,
                 residual_scale: float = 1.0):
        super().__init__()
        self.n_colors = n_colors
        self.residual_scale = residual_scale
        self.residual = nn.Sequential(
            nn.Conv2d(3, hidden, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, n_colors, 3, 1, 1),
        )
        # Zero-init last conv → no residual at start → masks = HSV CN
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, img: torch.Tensor, hsv_grouping: str = 'compact3'
                ) -> torch.Tensor:
        """img: (B, 3, H, W) in [0, 1] → soft masks (B, n_colors, H, W)."""
        with torch.no_grad():
            hsv_maps = compute_color_naming_maps(img, grouping=hsv_grouping)
            hsv_logits = torch.log(hsv_maps.clamp(min=1e-6))
        delta = self.residual(img)                       # (B, N, H, W)
        combined = hsv_logits + self.residual_scale * delta
        return torch.softmax(combined, dim=1)


class NamedCurvesPredictor(nn.Module):
    """v9: NamedCurves-inspired predictor.

    Architecture:
      orig → encoder → feat (+ optional action_emb fusion)
        feat (+ action_emb) → BCPE → Bezier control points
        orig → color_naming → N probability maps (deterministic, soft HSV)
        For each color group n, per channel c: apply Bezier curve B_{n,c} to orig
        → N globally-adjusted images
        Optional: _SimpleCrossAttention refines each globally-adjusted image
        Weighted blend (threshold CN map at 0.2, renormalize) → final refined
        Optional: 7D param head for interpretability anchor

    Variants (via constructor flags):
      v9a (simplified hybrid): n_colors=3, n_control_points=7,
        use_attention=False, per_action_curves=False, use_7d_anchor=True
      v9b (paper-faithful):    n_colors=6, n_control_points=11,
        use_attention=True,  per_action_curves=False, use_7d_anchor=False
      v9c (per-action):        n_colors=6, n_control_points=11,
        use_attention=True,  per_action_curves=True,  use_7d_anchor=True
    """

    def __init__(self,
                 n_colors: int = 6,
                 n_control_points: int = 11,
                 use_attention: bool = True,
                 per_action_curves: bool = False,
                 use_7d_anchor: bool = False,
                 use_context: bool = False,
                 action_gated_context: bool = False,
                 use_action_context: bool = False,
                 use_region_basis: bool = False,
                 use_region_param_delta: bool = False,
                 use_learned_cn: bool = False,
                 use_wb_head: bool = False,
                 use_nilut_residual: bool = False,
                 use_vera_renderer: bool = False,
                 use_implicit_head: bool = False,
                 implicit_head_base_ch: int = 32,
                 implicit_head_gate_init: float = 0.0,
                 nilut_hidden: int = 32,
                 nilut_n_layers: int = 3,
                 nilut_n_freq: int = 4,
                 nilut_gate_init: float = 1.0,
                 force_nilut_gate: bool = False,
                 vera_latent_dim: int = 32,
                 vera_hidden: int = 64,
                 vera_n_layers: int = 4,
                 vera_gate_init: float = 0.0,
                 image_size: int = 256,
                 visual_dim: int = 384,
                 n_actions: int = 5,
                 dropout: float = 0.5):
        super().__init__()
        self.n_colors = n_colors
        self.n_control_points = n_control_points
        self.use_attention = use_attention
        self.per_action_curves = per_action_curves
        self.use_7d_anchor = use_7d_anchor
        self.use_context = use_context
        self.action_gated_context = action_gated_context
        self.use_action_context = use_action_context
        self.use_region_basis = use_region_basis
        self.use_region_param_delta = use_region_param_delta
        if use_region_basis and use_context:
            raise ValueError('use_region_basis and use_context are mutually exclusive')
        self.use_context_axis = use_context or use_region_basis
        self.n_context_bins = 6 if use_region_basis else (2 if use_context else 1)
        self.use_learned_cn = use_learned_cn
        self.use_wb_head = use_wb_head
        self.use_nilut_residual = use_nilut_residual
        self.use_vera_renderer = use_vera_renderer
        self.use_implicit_head = use_implicit_head
        self.n_actions = n_actions
        assert n_colors in (3, 6), 'n_colors must be 3 (compact3) or 6 (full6)'
        self.cn_grouping = 'full6' if n_colors == 6 else 'compact3'
        context_action_mask = torch.zeros(n_actions, dtype=torch.float32)
        for name in ('contrast', 'shadows', 'highlights'):
            idx = ACTION_TO_IDX.get(name)
            if idx is not None and idx < n_actions:
                context_action_mask[idx] = 1.0
        self.register_buffer('context_action_mask', context_action_mask)
        region_delta_action_mask = torch.zeros(n_actions, 5, dtype=torch.float32)
        delta_idx = {
            'brightness': 0, 'contrast': 1, 'shadows': 2,
            'highlights': 3, 'saturation': 4,
        }
        if ACTION_TO_IDX.get('contrast') is not None:
            region_delta_action_mask[ACTION_TO_IDX['contrast'],
                                     delta_idx['brightness']] = 1.0
            region_delta_action_mask[ACTION_TO_IDX['contrast'],
                                     delta_idx['contrast']] = 1.0
        if ACTION_TO_IDX.get('shadows') is not None:
            region_delta_action_mask[ACTION_TO_IDX['shadows'],
                                     delta_idx['brightness']] = 1.0
            region_delta_action_mask[ACTION_TO_IDX['shadows'],
                                     delta_idx['shadows']] = 1.0
        if ACTION_TO_IDX.get('highlights') is not None:
            region_delta_action_mask[ACTION_TO_IDX['highlights'],
                                     delta_idx['brightness']] = 1.0
            region_delta_action_mask[ACTION_TO_IDX['highlights'],
                                     delta_idx['highlights']] = 1.0
        if ACTION_TO_IDX.get('saturation') is not None:
            region_delta_action_mask[ACTION_TO_IDX['saturation'],
                                     delta_idx['saturation']] = 1.0
        self.register_buffer('region_delta_action_mask',
                             region_delta_action_mask)
        self.register_buffer('region_delta_scales',
                             torch.tensor([0.08, 0.35, 0.15, 0.15, 0.35],
                                          dtype=torch.float32))

        # Shared encoder (same as v8 for fair comparison)
        self.encoder = MobileViTSmall(
            image_size=image_size, output_dim=visual_dim,
            use_se=True, use_fpn=True,
        )

        # Action embedding (shared)
        self.action_emb = nn.Linear(n_actions, 32)
        fused_dim = visual_dim + 32

        # Bezier Control Point Estimator (BCPE)
        # Output shape:
        #   per_action_curves=True:  (n_actions, n_colors, 3, K, M)
        #   per_action_curves=False: (n_colors, 3, K, M)
        # where K = n_context_bins (1 if no context, 2 if v9d context-aware)
        n_curves = n_colors * 3              # one curve per (color, channel)
        cp_per_curve = n_control_points       # raw deltas from identity
        K = self.n_context_bins
        if per_action_curves:
            total_outputs = n_actions * n_curves * K * cp_per_curve
            bcpe_in = visual_dim   # no action_emb concat (use action to index)
        else:
            total_outputs = n_curves * K * cp_per_curve
            bcpe_in = fused_dim   # fused: feat + action_emb

        self.bcpe = nn.Sequential(
            nn.LayerNorm(bcpe_in),
            nn.Dropout(dropout),
            nn.Linear(bcpe_in, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, total_outputs),
            nn.Tanh(),  # bounded delta in [-1, 1]
        )
        # Zero-init final Linear → identity curves at start (stable training)
        nn.init.zeros_(self.bcpe[-2].weight)
        nn.init.zeros_(self.bcpe[-2].bias)

        # v9d: break cp_low / cp_high symmetry via asymmetric bias init.
        # Without this, both bins start at identity → adj_low == adj_high
        # → d_loss/d_ctx ≡ 0 → ContextHead never learns (symmetry trap).
        # We add fixed opposite biases to middle control points so curves
        # diverge: low context bin gets compressive tone curve (slightly
        # darker mid-tones), high context bin gets expansive tone curve
        # (slightly brighter mid-tones). Both endpoints stay at identity.
        if use_context:
            with torch.no_grad():
                bias = self.bcpe[-2].bias
                M = n_control_points
                if per_action_curves:
                    b = bias.view(n_actions, n_colors, 3, 2, M)
                else:
                    b = bias.view(n_colors, 3, 2, M)
                # Asymmetric pattern: middle ctrl pts get +/- 0.3 (→ tanh
                # ≈ ±0.29 → cp shift ≈ ±0.15 from identity).
                # Endpoints (m=0, m=M-1) stay zero so curves remain
                # anchored at 0→0 and 1→1.
                if M >= 3:
                    b[..., 0, 1:M-1] = -0.3   # low context bin (darker)
                    b[..., 1, 1:M-1] = +0.3   # high context bin (brighter)

        # Identity-init curve Y values: linspace 0→1 with M points
        identity_y = torch.linspace(0.0, 1.0, n_control_points)
        self.register_buffer('identity_y', identity_y)

        # Optional 7D param anchor head (for interpretability + MSE supervision)
        if use_7d_anchor:
            self.param_head = nn.Sequential(
                nn.LayerNorm(fused_dim),
                nn.Dropout(dropout),
                nn.Linear(fused_dim, 128),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(128, len(PARAM_NAMES_7D)),
                nn.Tanh(),
            )
            nn.init.zeros_(self.param_head[-2].weight)
            nn.init.zeros_(self.param_head[-2].bias)

        # Optional cross-attention refinement (shared across color groups)
        if use_attention:
            self.attn = _SimpleCrossAttention(dim=16)

        # v9d: per-pixel context head (SA-LUT inspired 4th axis)
        if use_context:
            if use_action_context:
                self.context_head = _ActionContextHead(
                    fused_dim=fused_dim, hidden=16)
            else:
                self.context_head = _ContextHead(hidden=16)

        # v9e: learnable semantic-aware color naming (HSV + residual)
        if use_learned_cn:
            self.learned_cn = _LearnableColorNaming(n_colors=n_colors,
                                                     hidden=16,
                                                     residual_scale=1.0)

        # v9f: dedicated 3-gain WB head (precedes Bezier in forward)
        if use_wb_head:
            self.wb_head = _WBHead(fused_dim=fused_dim, gain_range=0.4,
                                    dropout=dropout, hidden=64)

        if use_region_param_delta:
            self.region_param_head = nn.Sequential(
                nn.LayerNorm(fused_dim),
                nn.Dropout(dropout),
                nn.Linear(fused_dim, 128),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(128, 6 * 5),
                nn.Tanh(),
            )
            nn.init.zeros_(self.region_param_head[-2].weight)
            nn.init.zeros_(self.region_param_head[-2].bias)

        # v10b: NILUT residual color transform applied AFTER Bezier blend.
        # gate_init=0 + zero-init last layer → identity at start, safe drop-in.
        # Inspired by [R6] NILUT (AAAI 2024).
        # Capacity controlled by nilut_hidden / nilut_n_layers / nilut_n_freq
        # so that v10b_planB can run with hidden=64, n_layers=4 etc.
        if use_nilut_residual:
            self.nilut = NILUT(n_freq=nilut_n_freq, hidden=nilut_hidden,
                                n_layers=nilut_n_layers,
                                residual=True, gate_init=nilut_gate_init)
            # Track 2 Path B: optionally freeze gate at its init value so the
            # residual head is forced to always contribute. v10b's gate self-
            # zeroed even when init=1; this prevents that collapse.
            if force_nilut_gate:
                self.nilut.gate.requires_grad_(False)

        # v10c: VeraRetouch-style per-pixel conditional MLP renderer that
        # REPLACES Bezier + CN + attention pipeline. Inspired by [R3]
        # VeraRetouch (arxiv 2604.27375). gate=0 init → identity at start.
        if use_vera_renderer:
            self.vera = VeraRenderer(
                fused_dim=fused_dim, latent_dim=vera_latent_dim,
                rgb_n_freq=4, coord_n_freq=4,
                hidden=vera_hidden, n_layers=vera_n_layers,
                dropout=dropout, residual=True, gate_init=vera_gate_init)

        # Path X: implicit residual head (INRetouch-inspired U-Net).
        # Learns pixel residual between backbone output and FireRed PNG target.
        # Breaks the 7D ISP rendering ceiling for non-ISP edits (WB palette,
        # spatial masks, generative color casts). gate=0 → identity at start.
        if use_implicit_head:
            self.implicit_head = ImplicitResidualHead(
                in_ch=6, base_ch=implicit_head_base_ch,
                n_actions=n_actions, gate_init=implicit_head_gate_init)

    def _compute_curves(self, feat: torch.Tensor, fused: torch.Tensor,
                        action_onehot: torch.Tensor) -> torch.Tensor:
        """Compute Bezier control points.

        Returns:
          if use_context: (B, K=n_context_bins, n_colors, 3, M)
          else:           (B, n_colors, 3, M)
        Initialized to identity (P_m = m/(M-1)); BCPE output is a bounded
        delta added to the identity baseline.
        """
        B = feat.shape[0]
        M = self.n_control_points
        K = self.n_context_bins
        if self.per_action_curves:
            raw = self.bcpe(feat)
            raw = raw.view(B, self.n_actions, self.n_colors, 3, K, M)
            action_idx = action_onehot.argmax(dim=1)              # (B,)
            raw = raw[torch.arange(B, device=raw.device), action_idx]
            # raw: (B, n_colors, 3, K, M)
        else:
            raw = self.bcpe(fused)
            raw = raw.view(B, self.n_colors, 3, K, M)

        # cp = identity + 0.5 * tanh(raw)   (max deviation ±0.5)
        identity = self.identity_y.view(1, 1, 1, 1, M)
        cp = identity + 0.5 * raw
        # Clamp to [0, 1] to keep curves in valid range
        cp = cp.clamp(0.0, 1.0)
        # Permute to (B, K, n_colors, 3, M) for easier per-bin processing
        cp = cp.permute(0, 3, 1, 2, 4).contiguous()
        if not self.use_context_axis:
            # Squeeze the singleton K dim back out for v9a/b/c compatibility
            cp = cp.squeeze(1)  # (B, n_colors, 3, M)
        return cp

    def _compute_region_basis_maps(self, img: torch.Tensor,
                                   cn_maps: torch.Tensor) -> torch.Tensor:
        r, g, b = img[:, 0], img[:, 1], img[:, 2]
        lum = (0.299 * r + 0.587 * g + 0.114 * b).clamp(0.0, 1.0)
        centers = torch.tensor([0.2, 0.5, 0.8], device=img.device,
                               dtype=img.dtype).view(1, 3, 1, 1)
        lum_maps = torch.exp(-0.5 * ((lum.unsqueeze(1) - centers) / 0.18) ** 2)
        lum_maps = lum_maps / (lum_maps.sum(dim=1, keepdim=True) + 1e-8)
        if self.n_colors == 3:
            color_maps = cn_maps
        else:
            warm = cn_maps[:, 0] + cn_maps[:, 3] + cn_maps[:, 4]
            cool = cn_maps[:, 1] + cn_maps[:, 2]
            neutral = cn_maps[:, 5]
            color_maps = torch.stack([warm, cool, neutral], dim=1)
            color_maps = color_maps / (color_maps.sum(dim=1, keepdim=True) + 1e-8)
        maps = torch.cat([lum_maps, color_maps], dim=1)
        return maps / (maps.sum(dim=1, keepdim=True) + 1e-8)

    def _apply_region_param_delta(self, img: torch.Tensor,
                                  fused: torch.Tensor,
                                  region_maps: torch.Tensor,
                                  action_onehot: torch.Tensor) -> torch.Tensor:
        B, _, H, W = img.shape
        delta = self.region_param_head(fused).view(B, 6, 5)
        scales = self.region_delta_scales.to(device=img.device,
                                             dtype=img.dtype)
        action_mask = torch.matmul(
            action_onehot.to(dtype=img.dtype),
            self.region_delta_action_mask.to(device=img.device,
                                             dtype=img.dtype))
        delta = delta * scales.view(1, 1, 5) * action_mask.view(B, 1, 5)
        maps = torch.einsum('bkhw,bkp->bphw', region_maps, delta)
        b_map = maps[:, 0:1]
        c_map = maps[:, 1:2]
        sh_map = maps[:, 2:3]
        hi_map = maps[:, 3:4]
        sat_map = maps[:, 4:5]
        out = (img + b_map).clamp(0.0, 1.0)
        out = ((out - 0.5) * (1.0 + c_map).clamp(0.25, 2.0) + 0.5)
        lum = (0.299 * out[:, 0:1] + 0.587 * out[:, 1:2]
               + 0.114 * out[:, 2:3]).clamp(0.0, 1.0)
        out = out + sh_map * (1.0 - lum) ** 2 + hi_map * lum ** 2
        out = out.clamp(0.0, 1.0)
        gray = (0.299 * out[:, 0:1] + 0.587 * out[:, 1:2]
                + 0.114 * out[:, 2:3])
        out = gray + (out - gray) * (1.0 + sat_map).clamp(0.25, 2.0)
        self._last_region_param_delta = delta.detach()
        return out.clamp(0.0, 1.0)

    def forward(self, enc_input, orig, action_onehot):
        """
        Returns (compatible with v8 train/eval loop):
          refined:     (B, 3, H, W) final output
          cn_avg:      (B, n_colors) batch-spatial-mean CN maps (proxy weights)
          coarse:      (B, 3, H, W) y_b (= orig for now, no learnable backbone)
          params_norm: (B, 7) 7D anchor if enabled, else zeros
        """
        feat = self.encoder(enc_input)                       # (B, visual_dim)
        a_emb = self.action_emb(action_onehot)
        fused = torch.cat([feat, a_emb], dim=1)              # (B, fused_dim)

        # v9f: WB head pre-processing (3-gain RGB scaling).
        # For wb action, this does the main color shift; for other actions,
        # it learns to be identity (gains≈1) through action_emb conditioning.
        # Zero-init last linear → identity at start.
        if self.use_wb_head:
            wb_corrected, log_gains = self.wb_head(fused, orig)
            # Keep gradient on log_gains so v9g WB supervision loss can backprop.
            # Callers in viewer/eval should call .detach() themselves.
            self._last_wb_log_gains = log_gains
        else:
            wb_corrected = orig

        # y_b: standardized intermediate. v9f uses wb_corrected, others use orig
        y_b = wb_corrected

        # v10c: short-circuit — VeraRenderer replaces all Bezier+CN+attn logic.
        # We still allow optional NILUT residual + 7D anchor afterwards.
        if self.use_vera_renderer:
            refined = self.vera(y_b, fused)
            # cn_avg returned as zeros (no CN computed) for forward compat.
            cn_avg = torch.zeros(feat.shape[0], self.n_colors,
                                  device=feat.device, dtype=feat.dtype)
            self._last_vera_gate = self.vera.gate.detach()
            if self.use_nilut_residual:
                refined = self.nilut(refined)
                self._last_nilut_gate = self.nilut.gate.detach()
            if self.use_7d_anchor:
                params_norm = self.param_head(fused)
            else:
                params_norm = torch.zeros(
                    feat.shape[0], len(PARAM_NAMES_7D),
                    device=feat.device, dtype=feat.dtype)
            if self.use_implicit_head:
                refined = self.implicit_head(orig, refined, action_onehot)
                self._last_implicit_gate = self.implicit_head.gate.detach()
            return refined, cn_avg, y_b, params_norm

        # 1. Color naming maps
        if self.use_learned_cn:
            # v9e: learnable semantic CN (HSV warm-start + residual)
            cn_maps = self.learned_cn(y_b, hsv_grouping=self.cn_grouping)
            # cn_maps: (B, n_colors, H, W), each pixel sums to 1 (softmax)
            # Apply threshold + renorm with stop_grad to preserve magnitudes
            cn_thresh = torch.where(cn_maps > 0.2, cn_maps,
                                     torch.zeros_like(cn_maps))
            cn_thresh = cn_thresh / (cn_thresh.sum(dim=1, keepdim=True) + 1e-8)
            self._last_cn_maps = cn_maps.detach()  # for viz
        else:
            # v9a/b/c/d: deterministic HSV CN (no params)
            with torch.no_grad():
                cn_maps = compute_color_naming_maps(
                    y_b, grouping=self.cn_grouping)
                cn_thresh = torch.where(cn_maps > 0.2, cn_maps,
                                         torch.zeros_like(cn_maps))
                cn_thresh = (cn_thresh
                             / (cn_thresh.sum(dim=1, keepdim=True) + 1e-8))

        # 2. Predict Bezier control points
        cp = self._compute_curves(feat, fused, action_onehot)
        # cp: (B, n_colors, 3, M) or (B, K, n_colors, 3, M) when use_context

        # 3. Apply per-color × per-channel Bezier curves to y_b
        if self.use_region_basis:
            region_maps = self._compute_region_basis_maps(y_b, cn_maps)
            self._last_region_basis_maps = region_maps.detach()
            adjusted_bins = []
            for k_idx in range(self.n_context_bins):
                adjusted_k = apply_per_color_per_channel_bezier(
                    y_b, cp[:, k_idx], fix_first_zero=True)
                adjusted_bins.append(adjusted_k)
            adjusted_stack = torch.stack(adjusted_bins, dim=1)
            weights = region_maps.reshape(
                region_maps.shape[0], self.n_context_bins, 1, 1,
                region_maps.shape[2], region_maps.shape[3])
            adjusted = (adjusted_stack * weights).sum(dim=1)
        elif self.use_context:
            # v9d: 2 context bins blended by per-pixel context map
            cp_low = cp[:, 0]   # (B, n_colors, 3, M)
            cp_high = cp[:, 1]  # (B, n_colors, 3, M)
            adj_low = apply_per_color_per_channel_bezier(
                y_b, cp_low, fix_first_zero=True)   # (B, N, 3, H, W)
            adj_high = apply_per_color_per_channel_bezier(
                y_b, cp_high, fix_first_zero=True)
            # Per-pixel context blend
            if self.use_action_context:
                ctx_map = self.context_head(y_b, fused)
            else:
                ctx_map = self.context_head(y_b)
            self._last_context_map = ctx_map.detach()  # save for viz
            c = ctx_map.unsqueeze(1)                  # (B, 1, 1, H, W)
            adjusted_context = (1.0 - c) * adj_low + c * adj_high
            if self.action_gated_context:
                adjusted_global = 0.5 * (adj_low + adj_high)
                mask = self.context_action_mask.to(
                    device=action_onehot.device, dtype=action_onehot.dtype)
                action_gate = (action_onehot * mask.view(1, -1)).sum(dim=1)
                action_gate = action_gate.view(-1, 1, 1, 1, 1)
                self._last_context_action_gate = action_gate.detach()
                adjusted = adjusted_global + action_gate * (
                    adjusted_context - adjusted_global)
            else:
                adjusted = adjusted_context
        else:
            adjusted = apply_per_color_per_channel_bezier(
                y_b, cp, fix_first_zero=True)
        # adjusted: (B, n_colors, 3, H, W)
        adjusted = adjusted.clamp(0.0, 1.0)

        # 4. Optional local refinement via simple cross-attention
        if self.use_attention:
            refined_list = []
            for n_idx in range(self.n_colors):
                img_n = adjusted[:, n_idx]                    # (B, 3, H, W)
                refined_n = self.attn(y_b, img_n)
                refined_list.append(refined_n)
            adjusted = torch.stack(refined_list, dim=1)      # (B, N, 3, H, W)

        # 5. Weighted blend using thresholded CN maps
        refined = (adjusted * cn_thresh.unsqueeze(2)).sum(dim=1)  # (B, 3, H, W)
        refined = refined.clamp(0.0, 1.0)

        if self.use_region_param_delta:
            region_maps = self._compute_region_basis_maps(y_b, cn_maps)
            self._last_region_param_maps = region_maps.detach()
            refined = self._apply_region_param_delta(
                refined, fused, region_maps, action_onehot)

        # 5b. v10b: NILUT residual color transform (off-Planckian, hue rotation,
        # split-toning that 7D ISP / Bezier curves cannot represent).
        # gate=0 at start → identity; gradient ramps up gate via pixel loss.
        if self.use_nilut_residual:
            refined = self.nilut(refined)
            self._last_nilut_gate = self.nilut.gate.detach()

        # 6. Optional 7D anchor predictions (only for MSE supervision)
        if self.use_7d_anchor:
            params_norm = self.param_head(fused)
        else:
            params_norm = torch.zeros(feat.shape[0], len(PARAM_NAMES_7D),
                                       device=feat.device, dtype=feat.dtype)

        # Per-batch CN distribution (proxy for "weights" reporting)
        cn_avg = cn_maps.mean(dim=[2, 3])                    # (B, n_colors)

        # Path X: optional implicit residual head to break 7D ISP ceiling.
        if self.use_implicit_head:
            refined = self.implicit_head(orig, refined, action_onehot)
            self._last_implicit_gate = self.implicit_head.gate.detach()

        return refined, cn_avg, y_b, params_norm


# ============================================================
# Data loading
# ============================================================
def build_data(jsonl_path: Path, val_ratio: float, tier_filter: tuple,
               seed: int, extra_jsonl_paths=None, action_filter=None):
    """加载 JSONL, 过滤 tier, 按 source_image 分 train/val.

    extra_jsonl_paths: optional list of additional JSONL files (e.g. synthetic
    WB augmentation). Samples from these are ALWAYS assigned to train (never
    val) so that the val set stays consistent with the original split.
    A sample is considered "synthetic" if its source_image starts with 'synth-'.
    """
    lines = jsonl_path.read_text(encoding='utf-8').strip().split('\n')
    samples = [json.loads(l) for l in lines]

    extra_samples = []
    if extra_jsonl_paths:
        for ep in extra_jsonl_paths:
            ep_path = Path(ep)
            if not ep_path.exists():
                logger.warning(f'extra jsonl not found: {ep_path}')
                continue
            elines = ep_path.read_text(encoding='utf-8').strip().split('\n')
            ext = [json.loads(l) for l in elines]
            extra_samples.extend(ext)
            logger.info(f'Loaded {len(ext)} extra samples from {ep_path}')

    # Filter by tier
    action_filter = set(action_filter) if action_filter is not None else None
    filtered = [s for s in samples if s.get('quality_tier', '') in tier_filter
                and (action_filter is None or s.get('action') in action_filter)]
    logger.info(f'Loaded {len(samples)} → filtered {len(filtered)} '
                f'(tiers: {tier_filter})')
    filtered_extra = [s for s in extra_samples
                       if s.get('quality_tier', '') in tier_filter
                       and (action_filter is None or s.get('action') in action_filter)]
    if filtered_extra:
        logger.info(f'  + {len(filtered_extra)} extra (synth) samples')

    # Target file existence check
    def _valid(s_list, label):
        out = []
        for s in s_list:
            tp = s['target_path']
            if not Path(tp).is_absolute():
                tp = PROJECT_ROOT / tp
            if Path(tp).exists():
                out.append(s)
            else:
                logger.warning(f'target missing ({label}): {tp}')
        return out

    valid = _valid(filtered, 'main')
    valid_extra = _valid(filtered_extra, 'synth')
    logger.info(f'Valid samples with target: {len(valid)} main + '
                f'{len(valid_extra)} synth')

    # Split MAIN samples by source_image
    img_set = sorted(set(s['source_image'] for s in valid))
    rng = random.Random(seed)
    rng.shuffle(img_set)
    n_val = max(1, int(len(img_set) * val_ratio))
    val_imgs = set(img_set[:n_val])

    train_s = [s for s in valid if s['source_image'] not in val_imgs]
    val_s = [s for s in valid if s['source_image'] in val_imgs]

    # Synthetic samples: always train, never val
    if valid_extra:
        train_s = train_s + valid_extra
        logger.info(f'  appended {len(valid_extra)} synth samples to train')

    logger.info(f'Split: train={len(train_s)}, val={len(val_s)} '
                f'(main {len(img_set)} unique images, val_ratio={val_ratio})')
    return train_s, val_s


# ============================================================
# Metrics
# ============================================================
@torch.no_grad()
def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """PSNR between two (B, 3, H, W) [0, 1] tensors. Returns mean over batch."""
    mse = ((pred - target) ** 2).mean(dim=[1, 2, 3])  # (B,)
    psnr = -10 * torch.log10(mse.clamp(min=1e-10))    # (B,)
    return psnr.mean().item()


# ============================================================
# Train / Eval loops
# ============================================================
class EMA:
    """Exponential Moving Average of model parameters (Plan C).

    Used to smooth training noise; at validation we apply_shadow() to swap
    EMA weights into the model, evaluate, then restore() back to live
    weights for next training step. Standard practice in vision (timm, SWA).
    """
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow: dict = {}
        self.backup: dict = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.data.clone()

    def update(self, model: nn.Module):
        for name, p in model.named_parameters():
            if p.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(
                    p.data, alpha=1.0 - self.decay)

    def apply_shadow(self, model: nn.Module):
        for name, p in model.named_parameters():
            if p.requires_grad and name in self.shadow:
                self.backup[name] = p.data.clone()
                p.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module):
        for name, p in model.named_parameters():
            if p.requires_grad and name in self.backup:
                p.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self):
        return {'decay': self.decay, 'shadow': self.shadow}

    def load_state_dict(self, sd):
        self.decay = sd['decay']
        self.shadow = sd['shadow']


def train_one_epoch(model, loader, opt, device, epoch, args, ema=None):
    model.train()
    # v8 (ParamResidualLUT) and v9 (NamedCurves) share the same 4-tuple
    # forward signature (refined, weights/cn_avg, coarse/y_b, params_norm).
    is_v8 = isinstance(model, (ParamResidualLUTPredictor, NamedCurvesPredictor))
    # v9 only has a real 7D anchor when use_7d_anchor=True;
    # ParamResidualLUT always has one. Guard param_weight loss accordingly.
    has_7d_anchor = (
        isinstance(model, ParamResidualLUTPredictor)
        or (isinstance(model, NamedCurvesPredictor) and model.use_7d_anchor))
    tot_l1, tot_ssim, tot_smooth, tot_mono, tot_loss = 0., 0., 0., 0., 0.
    tot_coarse_psnr = 0.
    tot_psnr, n = 0., 0
    n_bad = 0  # counter for batches with sanitized NaN/Inf grads

    for i, batch in enumerate(loader):
        enc_in = batch['enc_input'].to(device)
        orig = batch['orig'].to(device)
        target = batch['target'].to(device)
        a_oh = batch['action_onehot'].to(device)
        sw = batch['tier_weight'].to(device)
        gt_p7 = batch['params_norm_7d'].to(device)  # used for both
        # supervision (if param_weight>0) and perturbation re-render.

        # === Parameter Perturbation (PerTouch [R16] idea) ===
        # Add Gaussian noise to the *active* action's 7D parameter
        # (others stay 0 because data is per-action), then re-render
        # target via apply_diff_isp. Forces the model to learn the
        # smooth (param -> image) manifold rather than memorizing the
        # exact diff_isp output. Train-only (no eval perturbation).
        if args.param_perturb_sigma > 0:
            with torch.no_grad():
                action_idx = a_oh.argmax(dim=1)                    # (B,)
                idx_map = torch.tensor(
                    ACTION_TO_PARAM_7D_INDICES,
                    device=device, dtype=torch.long)
                per_sample_7d_idx = idx_map[action_idx]            # (B,)
                # mask: 1 at the active param column, 0 elsewhere
                mask = torch.zeros_like(gt_p7)
                mask.scatter_(1, per_sample_7d_idx.unsqueeze(1), 1.0)
                noise = torch.randn_like(gt_p7) * args.param_perturb_sigma
                perturbed_p7 = (gt_p7 + noise * mask).clamp(-1.0, 1.0)
                params_phys = denormalize_params_7d(perturbed_p7)
                new_target = apply_diff_isp(orig, params_phys)
                new_target = torch.nan_to_num(
                    new_target, nan=0.5, posinf=1.0, neginf=0.0
                ).clamp(0.0, 1.0)
                target = new_target
                gt_p7 = perturbed_p7

        if is_v8:
            output, weights, coarse, pred_params = model(enc_in, orig, a_oh)
        else:
            output, weights = model(enc_in, orig, a_oh)

        # Pixel losses (tier-weighted)
        l1_per = (output - target).abs().mean(dim=[1, 2, 3])  # (B,)
        L_l1 = (l1_per * sw).sum() / (sw.sum() + 1e-6)

        L_ssim = ssim_loss(output, target)

        # Regularization (handle shared / per-action / hybrid / v9 named curves)
        if isinstance(model, NamedCurvesPredictor):
            # v9 has no 3D LUT; use a tiny Bezier curve monotonicity prior
            # (encourage Y(P_m) to be non-decreasing with control point index)
            L_smooth = torch.zeros((), device=device)
            L_mono = torch.zeros((), device=device)
        elif isinstance(model, (PerActionLUTPredictor, HybridLUTPredictor,
                                 ParamResidualLUTPredictor)):
            L_smooth, L_mono = lut_reg_loss_per_action(model.action_luts)
        else:
            L_smooth = lut_smoothness_loss(model.lut_basis)
            L_mono = lut_monotonicity_loss(model.lut_basis)

        loss = (args.l1_weight * L_l1
                + args.ssim_weight * L_ssim
                + args.smooth_weight * L_smooth
                + args.mono_weight * L_mono)

        # v8: coarse ISP loss (v9 skips: y_b == orig has no learnable params)
        if isinstance(model, ParamResidualLUTPredictor):
            c_l1 = ((coarse - target).abs().mean(dim=[1, 2, 3]) * sw
                    ).sum() / (sw.sum() + 1e-6)
            c_ssim = ssim_loss(coarse, target)
            loss = loss + args.coarse_weight * (c_l1 + 0.5 * c_ssim)
        # v8/v9: 7D anchor MSE supervision (only when model actually predicts 7D)
        # gt_p7 already loaded (and possibly perturbed) at top of loop.
        if is_v8 and has_7d_anchor and args.param_weight > 0:
            loss = loss + args.param_weight * F.mse_loss(
                pred_params, gt_p7)

        # v9g: explicit WB head supervision (only when model has wb_head + wb action)
        # Pushes WB head's predicted log_gains toward analytical target/orig ratio,
        # forcing specialization on wb action that v9f's unsupervised setup missed.
        if (isinstance(model, NamedCurvesPredictor) and model.use_wb_head
                and args.wb_supervision_weight > 0):
            action_idx = a_oh.argmax(dim=1)            # (B,)
            wb_mask = (action_idx == ACTIONS.index('wb')).float()   # (B,)
            if wb_mask.sum() > 0:
                pred_log_gains = model._last_wb_log_gains.to(device)  # (B, 3)
                # Channel-mean ratio = analytical WB gain for the sample
                tmean = target.mean(dim=[2, 3]).clamp(min=1e-3)  # (B, 3)
                omean = orig.mean(dim=[2, 3]).clamp(min=1e-3)
                target_log_gains = torch.log(tmean / omean).clamp(-1.0, 1.0)
                # MSE only on wb samples, average over channel dim
                wb_loss = (((pred_log_gains - target_log_gains) ** 2).mean(dim=1)
                           * wb_mask).sum() / (wb_mask.sum() + 1e-6)
                loss = loss + args.wb_supervision_weight * wb_loss

        if not torch.isfinite(loss):
            opt.zero_grad()
            continue

        opt.zero_grad()
        loss.backward()
        # Sanitize gradients: replace NaN/Inf with 0 BEFORE clip_grad_norm_.
        # Without this, a single NaN grad poisons the total_norm → all grads
        # become NaN after clipping → all weights become NaN after step.
        bad_grad = False
        for p in model.parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                p.grad = torch.nan_to_num(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
                bad_grad = True
        if bad_grad:
            n_bad += 1
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if ema is not None:
            ema.update(model)

        with torch.no_grad():
            psnr = compute_psnr(output, target)
            if is_v8:
                tot_coarse_psnr += compute_psnr(coarse, target)

        tot_l1 += L_l1.item()
        tot_ssim += L_ssim.item()
        tot_smooth += L_smooth.item()
        tot_mono += L_mono.item()
        tot_loss += loss.item()
        tot_psnr += psnr
        n += 1

        if (i + 1) % args.log_every == 0:
            extra = (f' coarse={tot_coarse_psnr/n:.2f}dB'
                     if is_v8 else '')
            logger.info(
                f'  [{i+1}/{len(loader)}] '
                f'L1={tot_l1/n:.4f} SSIM={tot_ssim/n:.4f} '
                f'smooth={tot_smooth/n:.5f} mono={tot_mono/n:.5f} '
                f'total={tot_loss/n:.4f} PSNR={tot_psnr/n:.2f}dB{extra}')

    return {
        'loss': tot_loss / max(n, 1),
        'l1': tot_l1 / max(n, 1),
        'ssim': tot_ssim / max(n, 1),
        'psnr': tot_psnr / max(n, 1),
        'n_bad': n_bad,
        'n_total': n,
    }


@torch.no_grad()
def evaluate(model, loader, device, args):
    model.eval()
    is_v8 = isinstance(model, (ParamResidualLUTPredictor, NamedCurvesPredictor))
    tot_l1, tot_ssim, tot_psnr, n = 0., 0., 0., 0
    tot_coarse_psnr = 0.
    all_weights = []

    for batch in loader:
        enc_in = batch['enc_input'].to(device)
        orig = batch['orig'].to(device)
        target = batch['target'].to(device)
        a_oh = batch['action_onehot'].to(device)

        if is_v8:
            output, weights, coarse, _ = model(enc_in, orig, a_oh)
        else:
            output, weights = model(enc_in, orig, a_oh)

        l1 = (output - target).abs().mean()
        s = ssim_loss(output, target)
        psnr = compute_psnr(output, target)
        if is_v8:
            tot_coarse_psnr += compute_psnr(coarse, target)

        tot_l1 += l1.item()
        tot_ssim += s.item()
        tot_psnr += psnr
        n += 1
        all_weights.append(weights.cpu())

    # 统计权重分布
    all_w = torch.cat(all_weights, dim=0)  # (N, n_luts)
    w_mean = all_w.mean(dim=0).numpy()

    result = {
        'l1': tot_l1 / max(n, 1),
        'ssim_loss': tot_ssim / max(n, 1),
        'psnr': tot_psnr / max(n, 1),
        'weight_mean': w_mean.tolist(),
    }
    if is_v8:
        result['coarse_psnr'] = tot_coarse_psnr / max(n, 1)
    return result


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jsonl',
                    default='outputs/inverse_fit_pilot/fivek_500_master/'
                            'pseudo_labels.jsonl')
    ap.add_argument('--out_dir', default='checkpoints/lut_v1')
    ap.add_argument('--image_size', type=int, default=256)
    ap.add_argument('--batch_size', type=int, default=4)
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--lut_lr', type=float, default=1e-3,
                    help='LUT parameters 单独学习率 (通常比 encoder 高)')
    ap.add_argument('--weight_decay', type=float, default=1e-3)
    ap.add_argument('--dropout', type=float, default=0.3)
    ap.add_argument('--val_ratio', type=float, default=0.2)
    ap.add_argument('--num_workers', type=int, default=0)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--split_seed', type=int, default=None,
                    help='Data split seed. Defaults to --seed; set to 42 for seed-repeat runs.')
    ap.add_argument('--actions', type=str, nargs='+', default=None,
                    help='Action list/order. Defaults to the legacy 5-action setup; pass brightness and clarity for 7-action training.')
    ap.add_argument('--log_every', type=int, default=10)
    ap.add_argument('--patience', type=int, default=12)
    # LUT config
    ap.add_argument('--n_luts', type=int, default=3)
    ap.add_argument('--lut_dim', type=int, default=33)
    # Loss weights
    ap.add_argument('--l1_weight', type=float, default=1.0)
    ap.add_argument('--ssim_weight', type=float, default=0.5)
    ap.add_argument('--smooth_weight', type=float, default=0.0001)
    ap.add_argument('--mono_weight', type=float, default=0.01)
    # Augmentation
    ap.add_argument('--color_aug', action='store_true',
                    help='同步 brightness/contrast jitter (orig+target)')
    ap.add_argument('--random_crop', action='store_true',
                    help='随机裁剪代替纯 resize')
    # Model variant
    ap.add_argument('--per_action_lut', action='store_true',
                    help='每个 action 独立 LUT 集合')
    ap.add_argument('--hybrid_wb', action='store_true',
                    help='wb 走 explicit 3-gain, 其他 action 走 per-action LUT')
    ap.add_argument('--param_residual_lut', action='store_true',
                    help='v8: 7D ISP 参数 + 残差 LUT 混合模型')
    ap.add_argument('--coarse_weight', type=float, default=0.5,
                    help='v8: coarse ISP output loss weight')
    ap.add_argument('--param_weight', type=float, default=0.1,
                    help='v8: param supervision loss weight (0=disable)')
    # Parameter Perturbation (PerTouch [R16]-inspired regularization).
    # Adds Gaussian noise (in normalized [-1, 1] space) to the active
    # action's 7D ISP parameter at train time, then re-renders the
    # target image via apply_diff_isp. Encourages the model to learn
    # the underlying retouching intent rather than overfitting the exact
    # diff_isp output. Recommended range: 0.02-0.05. 0.0 = disabled.
    ap.add_argument('--param_perturb_sigma', type=float, default=0.0,
                    help='Parameter perturbation Gaussian sigma in '
                         'normalized [-1,1] space (PerTouch-inspired). '
                         '0.0 disables. Recommended: 0.02-0.05.')
    # v9: NamedCurves-inspired
    ap.add_argument('--named_curves', action='store_true',
                    help='v9: color naming + Bezier curves + optional attention')
    ap.add_argument('--nc_n_colors', type=int, default=6, choices=[3, 6],
                    help='v9: number of color groups (3=compact/warm-cool-neutral, '
                         '6=full like paper)')
    ap.add_argument('--nc_n_control_points', type=int, default=11,
                    help='v9: number of Bezier control points per curve')
    ap.add_argument('--nc_use_attention', action='store_true',
                    help='v9: enable simple cross-attention refinement')
    ap.add_argument('--nc_per_action_curves', action='store_true',
                    help='v9: predict separate curve set for each of 5 actions')
    ap.add_argument('--nc_use_7d_anchor', action='store_true',
                    help='v9: add 7D param head as interpretability anchor')
    ap.add_argument('--nc_use_context', action='store_true',
                    help='v9d: SA-LUT inspired per-pixel context map blending'
                         ' (2-bin Bezier curves)')
    ap.add_argument('--nc_action_gated_context', action='store_true',
                    help='v11a: enable context only for contrast/shadows/highlights')
    ap.add_argument('--nc_use_action_context', action='store_true',
                    help='v11d: action-conditioned context map head')
    ap.add_argument('--nc_use_region_basis', action='store_true',
                    help='v11b: use fixed luminance/color region-basis maps'
                         ' instead of learned 1-channel context')
    ap.add_argument('--nc_use_region_param_delta', action='store_true',
                    help='v11c: add region-wise local tone/saturation delta'
                         ' after NamedCurves output')
    ap.add_argument('--nc_use_learned_cn', action='store_true',
                    help='v9e: replace fixed HSV color naming with a learnable'
                         ' semantic mask (HSV warm-start + small conv residual)')
    ap.add_argument('--nc_use_wb_head', action='store_true',
                    help='v9f: add dedicated 3-gain RGB WB head (precedes Bezier)')
    ap.add_argument('--nc_use_nilut_residual', action='store_true',
                    help='v10b: append NILUT residual color transform after'
                         ' Bezier blend (gate=0 init, residual color delta)')
    ap.add_argument('--nilut_hidden', type=int, default=32,
                    help='v10b NILUT hidden dim (Plan B doubles to 64)')
    ap.add_argument('--nilut_n_layers', type=int, default=3,
                    help='v10b NILUT hidden layers (Plan B raises to 4)')
    ap.add_argument('--nilut_n_freq', type=int, default=4,
                    help='v10b NILUT positional encoding frequencies')
    ap.add_argument('--nilut_gate_init', type=float, default=1.0,
                    help='v10b NILUT residual gate init')
    ap.add_argument('--nc_force_nilut_gate', action='store_true',
                    help='Track 2 Path B: freeze NILUT gate at init value '
                         '(prevents v10b-style self-zeroing collapse)')
    ap.add_argument('--vera_latent_dim', type=int, default=32,
                    help='Track 2 Path A: VeraRenderer latent dim '
                         '(v10c=32, Path A bumps to 64)')
    ap.add_argument('--vera_hidden', type=int, default=64,
                    help='Track 2 Path A: VeraRenderer hidden dim '
                         '(v10c=64, Path A bumps to 128)')
    ap.add_argument('--vera_n_layers', type=int, default=4,
                    help='Track 2 Path A: VeraRenderer hidden layers '
                         '(v10c=4, Path A bumps to 6)')
    ap.add_argument('--vera_gate_init', type=float, default=0.0,
                    help='Track 2 Path A: VeraRenderer gate init '
                         '(v10c=0.0, Path A=1.0)')
    ap.add_argument('--nc_use_vera_renderer', action='store_true',
                    help='v10c: REPLACE Bezier+CN+attn with VeraRetouch-style'
                         ' per-pixel conditional MLP renderer (gate=0 init)')
    ap.add_argument('--wb_supervision_weight', type=float, default=0.0,
                    help='v9g: WB head log_gains MSE supervision weight'
                         ' (only fires on wb action; targets log(target_mean/orig_mean))')
    ap.add_argument('--extra_jsonl', type=str, nargs='+', default=None,
                    help='Additional JSONL files to merge as training data'
                         ' (e.g. synthetic WB samples). Always assigned to train.')
    # ===== Path X: Implicit Residual Head (INRetouch-inspired) =====
    ap.add_argument('--use_implicit_head', action='store_true',
                    help='Path X: append U-Net residual head after backbone'
                         ' to break 7D ISP rendering ceiling. Best paired'
                         ' with --backbone_ckpt + --freeze_backbone.')
    ap.add_argument('--implicit_head_base_ch', type=int, default=32,
                    help='Path X: U-Net base channel width (32 ≈ 1.2M params)')
    ap.add_argument('--implicit_head_gate_init', type=float, default=0.0,
                    help='Path X: gate init for residual head (0 = identity)')
    ap.add_argument('--implicit_head_lr', type=float, default=5e-4,
                    help='Path X: separate lr for implicit_head params')
    ap.add_argument('--backbone_ckpt', type=str, default=None,
                    help='Path X: load v11a backbone weights from this ckpt'
                         ' (loads model weights only, does NOT restore opt).')
    ap.add_argument('--freeze_backbone', action='store_true',
                    help='Path X: freeze ALL backbone params, only train'
                         ' implicit_head. Requires --use_implicit_head.')
    ap.add_argument('--resume', type=str, default=None,
                    help='Plan C: path to checkpoint to resume / fine-tune'
                         ' from (loads model + opt + sched if compatible)')
    ap.add_argument('--resume_model_only', action='store_true',
                    help='Only load model weights from --resume; do not restore optimizer/scheduler.')
    ap.add_argument('--ema_decay', type=float, default=0.0,
                    help='Plan C: enable EMA with given decay (0.999 typ.).'
                         ' 0 disables. EMA weights used for validation only.')

    args = ap.parse_args()
    if args.nc_action_gated_context and not args.nc_use_context:
        ap.error('--nc_action_gated_context requires --nc_use_context')
    if args.nc_use_action_context and not args.nc_use_context:
        ap.error('--nc_use_action_context requires --nc_use_context')
    if args.nc_use_region_basis and args.nc_use_context:
        ap.error('--nc_use_region_basis and --nc_use_context are mutually exclusive')
    if args.actions is not None:
        set_actions(args.actions)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    split_seed = args.seed if args.split_seed is None else args.split_seed

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'device={device}, n_luts={args.n_luts}, lut_dim={args.lut_dim}')
    logger.info(f'actions={ACTIONS}')
    logger.info(f'lr={args.lr}, lut_lr={args.lut_lr}, batch={args.batch_size}, '
                f'epochs={args.epochs}')
    logger.info(f'color_aug={args.color_aug}, random_crop={args.random_crop}, '
                f'per_action_lut={args.per_action_lut}, '
                f'hybrid_wb={args.hybrid_wb}, '
                f'param_residual_lut={args.param_residual_lut}')

    # Data
    tier_filter = ('A excellent', 'B good', 'C acceptable')
    train_s, val_s = build_data(
        Path(args.jsonl), args.val_ratio, tier_filter, split_seed,
        extra_jsonl_paths=args.extra_jsonl, action_filter=ACTIONS)

    from collections import Counter
    logger.info(f'train action dist: '
                f'{dict(Counter([s["action"] for s in train_s]))}')
    logger.info(f'val   action dist: '
                f'{dict(Counter([s["action"] for s in val_s]))}')

    train_ds = LUTDataset(train_s, args.image_size, is_train=True,
                           color_aug=args.color_aug,
                           random_crop=args.random_crop)
    val_ds = LUTDataset(val_s, args.image_size, is_train=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers,
                            pin_memory=True)

    # Model
    if args.named_curves:
        model = NamedCurvesPredictor(
            n_colors=args.nc_n_colors,
            n_control_points=args.nc_n_control_points,
            use_attention=args.nc_use_attention,
            per_action_curves=args.nc_per_action_curves,
            use_7d_anchor=args.nc_use_7d_anchor,
            use_context=args.nc_use_context,
            action_gated_context=args.nc_action_gated_context,
            use_action_context=args.nc_use_action_context,
            use_region_basis=args.nc_use_region_basis,
            use_region_param_delta=args.nc_use_region_param_delta,
            use_learned_cn=args.nc_use_learned_cn,
            use_wb_head=args.nc_use_wb_head,
            use_nilut_residual=args.nc_use_nilut_residual,
            use_vera_renderer=args.nc_use_vera_renderer,
            nilut_hidden=args.nilut_hidden,
            nilut_n_layers=args.nilut_n_layers,
            nilut_n_freq=args.nilut_n_freq,
            nilut_gate_init=args.nilut_gate_init,
            force_nilut_gate=args.nc_force_nilut_gate,
            vera_latent_dim=args.vera_latent_dim,
            vera_hidden=args.vera_hidden,
            vera_n_layers=args.vera_n_layers,
            vera_gate_init=args.vera_gate_init,
            use_implicit_head=args.use_implicit_head,
            implicit_head_base_ch=args.implicit_head_base_ch,
            implicit_head_gate_init=args.implicit_head_gate_init,
            image_size=args.image_size, n_actions=len(ACTIONS),
            dropout=args.dropout,
        ).to(device)
        logger.info(
            f'Using NamedCurvesPredictor (v9): '
            f'colors={args.nc_n_colors}, ctrl_pts={args.nc_n_control_points}, '
            f'attn={args.nc_use_attention}, '
            f'per_action={args.nc_per_action_curves}, '
            f'7d_anchor={args.nc_use_7d_anchor}, '
            f'context={args.nc_use_context}, '
            f'action_gated_context={args.nc_action_gated_context}, '
            f'action_context={args.nc_use_action_context}, '
            f'region_basis={args.nc_use_region_basis}, '
            f'region_param_delta={args.nc_use_region_param_delta}, '
            f'learned_cn={args.nc_use_learned_cn}, '
            f'wb_head={args.nc_use_wb_head}, '
            f'nilut_residual={args.nc_use_nilut_residual}, '
            f'vera_renderer={args.nc_use_vera_renderer}')
    elif args.param_residual_lut:
        model = ParamResidualLUTPredictor(
            n_luts=args.n_luts, lut_dim=args.lut_dim,
            image_size=args.image_size, n_actions=len(ACTIONS),
            dropout=args.dropout,
        ).to(device)
        logger.info('Using ParamResidualLUTPredictor '
                    '(7D ISP params + per-action residual LUT)')
    elif args.hybrid_wb:
        model = HybridLUTPredictor(
            n_luts=args.n_luts, lut_dim=args.lut_dim,
            image_size=args.image_size, dropout=args.dropout,
        ).to(device)
        logger.info('Using HybridLUTPredictor (wb=3-gain + others=per-action LUT)')
    elif args.per_action_lut:
        model = PerActionLUTPredictor(
            n_luts=args.n_luts, lut_dim=args.lut_dim,
            image_size=args.image_size, n_actions=len(ACTIONS),
            dropout=args.dropout,
        ).to(device)
        logger.info('Using PerActionLUTPredictor (action-specific LUTs)')
    else:
        model = LUTPredictor(
            n_luts=args.n_luts, lut_dim=args.lut_dim,
            image_size=args.image_size, n_actions=len(ACTIONS),
            dropout=args.dropout,
        ).to(device)

    # ===== Path X: load v11a backbone weights BEFORE freezing/optimizer =====
    if args.backbone_ckpt:
        logger.info(f'[Path X] loading backbone from: {args.backbone_ckpt}')
        bk = torch.load(args.backbone_ckpt, map_location=device,
                         weights_only=False)
        missing, unexpected = model.load_state_dict(
            bk['model_state_dict'], strict=False)
        # implicit_head.* will be in 'missing' (new module, not in v11a ckpt).
        # That is expected; warn only if other params are missing.
        non_head_missing = [m for m in missing
                            if not m.startswith('implicit_head')]
        if non_head_missing:
            logger.warning(
                f'[Path X] backbone load: {len(non_head_missing)} non-head'
                f' params missing (first 3): {non_head_missing[:3]}')
        if unexpected:
            logger.warning(
                f'[Path X] backbone load: {len(unexpected)} unexpected'
                f' params (first 3): {unexpected[:3]}')
        logger.info(
            f'[Path X] backbone loaded'
            f' (val_psnr={bk.get("val_psnr", 0):.2f} @ Ep{bk.get("epoch", "?")})')

    # ===== Path X: freeze backbone if requested =====
    if args.freeze_backbone:
        if not args.use_implicit_head:
            logger.error('--freeze_backbone requires --use_implicit_head')
            sys.exit(1)
        n_frozen = 0
        n_trainable = 0
        for name, p in model.named_parameters():
            if name.startswith('implicit_head'):
                p.requires_grad_(True)
                n_trainable += p.numel()
            else:
                p.requires_grad_(False)
                n_frozen += p.numel()
        logger.info(
            f'[Path X] backbone frozen: {n_frozen/1e6:.2f}M frozen,'
            f' {n_trainable/1e6:.2f}M trainable (implicit_head only)')

    if args.named_curves:
        lut_param_names = ['bcpe']
        if args.nc_use_nilut_residual:
            lut_param_names.append('nilut')
        if args.nc_use_region_param_delta:
            lut_param_names.append('region_param_head')
        lut_param_names = tuple(lut_param_names)
    elif args.param_residual_lut or args.hybrid_wb or args.per_action_lut:
        lut_param_names = ('action_luts',)
    else:
        lut_param_names = ('lut_basis',)
    n_enc = sum(p.numel() for n, p in model.named_parameters()
                if not any(k in n for k in lut_param_names))
    n_lut = sum(p.numel() for n, p in model.named_parameters()
                if any(k in n for k in lut_param_names))
    n_implicit = sum(p.numel() for n, p in model.named_parameters()
                     if n.startswith('implicit_head'))
    logger.info(f'model: encoder+head={n_enc/1e6:.2f}M, '
                f'LUT basis={n_lut/1e3:.1f}K, '
                f'implicit_head={n_implicit/1e6:.2f}M, '
                f'total={( n_enc + n_lut)/1e6:.2f}M')

    # 分组学习率: LUT 参数用更高 lr; implicit_head 单独 lr (Path X)
    lut_params = [p for n, p in model.named_parameters()
                  if any(k in n for k in lut_param_names)
                  and not n.startswith('implicit_head')
                  and p.requires_grad]
    implicit_params = [p for n, p in model.named_parameters()
                       if n.startswith('implicit_head') and p.requires_grad]
    other_params = [p for n, p in model.named_parameters()
                    if not any(k in n for k in lut_param_names)
                    and not n.startswith('implicit_head')
                    and p.requires_grad]
    opt_groups = [
        {'params': other_params, 'lr': args.lr},
        {'params': lut_params, 'lr': args.lut_lr},
    ]
    if implicit_params:
        opt_groups.append({'params': implicit_params,
                           'lr': args.implicit_head_lr})
        logger.info(f'[Path X] optimizer: implicit_head lr='
                    f'{args.implicit_head_lr}, '
                    f'{sum(p.numel() for p in implicit_params)/1e6:.2f}M params')
    opt = optim.AdamW(opt_groups, weight_decay=args.weight_decay)

    sched = optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=args.lr * 0.01)

    # Plan C: optionally resume from a checkpoint (model only by default;
    # opt/sched only if they're available and compatible).
    if args.resume:
        logger.info(f'[resume] loading checkpoint: {args.resume}')
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(
            ck['model_state_dict'], strict=False)
        if missing or unexpected:
            logger.warning(
                f'[resume] missing={len(missing)} unexpected={len(unexpected)}'
                f' (architecture differs). Loaded compatible params only.')
        if args.resume_model_only:
            logger.info('[resume] model-only mode: skip opt/sched state restore')
        elif 'opt_state_dict' in ck:
            try:
                opt.load_state_dict(ck['opt_state_dict'])
                logger.info('[resume] opt state restored')
            except Exception as e:
                logger.warning(f'[resume] opt state load failed: {e}')
        if (not args.resume_model_only) and 'sched_state_dict' in ck:
            try:
                sched.load_state_dict(ck['sched_state_dict'])
                logger.info('[resume] sched state restored')
            except Exception as e:
                logger.warning(f'[resume] sched state load failed: {e}')
        logger.info(
            f'[resume] loaded best_val_psnr={ck.get("val_psnr", 0):.2f}dB'
            f' from Ep{ck.get("epoch", "?")}')

    # Plan C: EMA wrapper (decay=0 disables)
    ema = EMA(model, decay=args.ema_decay) if args.ema_decay > 0 else None
    if ema is not None:
        logger.info(f'[ema] enabled with decay={args.ema_decay}')

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    history = []
    best_psnr = 0.0
    best_epoch = -1
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr = train_one_epoch(model, train_loader, opt, device, epoch, args, ema=ema)
        # Plan C: validate with EMA weights when enabled, then restore live.
        if ema is not None:
            ema.apply_shadow(model)
            val = evaluate(model, val_loader, device, args)
            ema.restore(model)
        else:
            val = evaluate(model, val_loader, device, args)
        sched.step()
        elapsed = time.time() - t0

        coarse_str = (f'  coarse={val["coarse_psnr"]:.2f}dB'
                       if 'coarse_psnr' in val else '')
        bad_str = (f'  bad_grad={tr["n_bad"]}/{tr["n_total"]}'
                   if tr.get('n_bad', 0) > 0 else '')
        logger.info(
            f'Ep {epoch:3d}/{args.epochs}  '
            f'train: L1={tr["l1"]:.4f} PSNR={tr["psnr"]:.2f}dB  '
            f'val: L1={val["l1"]:.4f} PSNR={val["psnr"]:.2f}dB '
            f'SSIM_loss={val["ssim_loss"]:.4f}{coarse_str}{bad_str}  '
            f'w_mean={[f"{w:.2f}" for w in val["weight_mean"]]}  '
            f'{elapsed:.1f}s')

        rec = {
            'epoch': epoch,
            'train': tr,
            'val': val,
            'lr': sched.get_last_lr()[0],
            'elapsed': elapsed,
        }
        history.append(rec)
        json.dump(history, open(out_dir / 'history.json', 'w'),
                  indent=2, default=float)

        if val['psnr'] > best_psnr:
            best_psnr = val['psnr']
            best_epoch = epoch
            no_improve = 0
            # Plan C: save EMA weights as primary if available; live weights
            # also persisted under 'live_state_dict' for resume continuity.
            if ema is not None:
                ema.apply_shadow(model)
                ema_state = {k: v.clone() for k, v in model.state_dict().items()}
                ema.restore(model)
                save_pkg = {
                    'epoch': epoch,
                    'model_state_dict': ema_state,
                    'live_state_dict': model.state_dict(),
                    'ema_state_dict': ema.state_dict(),
                    'opt_state_dict': opt.state_dict(),
                    'sched_state_dict': sched.state_dict(),
                    'val_psnr': val['psnr'],
                    'val_l1': val['l1'],
                    'val': val,
                    'args': vars(args),
                }
            else:
                save_pkg = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'opt_state_dict': opt.state_dict(),
                    'sched_state_dict': sched.state_dict(),
                    'val_psnr': val['psnr'],
                    'val_l1': val['l1'],
                    'val': val,
                    'args': vars(args),
                }
            torch.save(save_pkg, out_dir / 'best.pt')
            logger.info(f'  * best val_psnr={val["psnr"]:.2f}dB  saved')
        else:
            no_improve += 1
            if no_improve >= args.patience:
                logger.info(
                    f'  early stop: no improve for {no_improve} epochs')
                break

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'val': val,
        'args': vars(args),
    }, out_dir / 'last.pt')

    logger.info(f'\n{"="*70}')
    logger.info(f'[DONE] best_val_psnr={best_psnr:.2f}dB @ Ep{best_epoch}')
    logger.info(f'  checkpoint: {out_dir / "best.pt"}')
    logger.info(f'  7D ISP ceiling: ~23.94 dB (for reference)')
    logger.info(f'{"="*70}')


if __name__ == '__main__':
    main()

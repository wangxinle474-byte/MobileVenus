"""Baseline 方法实现，用于论文对比实验。

包含:
- RuleBasedPredictor: 基于规则的参数预测
- RandomPredictor: 随机参数预测
- OraclePredictor: Expert C 参数直接返回 (理论上限)
- StatisticalPredictor: 基于图像统计量的预测
"""

import torch
import numpy as np
from training.fivek_8param.config import PARAM_NAMES, PARAM_RANGES


class RuleBasedPredictor:
    """基于手工规则的参数预测 (Baseline)。

    根据图像亮度、色温等统计量确定参数。
    """

    def predict(self, image):
        """
        Args:
            image: (3, H, W) tensor [0, 1]
        Returns:
            dict: {param_name: value}
        """
        img = image.numpy() if isinstance(image, torch.Tensor) else image

        # 亮度
        lum = 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]
        mean_lum = lum.mean()

        # EV: 目标亮度 0.45
        ev = np.clip((0.45 - mean_lum) * 6.0, -3.0, 3.0)

        # WB: 基于 R/B 比值
        rb_ratio = img[0].mean() / (img[2].mean() + 1e-6)
        wb = np.clip(6500 / rb_ratio, 2000, 10000)

        # 对比度: 基于标准差
        contrast = np.clip((lum.std() - 0.18) * 200, -100, 100)

        # 阴影/高光
        dark_ratio = (lum < 0.2).mean()
        bright_ratio = (lum > 0.8).mean()
        shadows = np.clip(dark_ratio * 100 - 20, -100, 100)
        highlights = np.clip(-bright_ratio * 100 + 20, -100, 100)

        # 饱和度
        hsv_s = 1 - img.min(axis=0) / (img.max(axis=0) + 1e-6)
        saturation = np.clip((hsv_s.mean() - 0.3) * 100, -100, 100)

        return {
            'ev_compensation': float(ev),
            'white_balance': float(wb),
            'contrast': float(contrast),
            'shadows': float(shadows),
            'highlights': float(highlights),
            'saturation': float(saturation),
        }


class RandomPredictor:
    """随机参数预测 (下限 Baseline)。"""

    def __init__(self, seed=42):
        self.rng = np.random.RandomState(seed)

    def predict(self, image=None):
        params = {}
        for name in PARAM_NAMES:
            lo, hi = PARAM_RANGES[name]
            params[name] = float(self.rng.uniform(lo, hi))
        return params


class OraclePredictor:
    """Oracle 预测器: 直接返回 GT 参数 (理论上限)。"""

    def __init__(self, expert_params):
        self.expert_params = expert_params

    def predict(self, image_id):
        if image_id in self.expert_params:
            return self.expert_params[image_id]
        return {name: 0.0 for name in PARAM_NAMES}


class StatisticalPredictor:
    """基于训练集统计量的预测 (均值预测)。"""

    def __init__(self, train_params=None):
        if train_params:
            self.means = {}
            for name in PARAM_NAMES:
                vals = [p[name] for p in train_params.values()
                        if name in p]
                self.means[name] = float(np.mean(vals)) if vals else 0.0
        else:
            self.means = {name: 0.0 for name in PARAM_NAMES}

    def predict(self, image=None):
        return dict(self.means)

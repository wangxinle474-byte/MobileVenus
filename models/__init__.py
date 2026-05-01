"""IntelligenceCamera 模型模块。

核心组件:
- MobileViTSmall: 视觉编码器 (SE + FPN, ~5.6M)
- SemanticProjector: 视觉 → 语义投影 (~130K)
- TextProjector: 文本 → 语义投影 (Stage A teacher)
- LightroomDecoder: 语义 → 6 ISP 参数 (~50K)
- DiffISP: 可微 Lightroom 渲染管线
- RefinementNetV4: 像素级图像精修网络 (~16M)
"""

from .vision_encoder import MobileViTSmall, mobilevit_small
from .semantic_bridge import (
    SemanticProjector,
    TextProjector,
    LightroomDecoder,
    ParameterDecoder,
)
from .diff_isp import apply_diff_isp, ssim_loss
from .refinement_net_v4 import RefinementNetV4
from .isp_pipeline import render_params

__all__ = [
    'MobileViTSmall',
    'mobilevit_small',
    'SemanticProjector',
    'TextProjector',
    'LightroomDecoder',
    'ParameterDecoder',
    'apply_diff_isp',
    'ssim_loss',
    'RefinementNetV4',
    'render_params',
]

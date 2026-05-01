"""FiveK 8参数训练配置。"""

PARAM_NAMES = [
    'ev_compensation', 'white_balance', 'contrast',
    'shadows', 'highlights', 'saturation',
]

# 6 参数范围 (v6+ 移除了 brightness 和 vibrance)
PARAM_RANGES = {
    'ev_compensation': (-3.0, 3.0),
    'white_balance': (2000.0, 10000.0),
    'contrast': (-100.0, 100.0),
    'shadows': (-100.0, 100.0),
    'highlights': (-100.0, 100.0),
    'saturation': (-100.0, 100.0),
}

# 归一化范围 (用于将 GT 参数归一化到 [-1, 1])
PARAM_NORM = {
    'ev_compensation': {'center': 0.0, 'scale': 3.0},
    'white_balance': {'center': 6000.0, 'scale': 4000.0},
    'contrast': {'center': 0.0, 'scale': 100.0},
    'shadows': {'center': 0.0, 'scale': 100.0},
    'highlights': {'center': 0.0, 'scale': 100.0},
    'saturation': {'center': 0.0, 'scale': 100.0},
}

# 训练超参
TRAIN_CONFIG = {
    'image_size': 224,
    'batch_size': 16,
    'lr': 1e-4,
    'weight_decay': 1e-4,
    'epochs': 50,
    'warmup_epochs': 5,
    'grad_clip': 1.0,
    'visual_dim': 384,
    'semantic_dim': 256,
}

# 专家标注
EXPERTS = ['expert_a', 'expert_b', 'expert_c', 'expert_d', 'expert_e']
DEFAULT_EXPERT = 'expert_c'


def normalize_param(name, value):
    """将物理参数归一化到 [-1, 1]。"""
    lo, hi = PARAM_RANGES[name]
    return 2.0 * (value - lo) / (hi - lo) - 1.0


def denormalize_param(name, norm_value):
    """将归一化参数 [-1, 1] 还原到物理范围。"""
    lo, hi = PARAM_RANGES[name]
    return (norm_value + 1.0) / 2.0 * (hi - lo) + lo

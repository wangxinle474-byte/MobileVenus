"""
Venus-Q-Stage1 美学评分服务

调用 Venus 大模型（Qwen-VL-Chat 微调版）对图片进行结构化美学评分。
首次调用时加载模型（约 20GB），之后缓存复用。
"""
import re
import warnings
from pathlib import Path

import torch
from django.conf import settings

# Venus-Q-Stage1 权重路径：APP/backend → MobileVenus → Venus_CVPR2026-main → pretrained_weights
_VENUS_ROOT = Path(settings.BASE_DIR).parent.parent.parent / "pretrained_weights" / "Venus-Q-Stage1"

DIMENSION_NAMES = ["composition", "lighting", "color", "clarity", "subject"]
DIMENSION_LABELS = {
    "composition": "构图",
    "lighting": "光线",
    "color": "色彩",
    "clarity": "清晰度",
    "subject": "主体",
}

_EVAL_PROMPT = (
    "You are a professional photography critic. "
    "Rate this photo on a scale of 1-10 for each dimension. Be strict and precise.\n\n"
    "Please output ONLY the scores in this exact format:\n"
    "Composition: X\n"
    "Lighting: X\n"
    "Color: X\n"
    "Clarity: X\n"
    "Subject: X\n"
    "Overall: X\n\n"
    "Where X is a number from 1 to 10 (can use decimals like 7.5).\n"
    "Do not add any explanation."
)

_model = None
_tokenizer = None


def _find_venus_path() -> str | None:
    """在常见位置搜索 Venus-Q-Stage1 权重目录。"""
    candidates = [
        _VENUS_ROOT,
        Path(getattr(settings, "VENUS_MODEL_PATH", "")),
    ]
    for p in candidates:
        if p and (p / "config.json").exists():
            return str(p)
    return None


def get_model():
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer

    venus_path = _find_venus_path()
    if venus_path is None:
        warnings.warn(
            f"[aesthetic] Venus-Q-Stage1 未找到，路径: {_VENUS_ROOT}，"
            "美学评分将被跳过。可在 dev.py 中设置 VENUS_MODEL_PATH。"
        )
        return None, None

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained(
            venus_path, trust_remote_code=True, local_files_only=True
        )
        _model = AutoModelForCausalLM.from_pretrained(
            venus_path,
            device_map="auto",
            trust_remote_code=True,
            bf16=True,
            local_files_only=True,
        ).eval()
        # transformers >=4.40 的 generate() 默认返回 GenerateOutput(含 DynamicCache)
        # Qwen-VL chat() 期望普通 tensor，强制关闭 dict 返回
        if hasattr(_model, "generation_config"):
            _model.generation_config.return_dict_in_generate = False
        return _model, _tokenizer
    except Exception as exc:
        warnings.warn(f"[aesthetic] 加载 Venus 失败: {exc}")
        return None, None


def _parse_scores(response: str) -> dict:
    """从 Venus 文本输出中提取 6 维评分。"""
    scores: dict[str, float] = {}
    for dim in DIMENSION_NAMES + ["overall"]:
        m = re.search(rf"{dim}\s*[:：]\s*(\d+\.?\d*)", response, re.IGNORECASE)
        if m:
            scores[dim] = min(max(float(m.group(1)), 1.0), 10.0)

    # fallback：按顺序提取 1–10 范围内的数字
    if len(scores) < 6:
        numbers = [float(n) for n in re.findall(r"\d+\.?\d*", response)
                   if 1.0 <= float(n) <= 10.0]
        for j, dim in enumerate(DIMENSION_NAMES + ["overall"]):
            if dim not in scores and j < len(numbers):
                scores[dim] = numbers[j]

    # 最终兜底
    for dim in DIMENSION_NAMES + ["overall"]:
        scores.setdefault(dim, 5.0)

    return scores


def score(image_path: str) -> dict | None:
    """
    对单张图片调用 Venus 打分。

    Returns:
        {"overall": float, "dimensions": {"composition": ..., ...}} 或 None（模型未加载时）
    """
    model, tokenizer = get_model()
    if model is None:
        return None

    try:
        query = tokenizer.from_list_format([
            {"image": image_path},
            {"text": _EVAL_PROMPT},
        ])
        with torch.no_grad():
            response, _ = model.chat(tokenizer, query=query, history=None)

        raw = _parse_scores(response)
        return {
            "overall": round(raw["overall"], 2),
            "dimensions": {dim: round(raw[dim], 2) for dim in DIMENSION_NAMES},
        }
    except Exception as exc:
        warnings.warn(f"[aesthetic] Venus 推理失败: {exc}")
        return None

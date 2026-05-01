import sys
import time
from pathlib import Path
from typing import Dict

import torch
from django.conf import settings

from .exceptions import InferenceFailedException

# 将研究项目根目录加入路径，以便直接复用已有模型代码
_PROJECT_ROOT = Path(settings.BASE_DIR).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

PARAM_NAMES = [
    "ev_compensation",
    "white_balance",
    "contrast",
    "brightness",
    "shadows",
    "highlights",
    "saturation",
    "vibrance",
]


class ModelRegistry:
    _models: Dict[str, torch.nn.Module] = {}

    @classmethod
    def load(cls) -> None:
        import warnings
        checkpoint_map: dict = settings.INFERENCE_CHECKPOINT_MAP
        device: str = settings.INFERENCE_DEVICE

        for version, ckpt_path in checkpoint_map.items():
            if not Path(ckpt_path).exists():
                warnings.warn(f"[ModelRegistry] checkpoint not found, skipping: {ckpt_path}")
                continue
            try:
                if version == "baseline":
                    from training.fivek_8param.model import FiveK8ParamModel
                    model = FiveK8ParamModel()
                else:
                    from training.semantic_distill.model import SemanticDistillModel, DistillParamModel
                    stage_a = SemanticDistillModel()
                    model = DistillParamModel(stage_a_model=stage_a)

                state = torch.load(ckpt_path, map_location=device, weights_only=False)
                model.load_state_dict(state["model_state_dict"])
                model.eval()
                model.to(device)
                cls._models[version] = model
            except Exception as exc:
                warnings.warn(f"[ModelRegistry] failed to load {version}: {exc}")

    @classmethod
    def get(cls, model_version: str = "distill_v4") -> torch.nn.Module:
        if model_version not in cls._models:
            raise InferenceFailedException(
                "2001",
                f"Model '{model_version}' not loaded. Available: {list(cls._models.keys())}",
            )
        return cls._models[model_version]

    @classmethod
    def loaded_versions(cls) -> list:
        return list(cls._models.keys())


def run(tensor: torch.Tensor, model_version: str = "distill_v4") -> tuple[dict, float]:
    model = ModelRegistry.get(model_version)
    device = settings.INFERENCE_DEVICE

    try:
        tensor = tensor.to(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            output = model(tensor)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        raw_params = output["raw_params"]
        params = {
            name: round(float(raw_params[name].squeeze(-1).cpu().item()), 4)
            for name in PARAM_NAMES
        }
        return params, round(elapsed_ms, 1)
    except InferenceFailedException:
        raise
    except Exception as exc:
        raise InferenceFailedException("2001", f"Inference error: {exc}") from exc

import sys
from pathlib import Path
from PIL import Image

from django.conf import settings
from .exceptions import ISPRenderException

_PROJECT_ROOT = Path(settings.BASE_DIR).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def render(original_image_path: str, params: dict) -> Image.Image:
    try:
        from models.isp_pipeline import apply_lightroom_params

        img = Image.open(original_image_path).convert("RGB")
        return apply_lightroom_params(img, params)
    except ISPRenderException:
        raise
    except Exception as exc:
        raise ISPRenderException("2002", f"ISP render error: {exc}") from exc

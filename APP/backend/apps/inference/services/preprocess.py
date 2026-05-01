import torch
from PIL import Image
from torchvision import transforms
from django.conf import settings
from .exceptions import InputValidationException

_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def run(image_path: str) -> torch.Tensor:
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as exc:
        raise InputValidationException("1004", f"Failed to open image: {exc}") from exc

    w, h = img.size
    min_px = getattr(settings, "MIN_IMAGE_SIZE_PX", 64)
    if w < min_px or h < min_px:
        raise InputValidationException("1005", f"Image too small: {w}x{h}, minimum {min_px}px")

    tensor = _transform(img).unsqueeze(0)
    return tensor

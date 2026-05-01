import os
import uuid
from datetime import datetime
from pathlib import Path
from django.conf import settings
from .exceptions import StorageException


def _dated_subdir(base: str) -> Path:
    today = datetime.now().strftime("%Y/%m/%d")
    path = Path(settings.MEDIA_ROOT) / base / today
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_original(file_obj) -> str:
    try:
        subdir = _dated_subdir("uploads")
        ext = Path(file_obj.name).suffix or ".jpg"
        filename = f"{uuid.uuid4().hex}{ext}"
        dest = subdir / filename
        with open(dest, "wb") as f:
            for chunk in file_obj.chunks():
                f.write(chunk)
        rel = os.path.relpath(dest, settings.MEDIA_ROOT)
        return rel.replace("\\", "/")
    except Exception as exc:
        raise StorageException("3001", f"Failed to save original: {exc}") from exc


def save_enhanced(image, original_rel_path: str) -> str:
    try:
        subdir = _dated_subdir("outputs")
        stem = Path(original_rel_path).stem
        filename = f"{stem}_enhanced.jpg"
        dest = subdir / filename
        image.save(str(dest), format="JPEG", quality=95)
        rel = os.path.relpath(dest, settings.MEDIA_ROOT)
        return rel.replace("\\", "/")
    except Exception as exc:
        raise StorageException("3001", f"Failed to save enhanced: {exc}") from exc


def media_url(rel_path: str) -> str:
    base = getattr(settings, "MEDIA_BASE_URL", "").rstrip("/")
    return f"{base}{settings.MEDIA_URL}{rel_path}"

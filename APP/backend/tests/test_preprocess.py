import io
import pytest
import torch
from PIL import Image as PilImage

from apps.inference.services.preprocess import run
from apps.inference.services.exceptions import InputValidationException


def _write_temp_image(tmp_path, width=224, height=224, fmt="JPEG"):
    img = PilImage.new("RGB", (width, height), color=(80, 120, 200))
    path = tmp_path / f"test_{width}x{height}.jpg"
    img.save(str(path), format=fmt)
    return str(path)


class TestPreprocess:
    def test_output_shape(self, tmp_path):
        path = _write_temp_image(tmp_path)
        tensor = run(path)
        assert tensor.shape == (1, 3, 224, 224)

    def test_output_is_float_tensor(self, tmp_path):
        path = _write_temp_image(tmp_path)
        tensor = run(path)
        assert tensor.dtype == torch.float32

    def test_large_image_resized(self, tmp_path):
        path = _write_temp_image(tmp_path, width=1920, height=1280)
        tensor = run(path)
        assert tensor.shape == (1, 3, 224, 224)

    def test_small_image_raises(self, tmp_path, settings):
        settings.MIN_IMAGE_SIZE_PX = 64
        path = _write_temp_image(tmp_path, width=32, height=32)
        with pytest.raises(InputValidationException) as exc_info:
            run(path)
        assert exc_info.value.code == "1005"

    def test_invalid_path_raises(self):
        with pytest.raises(InputValidationException) as exc_info:
            run("/nonexistent/path/image.jpg")
        assert exc_info.value.code == "1004"

    def test_normalization_range(self, tmp_path):
        path = _write_temp_image(tmp_path)
        tensor = run(path)
        # ImageNet 标准化后值域约 [-3, 3]
        assert tensor.min().item() > -5.0
        assert tensor.max().item() < 5.0

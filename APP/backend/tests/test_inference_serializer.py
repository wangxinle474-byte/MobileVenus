import io
import pytest
from PIL import Image as PilImage
from django.core.files.uploadedfile import InMemoryUploadedFile

from apps.inference.serializers import EnhanceImageRequestSerializer


def _make_image_file(width=224, height=224, fmt="JPEG", name="test.jpg"):
    buf = io.BytesIO()
    img = PilImage.new("RGB", (width, height), color=(100, 150, 200))
    img.save(buf, format=fmt)
    buf.seek(0)
    content_type = "image/jpeg" if fmt == "JPEG" else "image/png"
    return InMemoryUploadedFile(
        buf, "file", name, content_type, buf.getbuffer().nbytes, None
    )


class TestEnhanceImageRequestSerializer:
    def test_valid_jpeg(self):
        data = {"model_version": "distill_v4", "client_id": "test_client"}
        files = {"file": _make_image_file()}
        s = EnhanceImageRequestSerializer(data={**data, **files})
        assert s.is_valid(), s.errors

    def test_valid_png(self):
        files = {"file": _make_image_file(fmt="PNG", name="test.png")}
        s = EnhanceImageRequestSerializer(data=files)
        assert s.is_valid(), s.errors

    def test_default_model_version(self):
        files = {"file": _make_image_file()}
        s = EnhanceImageRequestSerializer(data=files)
        assert s.is_valid()
        assert s.validated_data["model_version"] == "distill_v4"

    def test_invalid_model_version(self):
        files = {"file": _make_image_file()}
        s = EnhanceImageRequestSerializer(data={"file": files["file"], "model_version": "unknown_v99"})
        assert not s.is_valid()
        assert "model_version" in s.errors

    def test_missing_file(self):
        s = EnhanceImageRequestSerializer(data={"model_version": "distill_v4"})
        assert not s.is_valid()
        assert "file" in s.errors

    def test_file_too_large(self, settings):
        settings.MAX_UPLOAD_SIZE_MB = 0
        files = {"file": _make_image_file()}
        s = EnhanceImageRequestSerializer(data=files)
        assert not s.is_valid()

    def test_all_model_version_choices(self):
        for version in ("baseline", "distill_v2", "distill_v4"):
            files = {"file": _make_image_file()}
            s = EnhanceImageRequestSerializer(data={"file": files["file"], "model_version": version})
            assert s.is_valid(), f"Failed for version: {version}"

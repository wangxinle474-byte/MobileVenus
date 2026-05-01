import io
import pytest
from unittest.mock import patch, MagicMock
from PIL import Image as PilImage
from django.core.files.uploadedfile import InMemoryUploadedFile
from rest_framework.test import APIClient


def _make_image_file(width=224, height=224, name="photo.jpg"):
    buf = io.BytesIO()
    PilImage.new("RGB", (width, height), color=(80, 120, 200)).save(buf, format="JPEG")
    buf.seek(0)
    return InMemoryUploadedFile(buf, "file", name, "image/jpeg", buf.getbuffer().nbytes, None)


MOCK_PARAMS = {
    "ev_compensation": -0.14,
    "white_balance": 5376.0,
    "contrast": 7.22,
    "brightness": 3.43,
    "shadows": 6.30,
    "highlights": 8.13,
    "saturation": 1.74,
    "vibrance": 6.97,
}

MOCK_META = {
    "model_name": "MobileVenus Distill v4",
    "psnr_reference": "33.11 dB",
    "backbone": "MobileViT-Small (1.93M params)",
    "pipeline": "SemanticDistill + Gamma-aware ISP",
    "inference_time_ms": 120.0,
    "image_width": 224,
    "image_height": 224,
}


@pytest.fixture
def client():
    return APIClient()


@pytest.fixture
def mock_inference_pipeline():
    enhanced_img = PilImage.new("RGB", (224, 224), color=(200, 180, 160))
    with (
        patch("apps.inference.views.preprocess.run", return_value=MagicMock()) as mock_pre,
        patch("apps.inference.views.predictor.run", return_value=(MOCK_PARAMS, 120.0)) as mock_pred,
        patch("apps.inference.views.isp.render", return_value=enhanced_img) as mock_isp,
        patch("apps.inference.views.storage.save_enhanced", return_value="outputs/2026/03/28/test_enhanced.jpg") as mock_save,
        patch("apps.inference.views.storage.media_url", side_effect=lambda p: f"/media/{p}") as mock_url,
    ):
        yield {
            "preprocess": mock_pre,
            "predictor": mock_pred,
            "isp": mock_isp,
            "save_enhanced": mock_save,
            "media_url": mock_url,
        }


@pytest.mark.django_db
class TestEnhanceAPIView:
    def test_success_default_model(self, client, mock_inference_pipeline):
        resp = client.post(
            "/api/v1/inference/enhance/",
            data={"file": _make_image_file()},
            format="multipart",
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["code"] == 0
        assert "enhanced_image_url" in data["data"]
        assert "parameters" in data["data"]
        assert "parameter_confidence" in data["data"]
        assert "meta_info" in data["data"]

    def test_success_with_model_version(self, client, mock_inference_pipeline):
        for version in ("baseline", "distill_v2", "distill_v4"):
            resp = client.post(
                "/api/v1/inference/enhance/",
                data={"file": _make_image_file(), "model_version": version},
                format="multipart",
            )
            assert resp.status_code == 200, f"Failed for version: {version}"
            _, elapsed = mock_inference_pipeline["predictor"].call_args[0]
            assert mock_inference_pipeline["predictor"].called

    def test_missing_file_returns_error(self, client):
        resp = client.post("/api/v1/inference/enhance/", data={}, format="multipart")
        assert resp.status_code == 400
        assert resp.json()["code"] != 0

    def test_invalid_model_version_returns_error(self, client):
        resp = client.post(
            "/api/v1/inference/enhance/",
            data={"file": _make_image_file(), "model_version": "invalid_v99"},
            format="multipart",
        )
        assert resp.status_code == 400

    def test_inference_failure_handled(self, client):
        from apps.inference.services.exceptions import InferenceFailedException
        with (
            patch("apps.inference.views.preprocess.run", return_value=MagicMock()),
            patch("apps.inference.views.predictor.run",
                  side_effect=InferenceFailedException("2001", "model error")),
        ):
            resp = client.post(
                "/api/v1/inference/enhance/",
                data={"file": _make_image_file()},
                format="multipart",
            )
        assert resp.status_code == 422
        assert resp.json()["code"] == 2001

    def test_parameter_confidence_always_present(self, client, mock_inference_pipeline):
        resp = client.post(
            "/api/v1/inference/enhance/",
            data={"file": _make_image_file()},
            format="multipart",
        )
        conf = resp.json()["data"]["parameter_confidence"]
        assert conf["white_balance"] == "reference_only"
        assert conf["shadows"] == "high"

    def test_record_saved_to_db(self, client, mock_inference_pipeline):
        from apps.inference.models import InferenceRecord
        client.post(
            "/api/v1/inference/enhance/",
            data={"file": _make_image_file()},
            format="multipart",
        )
        assert InferenceRecord.objects.filter(status="success").exists()


@pytest.mark.django_db
class TestHealthAPIView:
    def test_health_returns_models_loaded(self, client):
        with patch(
            "apps.inference.views.predictor.ModelRegistry.loaded_versions",
            return_value=["baseline", "distill_v2", "distill_v4"],
        ):
            resp = client.get("/api/v1/health/")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert set(data["models_loaded"]) == {"baseline", "distill_v2", "distill_v4"}
        assert data["default_version"] == "distill_v4"


@pytest.mark.django_db
class TestLatestResultAPIView:
    def test_missing_client_id(self, client):
        resp = client.get("/api/v1/inference/latest/")
        assert resp.status_code == 400

    def test_no_result_found(self, client):
        resp = client.get("/api/v1/inference/latest/?client_id=nonexistent")
        assert resp.status_code == 404

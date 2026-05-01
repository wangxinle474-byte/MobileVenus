import pytest
from unittest.mock import patch, MagicMock

from apps.inference.services.predictor import ModelRegistry
from apps.inference.services.exceptions import InferenceFailedException


class TestModelRegistry:
    def setup_method(self):
        ModelRegistry._models = {}

    def test_loaded_versions_empty_initially(self):
        assert ModelRegistry.loaded_versions() == []

    def test_get_raises_when_not_loaded(self):
        with pytest.raises(InferenceFailedException) as exc_info:
            ModelRegistry.get("distill_v4")
        assert exc_info.value.code == "2001"

    def test_get_invalid_version_raises(self):
        mock_model = MagicMock()
        ModelRegistry._models["distill_v4"] = mock_model
        with pytest.raises(InferenceFailedException):
            ModelRegistry.get("nonexistent")

    def test_get_returns_correct_model(self):
        mock_v2 = MagicMock()
        mock_v4 = MagicMock()
        ModelRegistry._models["distill_v2"] = mock_v2
        ModelRegistry._models["distill_v4"] = mock_v4
        assert ModelRegistry.get("distill_v2") is mock_v2
        assert ModelRegistry.get("distill_v4") is mock_v4

    def test_loaded_versions_returns_all_keys(self):
        ModelRegistry._models = {"baseline": MagicMock(), "distill_v4": MagicMock()}
        versions = ModelRegistry.loaded_versions()
        assert set(versions) == {"baseline", "distill_v4"}

    def test_load_skips_missing_checkpoint(self, settings, tmp_path):
        settings.INFERENCE_CHECKPOINT_MAP = {
            "distill_v4": str(tmp_path / "nonexistent.pt"),
        }
        ModelRegistry.load()
        assert "distill_v4" not in ModelRegistry._models

    def test_load_all_three_versions(self, settings):
        if not all(
            __import__("pathlib").Path(p).exists()
            for p in settings.INFERENCE_CHECKPOINT_MAP.values()
        ):
            pytest.skip("checkpoints not available in this environment")
        ModelRegistry.load()
        assert set(ModelRegistry.loaded_versions()) == {"baseline", "distill_v2", "distill_v4"}

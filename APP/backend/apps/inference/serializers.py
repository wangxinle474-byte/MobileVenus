import os
from PIL import Image as PilImage
from rest_framework import serializers
from django.conf import settings

VALID_CONTENT_TYPES = {"image/jpeg", "image/png"}
VALID_MODEL_VERSIONS = {"baseline", "distill_v2", "distill_v4"}


class EnhanceImageRequestSerializer(serializers.Serializer):
    file = serializers.ImageField()
    model_version = serializers.ChoiceField(
        choices=list(VALID_MODEL_VERSIONS),
        default="distill_v4",
        required=False,
    )
    client_id = serializers.CharField(max_length=64, required=False, default="")

    def validate_file(self, value):
        max_mb = getattr(settings, "MAX_UPLOAD_SIZE_MB", 10)
        if value.size > max_mb * 1024 * 1024:
            raise serializers.ValidationError(
                f"File too large. Max allowed size is {max_mb}MB.",
                code="1003",
            )
        content_type = getattr(value, "content_type", "")
        ext = os.path.splitext(value.name)[-1].lower()
        if content_type not in VALID_CONTENT_TYPES and ext not in (".jpg", ".jpeg", ".png"):
            raise serializers.ValidationError(
                "Only JPEG and PNG files are accepted.",
                code="1002",
            )
        try:
            img = PilImage.open(value)
            img.verify()
            value.seek(0)
        except Exception as exc:
            raise serializers.ValidationError(
                f"Cannot parse image: {exc}",
                code="1004",
            ) from exc
        return value

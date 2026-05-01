from django.db import models
from apps.users.models import MiniProgramUser


class InferenceRecord(models.Model):
    STATUS_CHOICES = [
        ("pending", "Pending"),
        ("success", "Success"),
        ("failed", "Failed"),
    ]

    request_id = models.CharField(max_length=64, unique=True)
    user = models.ForeignKey(
        MiniProgramUser,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="records",
    )

    original_image = models.ImageField(upload_to="uploads/%Y/%m/%d/")
    enhanced_image = models.ImageField(upload_to="outputs/%Y/%m/%d/", null=True, blank=True)

    status = models.CharField(max_length=16, choices=STATUS_CHOICES, default="pending")
    model_version = models.CharField(max_length=32, default="distill_v4")

    parameters = models.JSONField(null=True, blank=True)
    meta_info = models.JSONField(null=True, blank=True)

    AESTHETIC_STATUS = [
        ("pending", "Pending"),
        ("running", "Running"),
        ("done", "Done"),
        ("failed", "Failed"),
        ("unavailable", "Unavailable"),
    ]
    aesthetic_status = models.CharField(
        max_length=16, choices=AESTHETIC_STATUS, default="pending"
    )
    aesthetic_before = models.JSONField(null=True, blank=True)
    aesthetic_after = models.JSONField(null=True, blank=True)

    original_image_data = models.BinaryField(null=True, blank=True)
    enhanced_image_data = models.BinaryField(null=True, blank=True)

    error_code = models.CharField(max_length=16, blank=True, default="")
    error_message = models.CharField(max_length=255, blank=True, default="")

    created_at = models.DateTimeField(auto_now_add=True)
    finished_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        db_table = "inference_record"
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.request_id} [{self.status}]"

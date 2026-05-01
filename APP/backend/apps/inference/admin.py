from django.contrib import admin
from .models import InferenceRecord


@admin.register(InferenceRecord)
class InferenceRecordAdmin(admin.ModelAdmin):
    list_display = ("request_id", "model_version", "status", "created_at", "finished_at")
    list_filter = ("status", "model_version")
    search_fields = ("request_id", "error_code")
    readonly_fields = ("request_id", "created_at", "finished_at")

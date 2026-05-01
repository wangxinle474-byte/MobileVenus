from django.contrib import admin
from .models import MiniProgramUser


@admin.register(MiniProgramUser)
class MiniProgramUserAdmin(admin.ModelAdmin):
    list_display = ("client_id", "openid", "created_at")
    search_fields = ("client_id", "openid")
    readonly_fields = ("created_at",)

from rest_framework import serializers
from .models import MiniProgramUser


class MiniProgramUserSerializer(serializers.ModelSerializer):
    class Meta:
        model = MiniProgramUser
        fields = ("client_id", "created_at")
        read_only_fields = ("created_at",)

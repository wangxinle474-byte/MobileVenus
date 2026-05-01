from django.apps import AppConfig


class InferenceConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "apps.inference"

    def ready(self):
        from .services.predictor import ModelRegistry
        ModelRegistry.load()

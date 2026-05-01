import django
import os

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings.dev")


def pytest_configure():
    from apps.inference.services.predictor import ModelRegistry
    ModelRegistry._models = {}

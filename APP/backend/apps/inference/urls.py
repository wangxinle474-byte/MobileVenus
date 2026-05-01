from django.urls import path
from .views import (
    EnhanceImageAPIView,
    LatestResultAPIView,
    HealthAPIView,
    AestheticScoreAPIView,
    PhotoListAPIView,
    PhotoDetailAPIView,
    PhotoImageAPIView,
)

urlpatterns = [
    path("inference/enhance/", EnhanceImageAPIView.as_view(), name="inference-enhance"),
    path("inference/latest/", LatestResultAPIView.as_view(), name="inference-latest"),
    path("inference/aesthetic/<str:request_id>/", AestheticScoreAPIView.as_view(), name="inference-aesthetic"),
    path("photos/", PhotoListAPIView.as_view(), name="photo-list"),
    path("photos/<int:pk>/", PhotoDetailAPIView.as_view(), name="photo-detail"),
    path("photos/<int:pk>/<str:image_type>/", PhotoImageAPIView.as_view(), name="photo-image"),
    path("health/", HealthAPIView.as_view(), name="health"),
]

from django.urls import path
from .views import RegisterClientView

urlpatterns = [
    path("users/register/", RegisterClientView.as_view(), name="user-register"),
]

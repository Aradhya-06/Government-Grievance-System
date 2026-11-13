from django.contrib.auth.views import LogoutView
from django.urls import path

from . import views

app_name = "accounts"
urlpatterns = [
    path("register/", views.register, name="register"),
    path("login/", views.CitizenLoginView.as_view(), name="login"),
    path("officer-login/", views.OfficerLoginView.as_view(), name="officer_login"),
    path("logout/", LogoutView.as_view(), name="logout"),
    path("go/", views.role_redirect, name="redirect"),
]

from django.urls import path

from . import views

app_name = "grievances"
urlpatterns = [
    path("", views.dashboard, name="dashboard"),
    path("new/", views.submit, name="submit"),
    path("track/", views.track, name="track"),
    path("<str:tracking_id>/submitted/", views.submitted, name="submitted"),
    path("<str:tracking_id>/", views.detail, name="detail"),
]

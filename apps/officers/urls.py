from django.urls import path

from . import views

app_name = "officers"
urlpatterns = [
    path("", views.dashboard, name="dashboard"),
    path("grievance/<int:pk>/", views.manage, name="manage"),
    path("reoptimize/", views.reoptimize, name="reoptimize"),
    path("export.csv", views.export_csv, name="export"),
]

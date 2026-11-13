from django.urls import path

from . import views

app_name = "analytics"
urlpatterns = [
    path("", views.reports, name="reports"),
    path("live.json", views.live_data, name="live"),
    path("playground/", views.playground, name="playground"),
    path("api/fuzzy/", views.api_fuzzy, name="api_fuzzy"),
    path("api/classify/", views.api_classify, name="api_classify"),
    path("admin-overview/", views.admin_overview, name="admin_overview"),
]

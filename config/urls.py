from django.contrib import admin
from django.urls import include, path

admin.site.site_header = "GrievIQ administration"
admin.site.site_title = "GrievIQ admin"
admin.site.index_title = "Manage citizens, officers, departments and grievances"

urlpatterns = [
    path("", include("apps.core.urls")),
    path("accounts/", include("apps.accounts.urls")),
    path("grievances/", include("apps.grievances.urls")),
    path("officer/", include("apps.officers.urls")),
    path("analytics/", include("apps.analytics.urls")),
    path("admin/", admin.site.urls),
]

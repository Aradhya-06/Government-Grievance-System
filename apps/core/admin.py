from django.contrib import admin

from .models import Department


@admin.register(Department)
class DepartmentAdmin(admin.ModelAdmin):
    list_display = ("code", "name", "officers_available", "hours_per_day", "is_active")
    list_filter = ("is_active",)
    search_fields = ("code", "name")

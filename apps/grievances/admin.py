from django.contrib import admin

from .models import ActionHistory, Grievance, QueueRun


class HistoryInline(admin.TabularInline):
    model = ActionHistory
    extra = 0
    readonly_fields = ("created_at",)


@admin.register(Grievance)
class GrievanceAdmin(admin.ModelAdmin):
    list_display = ("tracking_id", "subject", "predicted_category", "department", "status",
                    "priority_score", "queue_position", "submitted_at", "is_demo")
    list_filter = ("status", "department", "predicted_category", "priority_label", "is_demo")
    search_fields = ("tracking_id", "subject", "description", "location")
    readonly_fields = ("tracking_id", "fuzzy_trace", "similar")
    inlines = [HistoryInline]
    date_hierarchy = "submitted_at"


@admin.register(QueueRun)
class QueueRunAdmin(admin.ModelAdmin):
    list_display = ("department", "run_at", "n_items", "ga_cost", "baseline_cost", "runtime_s")
    list_filter = ("department",)

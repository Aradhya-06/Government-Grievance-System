import random
import string

from django.conf import settings
from django.db import models
from django.utils import timezone

PENDING_STATUSES = ("submitted", "under_review", "in_progress")


def generate_tracking_id() -> str:
    """GRV-2026-AB12CD  (unique; retried on the rare collision)."""
    alphabet = string.ascii_uppercase + string.digits
    return f"GRV-{timezone.now().year}-" + "".join(random.choices(alphabet, k=6))


class Grievance(models.Model):
    SUBMITTED, UNDER_REVIEW, IN_PROGRESS, RESOLVED, REJECTED = (
        "submitted", "under_review", "in_progress", "resolved", "rejected")
    STATUS_CHOICES = [
        (SUBMITTED, "Submitted"), (UNDER_REVIEW, "Under review"), (IN_PROGRESS, "In progress"),
        (RESOLVED, "Resolved"), (REJECTED, "Closed - not actionable"),
    ]

    tracking_id = models.CharField(max_length=20, unique=True, default=generate_tracking_id, editable=False)
    citizen = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="grievances")
    subject = models.CharField(max_length=150)
    description = models.TextField()
    location = models.CharField(max_length=120, help_text="Area / ward / city of the issue")
    affected_people = models.PositiveIntegerField(default=1, help_text="Approximate number of people affected")

    # --- ML outputs ---
    predicted_category = models.CharField(max_length=60, blank=True)
    category_confidence = models.FloatField(default=0.0)
    department = models.ForeignKey("core.Department", null=True, on_delete=models.PROTECT,
                                   related_name="grievances")
    similar = models.JSONField(default=list, blank=True)     # [{id, similarity, kind}]
    recurrence_count = models.PositiveIntegerField(default=0)

    # --- Fuzzy / GA outputs ---
    severity = models.FloatField(default=0.0)
    public_impact = models.FloatField(default=0.0)
    sla_days = models.PositiveIntegerField(default=15)
    sla_deadline = models.DateTimeField(null=True, blank=True)
    priority_score = models.FloatField(default=0.0)
    priority_label = models.CharField(max_length=10, default="Low")
    fuzzy_trace = models.JSONField(default=dict, blank=True)  # memberships + fired rules
    queue_position = models.PositiveIntegerField(null=True, blank=True)   # GA-optimised position

    # --- workflow ---
    status = models.CharField(max_length=15, choices=STATUS_CHOICES, default=SUBMITTED, db_index=True)
    submitted_at = models.DateTimeField(default=timezone.now, db_index=True)
    resolved_at = models.DateTimeField(null=True, blank=True)
    is_demo = models.BooleanField(default=False, help_text="Seeded demonstration record")

    class Meta:
        ordering = ["-submitted_at"]
        indexes = [models.Index(fields=["department", "status"])]

    def __str__(self):
        return f"{self.tracking_id} - {self.subject[:40]}"

    @property
    def is_pending(self) -> bool:
        return self.status in PENDING_STATUSES

    @property
    def waiting_days(self) -> float:
        end = self.resolved_at if self.resolved_at else timezone.now()
        return max(0.0, (end - self.submitted_at).total_seconds() / 86400)

    @property
    def days_to_sla(self) -> float | None:
        if not self.sla_deadline:
            return None
        return (self.sla_deadline - timezone.now()).total_seconds() / 86400

    @property
    def is_overdue(self) -> bool:
        d = self.days_to_sla
        return self.is_pending and d is not None and d < 0

    @property
    def priority_css(self) -> str:
        return {"Critical": "crit", "High": "high", "Medium": "med", "Low": "low"}.get(self.priority_label, "low")

    @property
    def status_css(self) -> str:
        return {"submitted": "s-new", "under_review": "s-rev", "in_progress": "s-prog",
                "resolved": "s-done", "rejected": "s-rej"}[self.status]


class ActionHistory(models.Model):
    """Append-only audit trail of every status change / officer remark."""

    grievance = models.ForeignKey(Grievance, on_delete=models.CASCADE, related_name="history")
    actor = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, on_delete=models.SET_NULL)
    from_status = models.CharField(max_length=15, blank=True)
    to_status = models.CharField(max_length=15)
    remarks = models.TextField(blank=True)
    created_at = models.DateTimeField(default=timezone.now)

    class Meta:
        ordering = ["created_at", "id"]
        verbose_name_plural = "action history"

    def __str__(self):
        return f"{self.grievance.tracking_id}: {self.from_status or '-'} -> {self.to_status}"


class QueueRun(models.Model):
    """One GA optimisation of a department queue - shown on the officer dashboard."""

    department = models.ForeignKey("core.Department", on_delete=models.CASCADE, related_name="queue_runs")
    run_at = models.DateTimeField(default=timezone.now)
    n_items = models.PositiveIntegerField()
    generations = models.PositiveIntegerField(default=0)
    runtime_s = models.FloatField(default=0.0)
    ga_cost = models.FloatField(default=0.0)
    baseline_cost = models.FloatField(default=0.0)
    ga_sla_violations = models.PositiveIntegerField(default=0)
    baseline_sla_violations = models.PositiveIntegerField(default=0)
    ga_avg_wait = models.FloatField(default=0.0)
    baseline_avg_wait = models.FloatField(default=0.0)

    class Meta:
        ordering = ["-run_at"]

    @property
    def cost_improvement_pct(self):
        if self.baseline_cost <= 0:
            return 0.0
        return 100.0 * (self.baseline_cost - self.ga_cost) / self.baseline_cost

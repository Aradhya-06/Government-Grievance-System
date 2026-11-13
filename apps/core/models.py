from django.db import models


class Department(models.Model):
    """A government department (demo codes, NOT real CPGRAMS org codes)."""

    code = models.CharField(max_length=20, unique=True)
    name = models.CharField(max_length=120)
    officers_available = models.PositiveSmallIntegerField(
        default=2, help_text="Officers working this department's queue (GA capacity input)")
    hours_per_day = models.FloatField(
        default=6.0, help_text="Productive grievance-handling hours per officer per day")
    is_active = models.BooleanField(default=True)

    class Meta:
        ordering = ["name"]

    def __str__(self):
        return self.name

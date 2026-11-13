import json
from pathlib import Path

from django.conf import settings
from django.db.models import Count, Q
from django.shortcuts import render

from apps.grievances.models import PENDING_STATUSES, Grievance
from ga.fuzzy.engine import RULES


def home(request):
    stats = Grievance.objects.aggregate(total=Count("id"), pending=Count("id", filter=Q(status__in=PENDING_STATUSES)),
                                        resolved=Count("id", filter=Q(status="resolved")))
    return render(request, "core/home.html", {"stats": stats, "n_rules": len(RULES)})


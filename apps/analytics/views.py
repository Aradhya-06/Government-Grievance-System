import json
from pathlib import Path

from django.conf import settings
from django.db.models import Avg, Count, Q
from django.db.models.functions import TruncDate
from django.http import JsonResponse
from django.shortcuts import render
from django.utils import timezone
from django.views.decorators.http import require_GET, require_POST

from apps.accounts.permissions import admin_required, officer_required
from apps.core.models import Department
from apps.grievances.models import PENDING_STATUSES, Grievance, QueueRun
from ga.fuzzy.engine import compute_priority

DS_RESULTS = Path(settings.BASE_DIR) / "ml" / "results"
SC_RESULTS = Path(settings.BASE_DIR) / "ga" / "experiments" / "results"


def _json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


@officer_required
def reports(request):
    """Analytics & reporting: live DB statistics + saved ML / GA results."""
    cls = _json(DS_RESULTS / "classification_metrics.json")
    dup = _json(DS_RESULTS / "duplicate_metrics.json")
    fc = _json(DS_RESULTS / "forecast_metrics.json")
    eda = _json(DS_RESULTS / "eda_summary.json")
    cmp_ = _json(SC_RESULTS / "comparison.json")
    models = sorted(cls["models"].items(), key=lambda kv: -kv[1]["cv_f1_macro_mean"]) if cls else []
    forecast_rows = []
    if fc:
        forecast_rows = [d for d in fc["departments"] if "models" in d]
    summary = []
    if cmp_:
        for name, v in cmp_["summary"].items():
            summary.append(dict(name=name, avg_wait=v["avg_wait"]["mean"], sla=v["sla_viol"]["mean"],
                                sla_rate=100 * v["sla_rate"]["mean"], hp=v["hp_wait"]["mean"],
                                cost=v["cost"]["mean"], runtime=v["runtime"]["mean"]))
    return render(request, "analytics/reports.html", dict(
        cls=cls, models=models, dup=dup, fc=fc, forecast_rows=forecast_rows, eda=eda, cmp=cmp_, summary=summary,
        best_dup=dup["best"] if dup else None))


@officer_required
def live_data(request):
    """JSON feed for the live Chart.js widgets."""
    qs = Grievance.objects.all()
    by_status = {r["status"]: r["n"] for r in qs.values("status").annotate(n=Count("id"))}
    by_cat = list(qs.values("predicted_category").annotate(n=Count("id")).order_by("-n")[:10])
    by_dept = list(Department.objects.annotate(
        pending=Count("grievances", filter=Q(grievances__status__in=PENDING_STATUSES)),
        resolved=Count("grievances", filter=Q(grievances__status="resolved"))).order_by("-pending")[:10]
        .values("name", "pending", "resolved"))
    by_prio = {r["priority_label"]: r["n"] for r in
               qs.filter(status__in=PENDING_STATUSES).values("priority_label").annotate(n=Count("id"))}
    daily = list(qs.annotate(d=TruncDate("submitted_at")).values("d").annotate(n=Count("id")).order_by("d"))
    return JsonResponse(dict(
        by_status=by_status,
        by_category=[dict(label=r["predicted_category"] or "Unclassified", n=r["n"]) for r in by_cat],
        by_department=by_dept, by_priority=by_prio,
        daily=[dict(d=str(r["d"]), n=r["n"]) for r in daily]))


@admin_required
def admin_overview(request):
    now = timezone.now()
    qs = Grievance.objects.all()
    stats = dict(
        total=qs.count(), pending=qs.filter(status__in=PENDING_STATUSES).count(),
        resolved=qs.filter(status="resolved").count(),
        overdue=qs.filter(status__in=PENDING_STATUSES, sla_deadline__lt=now).count(),
        critical=qs.filter(status__in=PENDING_STATUSES, priority_label="Critical").count(),
        citizens=Department.objects.none().count(),
    )
    from django.contrib.auth import get_user_model
    U = get_user_model()
    stats["citizens"] = U.objects.filter(profile__role="citizen").count()
    stats["officers"] = U.objects.filter(profile__role="officer").count()
    stats["demo"] = qs.filter(is_demo=True).count()
    depts = Department.objects.annotate(
        total=Count("grievances"),
        pending=Count("grievances", filter=Q(grievances__status__in=PENDING_STATUSES)),
        resolved=Count("grievances", filter=Q(grievances__status="resolved")),
        overdue=Count("grievances", filter=Q(grievances__status__in=PENDING_STATUSES,
                                             grievances__sla_deadline__lt=now)),
        avg_priority=Avg("grievances__priority_score", filter=Q(grievances__status__in=PENDING_STATUSES)),
    ).order_by("-pending")
    runs = QueueRun.objects.select_related("department")[:8]
    return render(request, "analytics/admin_overview.html", dict(stats=stats, depts=depts, runs=runs))


@officer_required
@require_GET
def api_fuzzy(request):
    """Interactive fuzzy playground: crisp inputs in, explainable priority out."""
    try:
        vals = [float(request.GET.get(k, 0)) for k in ("severity", "impact", "waiting", "sla")]
    except ValueError:
        return JsonResponse({"error": "inputs must be numbers"}, status=400)
    res = compute_priority(*vals)
    return JsonResponse(res.to_dict())


@officer_required
@require_GET
def api_classify(request):
    from ml.classification.predict import classify
    text = request.GET.get("text", "").strip()
    if len(text) < 5:
        return JsonResponse({"error": "Enter at least a short sentence."}, status=400)
    r = classify(text)
    return JsonResponse(dict(category=r.category, confidence=round(r.confidence, 3),
                             top3=[dict(category=c, p=round(p, 3)) for c, p in r.top3], model=r.model_name))


@officer_required
def playground(request):
    return render(request, "analytics/playground.html")

import csv

from django.contrib import messages
from django.core.paginator import Paginator
from django.db.models import Avg, Count, F, Q
from django.http import HttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone
from django.views.decorators.http import require_POST

from apps.accounts.permissions import officer_required
from apps.core.models import Department
from apps.grievances.models import PENDING_STATUSES, ActionHistory, Grievance, QueueRun
from apps.grievances.services.pipeline import reoptimize_department

from .forms import UpdateStatusForm


def _department_for(request):
    """Officers see only their own department; admins pick one with ?dept=<code>."""
    if request.user.is_staff:
        code = request.GET.get("dept") or request.POST.get("dept")
        return (Department.objects.filter(code=code).first() if code else None), True
    prof = request.user.profile
    return prof.department, False


def _base_qs(dept, is_admin):
    qs = Grievance.objects.select_related("department", "citizen")
    if dept:
        return qs.filter(department=dept)
    return qs if is_admin else qs.none()


@officer_required
def dashboard(request):
    dept, is_admin = _department_for(request)
    qs = _base_qs(dept, is_admin)
    pending = qs.filter(status__in=PENDING_STATUSES)

    q = request.GET.get("q", "").strip()
    f_status = request.GET.get("status", "")
    f_cat = request.GET.get("category", "")
    f_prio = request.GET.get("priority", "")
    view = request.GET.get("view", "queue")

    rows = pending if view == "queue" else qs
    if q:
        rows = rows.filter(Q(subject__icontains=q) | Q(tracking_id__icontains=q) |
                           Q(location__icontains=q) | Q(description__icontains=q))
    if f_status:
        rows = rows.filter(status=f_status)
    if f_cat:
        rows = rows.filter(predicted_category=f_cat)
    if f_prio:
        rows = rows.filter(priority_label=f_prio)
    rows = rows.order_by(F("queue_position").asc(nulls_last=True), "-priority_score") if view == "queue" \
        else rows.order_by("-submitted_at")

    now = timezone.now()
    stats = dict(
        pending=pending.count(),
        critical=pending.filter(priority_label="Critical").count(),
        overdue=pending.filter(sla_deadline__lt=now).count(),
        resolved=qs.filter(status="resolved").count(),
        avg_wait=pending.aggregate(a=Avg("priority_score"))["a"] or 0,
    )
    categories = sorted(set(qs.values_list("predicted_category", flat=True)))
    last_run = QueueRun.objects.filter(department=dept).first() if dept else QueueRun.objects.first()
    page = Paginator(rows, 12).get_page(request.GET.get("page"))
    return render(request, "officers/dashboard.html", dict(
        page=page, stats=stats, dept=dept, is_admin=is_admin, view=view, q=q, f_status=f_status,
        f_cat=f_cat, f_prio=f_prio, categories=categories, status_choices=Grievance.STATUS_CHOICES,
        departments=Department.objects.filter(is_active=True) if is_admin else None, last_run=last_run))


@officer_required
def manage(request, pk):
    dept, is_admin = _department_for(request)
    g = get_object_or_404(_base_qs(dept, is_admin), pk=pk)
    form = UpdateStatusForm(request.POST or None, initial={"status": g.status})
    if request.method == "POST" and form.is_valid():
        new = form.cleaned_data["status"]
        old = g.status
        g.status = new
        if new in ("resolved", "rejected"):
            g.resolved_at = timezone.now()
            g.queue_position = None
        else:
            g.resolved_at = None
        g.save(update_fields=["status", "resolved_at", "queue_position"])
        ActionHistory.objects.create(grievance=g, actor=request.user, from_status=old, to_status=new,
                                     remarks=form.cleaned_data["remarks"])
        reoptimize_department(g.department)            # queue changes when a status changes
        messages.success(request, "Status updated and the department queue was re-optimised.")
        return redirect("officers:manage", pk=g.pk)
    similar_ids = [s["id"] for s in g.similar]
    similar = {x.id: x for x in Grievance.objects.filter(id__in=similar_ids).select_related("department")}
    similar_rows = [dict(g=similar[s["id"]], similarity=s["similarity"], kind=s["kind"])
                    for s in g.similar if s["id"] in similar]
    trace = g.fuzzy_trace or {}
    return render(request, "officers/manage.html", dict(
        g=g, form=form, history=g.history.select_related("actor"), similar_rows=similar_rows,
        fired=trace.get("fired_rules", [])[:8], memberships=trace.get("memberships", {}),
        fz_inputs=trace.get("inputs", {})))


@officer_required
@require_POST
def reoptimize(request):
    dept, is_admin = _department_for(request)
    if not dept:
        messages.error(request, "Choose a department first.")
        return redirect("officers:dashboard")
    run = reoptimize_department(dept)
    if run:
        messages.success(request, f"Queue re-optimised: {run.n_items} grievances, {run.generations} generations "
                                  f"in {run.runtime_s:.2f}s.")
    else:
        messages.info(request, "No pending grievances in this department.")
    return redirect(f"{request.META.get('HTTP_REFERER') or '/officer/'}")


@officer_required
def export_csv(request):
    dept, is_admin = _department_for(request)
    qs = _base_qs(dept, is_admin).order_by(F("queue_position").asc(nulls_last=True))
    resp = HttpResponse(content_type="text/csv")
    resp["Content-Disposition"] = 'attachment; filename="grievance_queue.csv"'
    w = csv.writer(resp)
    w.writerow(["Queue position", "Tracking ID", "Subject", "Category", "Department", "Priority score",
                "Priority", "Status", "Submitted", "SLA deadline", "Waiting days", "Location"])
    for g in qs:
        w.writerow([g.queue_position or "", g.tracking_id, g.subject, g.predicted_category, g.department.name,
                    g.priority_score, g.priority_label, g.get_status_display(), g.submitted_at.strftime("%Y-%m-%d"),
                    g.sla_deadline.strftime("%Y-%m-%d") if g.sla_deadline else "", f"{g.waiting_days:.1f}", g.location])
    return resp

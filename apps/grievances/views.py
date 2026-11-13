from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.core.paginator import Paginator
from django.db.models import Count, Q
from django.shortcuts import get_object_or_404, redirect, render

from apps.accounts.permissions import citizen_required

from .forms import GrievanceForm, TrackForm
from .models import PENDING_STATUSES, Grievance
from .services.pipeline import process_new_grievance


@citizen_required
def dashboard(request):
    qs = Grievance.objects.filter(citizen=request.user).select_related("department")
    q = request.GET.get("q", "").strip()
    status = request.GET.get("status", "")
    if q:
        qs = qs.filter(Q(subject__icontains=q) | Q(tracking_id__icontains=q) | Q(description__icontains=q))
    if status:
        qs = qs.filter(status=status)
    counts = Grievance.objects.filter(citizen=request.user).aggregate(
        total=Count("id"), pending=Count("id", filter=Q(status__in=PENDING_STATUSES)),
        resolved=Count("id", filter=Q(status="resolved")))
    page = Paginator(qs, 8).get_page(request.GET.get("page"))
    return render(request, "grievances/dashboard.html",
                  {"page": page, "counts": counts, "q": q, "status": status,
                   "status_choices": Grievance.STATUS_CHOICES})


@citizen_required
def submit(request):
    form = GrievanceForm(request.POST or None)
    if request.method == "POST" and form.is_valid():
        g = form.save(commit=False)
        g.citizen = request.user
        g.save()                      # first save gives it a pk and tracking id
        process_new_grievance(g)      # ML + fuzzy + GA pipeline
        messages.success(request, f"Grievance registered. Your tracking ID is {g.tracking_id}.")
        return redirect("grievances:submitted", tracking_id=g.tracking_id)
    return render(request, "grievances/submit.html", {"form": form})


@citizen_required
def submitted(request, tracking_id):
    g = get_object_or_404(Grievance, tracking_id=tracking_id, citizen=request.user)
    # privacy: a citizen only sees HOW MANY similar complaints exist, never their content
    return render(request, "grievances/submitted.html", {"g": g, "n_similar": len(g.similar)})


@citizen_required
def detail(request, tracking_id):
    g = get_object_or_404(Grievance.objects.select_related("department"), tracking_id=tracking_id,
                          citizen=request.user)      # other citizens' IDs -> 404
    return render(request, "grievances/detail.html", {"g": g, "history": g.history.select_related("actor")})


@citizen_required
def track(request):
    form = TrackForm(request.GET or None)
    if request.GET and form.is_valid():
        tid = form.cleaned_data["tracking_id"]
        if Grievance.objects.filter(tracking_id=tid, citizen=request.user).exists():
            return redirect("grievances:detail", tracking_id=tid)
        form.add_error("tracking_id", "No grievance with this ID was found in your account.")
    return render(request, "grievances/track.html", {"form": form})

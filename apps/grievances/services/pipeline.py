"""
apps/grievances/services/pipeline.py
====================================
THE integration point of the ML, fuzzy and GA components.  Only this module and the views
touch both the ORM and the ML/fuzzy/GA packages; those packages never import Django.

  process_new_grievance(g)     ML -> fuzzy -> GA
  refresh_priority(g)          re-run the fuzzy engine with today's waiting/SLA values
  reoptimize_department(dept)  recompute fuzzy scores for the whole pending queue and
                               run the Genetic Algorithm, then store queue_position

Flow for a new grievance (spec Part 5):
  1. classify text                       (ml.classification.predict)
  2. map category -> department          (common.catalog)
  3. find similar earlier grievances     (ml.similarity.duplicates)
  4. severity / impact / SLA -> fuzzy    (ga.fuzzy.engine)
  5. GA re-orders the department queue   (ga.scheduling.scheduler)
"""

from __future__ import annotations

import logging
from datetime import timedelta

from django.conf import settings
from django.db import transaction
from django.utils import timezone

from apps.core.models import Department
from apps.grievances.models import PENDING_STATUSES, ActionHistory, Grievance, QueueRun
from common import catalog
from ml.classification.predict import classify
from ml.similarity.duplicates import find_similar
from ga.fuzzy.engine import (compute_priority, impact_from, severity_from,
                                         sla_urgency_from)
from ga.scheduling.scheduler import (Capacity, GAConfig, QueueItem, metrics_for_order,
                                         order_by_priority, run_ga)

log = logging.getLogger(__name__)

SIMILAR_POOL_LIMIT = 400   # compare against at most this many recent grievances


def _department(code: str) -> Department:
    dept = Department.objects.filter(code=code).first()
    if dept is None:
        dept = Department.objects.filter(code=catalog.FALLBACK_DEPT).first() or Department.objects.first()
    return dept


def _fuzzy_inputs(g: Grievance, now=None):
    now = now or timezone.now()
    meta = catalog.CATEGORIES.get(g.predicted_category, {})
    elapsed = max(0.0, (now - g.submitted_at).total_seconds() / 86400)
    sla = g.sla_days or meta.get("sla_days", 15)
    hits = catalog.keyword_hits(f"{g.subject} {g.description}")
    sev = severity_from(meta.get("base_severity", 5.0), hits, catalog.KEYWORD_BUMP, catalog.MAX_KEYWORD_BUMP)
    imp = impact_from(g.affected_people, g.recurrence_count)
    return sev, imp, elapsed, sla_urgency_from(elapsed, sla)


def refresh_priority(g: Grievance, save: bool = True) -> Grievance:
    """Recompute the fuzzy priority from current waiting time / SLA consumption."""
    sev, imp, elapsed, sla_u = _fuzzy_inputs(g)
    res = compute_priority(sev, imp, elapsed, sla_u)
    g.severity, g.public_impact = round(sev, 2), round(imp, 2)
    g.priority_score, g.priority_label = round(res.score, 2), res.label
    g.fuzzy_trace = res.to_dict()
    if save:
        g.save(update_fields=["severity", "public_impact", "priority_score", "priority_label", "fuzzy_trace"])
    return g


@transaction.atomic
def process_new_grievance(g: Grievance, optimize: bool = True) -> Grievance:
    """Run the ML + fuzzy + GA pipeline on a freshly created grievance."""
    text = f"{g.subject}. {g.description}"

    # 1-2 classification and department mapping
    cls = classify(text)
    category = cls.category
    g.category_confidence = round(cls.confidence, 3)
    if cls.confidence < catalog.LOW_CONFIDENCE or category not in catalog.CATEGORIES:
        dept = _department(catalog.FALLBACK_DEPT)
        g.predicted_category = category if category in catalog.CATEGORIES else "Unclassified"
    else:
        dept = _department(catalog.department_code_for(category))
        g.predicted_category = category
    g.department = dept
    g.sla_days = catalog.CATEGORIES.get(g.predicted_category, {}).get("sla_days", 15)
    g.sla_deadline = g.submitted_at + timedelta(days=g.sla_days)

    # 3 duplicate / recurring complaint detection (never rejects a complaint)
    pool = (Grievance.objects.exclude(pk=g.pk)
            .order_by("-submitted_at")
            .values_list("id", "subject", "description", "location")[:SIMILAR_POOL_LIMIT])
    corpus = [(i, f"{s}. {d}", loc) for i, s, d, loc in pool]
    matches = find_similar(text, corpus, top_k=5, location=g.location)
    g.similar = [m.__dict__ for m in matches]
    g.recurrence_count = len(matches)

    # 4 fuzzy priority
    refresh_priority(g, save=False)
    g.save()
    ActionHistory.objects.create(grievance=g, actor=g.citizen, from_status="", to_status=g.status,
                                 remarks="Grievance registered. Auto-classified as "
                                         f"{g.predicted_category} ({g.category_confidence:.0%}) and routed to {dept.name}.",
                                 created_at=g.submitted_at)

    # 5 GA queue optimisation for that department (the seeder defers this to the very end)
    if optimize:
        reoptimize_department(dept)
        g.refresh_from_db()
    return g


def reoptimize_department(dept: Department, ga_cfg: GAConfig | None = None) -> QueueRun | None:
    """Recompute fuzzy scores for the pending queue, run the GA, store the order."""
    pending = list(Grievance.objects.filter(department=dept, status__in=PENDING_STATUSES))
    # reset positions of anything that is no longer pending
    Grievance.objects.filter(department=dept).exclude(status__in=PENDING_STATUSES).update(queue_position=None)
    if not pending:
        return None

    now = timezone.now()
    for g in pending:                       # fuzzy scores change as grievances age
        refresh_priority(g, save=False)
    pending.sort(key=lambda x: -x.priority_score)

    cap_n = settings.GRIEVANCE_QUEUE_GA_MAX
    head, tail = pending[:cap_n], pending[cap_n:]
    items = []
    for g in head:
        meta = catalog.CATEGORIES.get(g.predicted_category, {})
        waited = max(0.0, (now - g.submitted_at).total_seconds() / 86400)
        items.append(QueueItem(
            id=g.id, priority=g.priority_score, waited_days=waited,
            deadline_days=(g.sla_deadline - now).total_seconds() / 86400 if g.sla_deadline else 15.0,
            hours=float(meta.get("process_hours", 4)), category=g.predicted_category))
    cap = Capacity(officers=dept.officers_available, hours_per_day=dept.hours_per_day, horizon_days=14.0)
    cfg = ga_cfg or GAConfig(population=50, generations=80, patience=20, seed=None)
    result = run_ga(items, cap, cfg)
    base_order = [items[i].id for i in order_by_priority(items)]
    base = metrics_for_order(base_order, items, cap)

    position = {gid: n for n, gid in enumerate(result.order, start=1)}
    for k, g in enumerate(tail, start=len(head) + 1):
        position[g.id] = k
    for g in pending:
        g.queue_position = position[g.id]
    Grievance.objects.bulk_update(pending, ["severity", "public_impact", "priority_score",
                                            "priority_label", "fuzzy_trace", "queue_position"])
    m = result.metrics
    return QueueRun.objects.create(
        department=dept, n_items=len(items), generations=result.generations_run, runtime_s=result.runtime_s,
        ga_cost=m.cost, baseline_cost=base.cost,
        ga_sla_violations=m.sla_violations, baseline_sla_violations=base.sla_violations,
        ga_avg_wait=m.avg_waiting_days, baseline_avg_wait=base.avg_waiting_days)

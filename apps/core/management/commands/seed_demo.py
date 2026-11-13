"""
python manage.py seed_demo [--grievances 80] [--reset]

Creates DEMONSTRATION data so every screen has something to show:
  * departments (demo codes), one admin, one officer per department, five demo citizens
  * ~80 grievances from the SYNTHETIC complaint generator, back-dated over 45 days and
    pushed through the real pipeline (ML -> duplicates -> fuzzy -> GA)
All of it is flagged is_demo=True.  Credentials are printed at the end - demo only.
"""
import random
from datetime import timedelta

from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand
from django.utils import timezone

from apps.accounts.models import Profile
from apps.core.models import Department
from apps.grievances.models import ActionHistory, Grievance
from apps.grievances.services.pipeline import process_new_grievance, reoptimize_department
from common import catalog
from ml.preprocessing.synthetic_corpus import generate

U = get_user_model()
CITIZENS = [("citizen1", "Aarav Sharma"), ("citizen2", "Diya Verma"), ("citizen3", "Rohan Mehta"),
            ("citizen4", "Ananya Iyer"), ("citizen5", "Kabir Singh")]
OFFICER_PW, CITIZEN_PW, ADMIN_PW = "Officer@123", "Citizen@123", "Admin@123"
REMARKS = {
    "under_review": ["Complaint received and assigned to the concerned section for verification.",
                     "Field inspection has been scheduled."],
    "in_progress": ["Work order issued to the field team.", "Spare parts requisitioned; repair under way.",
                    "Matter taken up with the concerned agency."],
    "resolved": ["Issue fixed and verified by the inspecting officer.", "Resolved; citizen informed over phone.",
                 "Corrective action completed. Closing the grievance."],
}


class Command(BaseCommand):
    help = "Create demo departments, users and grievances (flagged as demo data)."

    def add_arguments(self, p):
        p.add_argument("--grievances", type=int, default=80)
        p.add_argument("--reset", action="store_true", help="delete previous demo grievances first")
        p.add_argument("--seed", type=int, default=2026)

    def handle(self, *a, **o):
        rng = random.Random(o["seed"])
        if o["reset"]:
            Grievance.objects.filter(is_demo=True).delete()
        depts = {}
        for code, name, n in catalog.DEPARTMENTS:
            d, _ = Department.objects.update_or_create(code=code, defaults=dict(name=name, officers_available=n))
            depts[code] = d

        admin, created = U.objects.get_or_create(username="admin", defaults=dict(
            is_staff=True, is_superuser=True, email="admin@example.com", first_name="System Admin"))
        if created:
            admin.set_password(ADMIN_PW); admin.save()
        Profile.objects.get_or_create(user=admin, defaults=dict(role="officer", designation="Administrator"))

        officers = {}
        for code, d in depts.items():
            uname = "officer_" + code.split("-")[1].lower()
            u, c = U.objects.get_or_create(username=uname, defaults=dict(
                email=f"{uname}@example.com", first_name=f"{d.name.split()[0]} Officer"))
            if c:
                u.set_password(OFFICER_PW); u.save()
            Profile.objects.update_or_create(user=u, defaults=dict(
                role="officer", department=d, designation="Section Officer"))
            officers[code] = u

        citizens = []
        for uname, full in CITIZENS:
            u, c = U.objects.get_or_create(username=uname, defaults=dict(
                email=f"{uname}@example.com", first_name=full))
            if c:
                u.set_password(CITIZEN_PW); u.save()
            Profile.objects.get_or_create(user=u, defaults=dict(role="citizen"))
            citizens.append(u)

        rows = generate(n_total=max(o["grievances"], 14), seed=o["seed"] + 5)[: o["grievances"]]
        now = timezone.now()
        made = 0
        for r in rows:
            age = rng.choice([0, 0, 1, 1, 2, 3, 4, 6, 8, 10, 13, 17, 21, 27, 33, 40, 45]) + rng.random()
            g = Grievance.objects.create(
                citizen=rng.choice(citizens),
                subject=f"{r['category']} - {r['location']}",
                description=r["text"], location=r["location"],
                affected_people=int(10 ** rng.uniform(0, 3.6)),
                submitted_at=now - timedelta(days=age), is_demo=True)
            process_new_grievance(g, optimize=False)
            # simulate officers having worked on older complaints
            code = g.department.code
            roll = rng.random()
            if age > 3 and roll < 0.55:
                chain = ["under_review", "in_progress"] + (["resolved"] if roll < 0.35 else [])
                prev, t = g.status, g.submitted_at
                for st in chain:
                    t = t + timedelta(days=rng.uniform(0.3, max(0.5, age / 3)))
                    if t > now:
                        t = now - timedelta(hours=1)
                    ActionHistory.objects.create(grievance=g, actor=officers[code], from_status=prev, to_status=st,
                                                 remarks=rng.choice(REMARKS[st]), created_at=t)
                    prev = st
                g.status = prev
                if prev == "resolved":
                    g.resolved_at = t
                g.save(update_fields=["status", "resolved_at"])
            made += 1
        for d in depts.values():
            reoptimize_department(d)
        self.stdout.write(self.style.SUCCESS(f"Seeded {len(depts)} departments, {made} demo grievances."))
        self.stdout.write("Demo logins (demo data only):")
        self.stdout.write(f"  admin       / {ADMIN_PW}     (Django admin + Admin overview)")
        self.stdout.write(f"  officer_mun / {OFFICER_PW}   (one officer per department: officer_<dept>)")
        self.stdout.write(f"  citizen1    / {CITIZEN_PW}   (citizen1..citizen5)")

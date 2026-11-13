"""Django integration tests: pipeline, RBAC, forms, every page renders."""
from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse

from apps.accounts.models import Profile
from apps.core.models import Department
from apps.grievances.models import ActionHistory, Grievance
from apps.grievances.services.pipeline import process_new_grievance, reoptimize_department
from common import catalog

U = get_user_model()
WATER = ("No water supply in our lane", "There has been no water supply in our lane for five days and many "
         "families are suffering. The pipeline seems broken.")


class Base(TestCase):
    @classmethod
    def setUpTestData(cls):
        for code, name, n in catalog.DEPARTMENTS:
            Department.objects.create(code=code, name=name, officers_available=n)
        cls.c1 = U.objects.create_user("c1", password="pw-12345-x", first_name="Cit One")
        cls.c2 = U.objects.create_user("c2", password="pw-12345-x")
        Profile.objects.create(user=cls.c1, role="citizen"); Profile.objects.create(user=cls.c2, role="citizen")
        cls.off_mun = U.objects.create_user("off_mun", password="pw-12345-x")
        Profile.objects.create(user=cls.off_mun, role="officer", department=Department.objects.get(code="DEMO-MUN"))
        cls.off_pen = U.objects.create_user("off_pen", password="pw-12345-x")
        Profile.objects.create(user=cls.off_pen, role="officer", department=Department.objects.get(code="DEMO-PEN"))
        cls.admin = U.objects.create_superuser("adm", "a@example.com", "pw-12345-x")

    def make(self, user=None, subject=WATER[0], desc=WATER[1], loc="Sector 15", people=200):
        g = Grievance.objects.create(citizen=user or self.c1, subject=subject, description=desc,
                                     location=loc, affected_people=people)
        return process_new_grievance(g)


class PipelineTests(Base):
    def test_full_pipeline_outputs(self):
        g = self.make()
        self.assertEqual(g.predicted_category, "Water Supply")
        self.assertEqual(g.department.code, "DEMO-MUN")
        self.assertTrue(g.tracking_id.startswith("GRV-"))
        self.assertGreater(g.priority_score, 0)
        self.assertIn(g.priority_label, ("Low", "Medium", "High", "Critical"))
        self.assertEqual(g.queue_position, 1)
        self.assertTrue(g.fuzzy_trace["fired_rules"])
        self.assertEqual(g.sla_days, 7)
        self.assertEqual(ActionHistory.objects.filter(grievance=g).count(), 1)

    def test_recurrence_detected_not_rejected(self):
        a = self.make(loc="Sector 15")
        b = self.make(user=self.c2, loc="Sector 15")
        self.assertEqual(Grievance.objects.count(), 2)          # similar complaint is still stored
        self.assertEqual(b.recurrence_count, 1)
        self.assertEqual(b.similar[0]["id"], a.id)
        self.assertEqual(b.similar[0]["kind"], "duplicate")

    def test_queue_positions_are_a_permutation(self):
        for i in range(6):
            self.make(user=self.c1 if i % 2 else self.c2, subject=f"No water supply issue {i}",
                      loc=f"Ward {i}", people=10 ** (i % 4))
        pos = sorted(Grievance.objects.filter(department__code="DEMO-MUN").values_list("queue_position", flat=True))
        self.assertEqual(pos, list(range(1, 7)))

    def test_resolved_leaves_queue(self):
        g = self.make()
        g.status = "resolved"; g.save()
        reoptimize_department(g.department)
        g.refresh_from_db()
        self.assertIsNone(g.queue_position)

    def test_old_grievance_gains_priority(self):
        from datetime import timedelta
        from django.utils import timezone
        g = self.make(subject="Birth certificate has spelling mistake",
                      desc="My birth certificate has a spelling mistake in the name and needs correction soon.", people=1)
        before = g.priority_score
        g.submitted_at = timezone.now() - timedelta(days=40); g.save()
        reoptimize_department(g.department)
        g.refresh_from_db()
        self.assertGreater(g.priority_score, before)


class RBACTests(Base):
    def setUp(self):
        self.g1 = self.make(user=self.c1)

    def test_anonymous_redirected_to_login(self):
        for name in ("grievances:dashboard", "grievances:submit", "officers:dashboard", "analytics:reports"):
            r = self.client.get(reverse(name))
            self.assertEqual(r.status_code, 302, name)
            self.assertIn("/accounts/login/", r["Location"])

    def test_citizen_cannot_see_other_citizens_grievance(self):
        self.client.force_login(self.c2)
        self.assertEqual(self.client.get(reverse("grievances:detail", args=[self.g1.tracking_id])).status_code, 404)
        self.assertEqual(self.client.get(reverse("grievances:submitted", args=[self.g1.tracking_id])).status_code, 404)

    def test_citizen_dashboard_lists_only_own(self):
        self.make(user=self.c2, subject="Pension not credited", desc="My pension has not been credited for three months now.")
        self.client.force_login(self.c1)
        html = self.client.get(reverse("grievances:dashboard")).content.decode()
        self.assertIn(self.g1.tracking_id, html)
        self.assertNotIn("Pension not credited", html)

    def test_citizen_forbidden_from_officer_pages(self):
        self.client.force_login(self.c1)
        self.assertEqual(self.client.get(reverse("officers:dashboard")).status_code, 403)
        self.assertEqual(self.client.get(reverse("analytics:reports")).status_code, 403)

    def test_officer_forbidden_from_citizen_submit_and_admin_overview(self):
        self.client.force_login(self.off_mun)
        self.assertEqual(self.client.get(reverse("grievances:submit")).status_code, 403)
        self.assertEqual(self.client.get(reverse("analytics:admin_overview")).status_code, 403)

    def test_officer_sees_only_own_department(self):
        self.make(user=self.c2, subject="Pension not credited", desc="My pension has not been credited for three months now.")
        self.client.force_login(self.off_mun)
        html = self.client.get(reverse("officers:dashboard")).content.decode()
        self.assertIn(self.g1.tracking_id, html)
        self.assertNotIn("Pension not credited", html)
        other = Grievance.objects.get(subject="Pension not credited")
        self.assertEqual(self.client.get(reverse("officers:manage", args=[other.pk])).status_code, 404)

    def test_officer_login_rejects_citizen_account(self):
        r = self.client.post(reverse("accounts:officer_login"), {"username": "c1", "password": "pw-12345-x"})
        self.assertEqual(r.status_code, 200)
        self.assertContains(r, "not an authorised officer")

    def test_officer_status_update_writes_history_and_requeues(self):
        self.client.force_login(self.off_mun)
        r = self.client.post(reverse("officers:manage", args=[self.g1.pk]),
                             {"status": "resolved", "remarks": "Pipeline repaired and verified on site"})
        self.assertEqual(r.status_code, 302)
        self.g1.refresh_from_db()
        self.assertEqual(self.g1.status, "resolved")
        self.assertIsNotNone(self.g1.resolved_at)
        self.assertIsNone(self.g1.queue_position)
        self.assertEqual(self.g1.history.count(), 2)
        self.assertEqual(self.g1.history.last().actor, self.off_mun)

    def test_closing_requires_explanation(self):
        self.client.force_login(self.off_mun)
        r = self.client.post(reverse("officers:manage", args=[self.g1.pk]), {"status": "resolved", "remarks": "done ok"})
        self.assertEqual(r.status_code, 200)
        self.g1.refresh_from_db()
        self.assertEqual(self.g1.status, "submitted")

    def test_csrf_enforced_on_post(self):
        from django.test import Client
        c = Client(enforce_csrf_checks=True)
        c.force_login(self.c1)
        self.assertEqual(c.post(reverse("grievances:submit"), {"subject": "x"}).status_code, 403)


class FormTests(Base):
    def test_validation_errors(self):
        self.client.force_login(self.c1)
        r = self.client.post(reverse("grievances:submit"), {"subject": "short", "description": "too few", "location": "", "affected_people": 0})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(Grievance.objects.count(), 0)
        self.assertContains(r, "descriptive subject")

    def test_submit_flow_creates_and_redirects(self):
        self.client.force_login(self.c1)
        r = self.client.post(reverse("grievances:submit"), {"subject": WATER[0], "description": WATER[1],
                                                           "location": "Sector 15", "affected_people": 150})
        g = Grievance.objects.get()
        self.assertRedirects(r, reverse("grievances:submitted", args=[g.tracking_id]))
        self.assertEqual(g.citizen, self.c1)
        self.assertEqual(g.predicted_category, "Water Supply")

    def test_registration_creates_citizen_profile(self):
        r = self.client.post(reverse("accounts:register"), {
            "username": "newcit", "first_name": "New Cit", "email": "new@example.com", "phone": "9876543210",
            "password1": "Str0ng-Pass-77", "password2": "Str0ng-Pass-77"})
        self.assertEqual(r.status_code, 302)
        self.assertEqual(U.objects.get(username="newcit").profile.role, "citizen")

    def test_registration_rejects_bad_phone_and_duplicate_email(self):
        r = self.client.post(reverse("accounts:register"), {
            "username": "x1", "first_name": "X", "email": "c1@example.com", "phone": "12345",
            "password1": "Str0ng-Pass-77", "password2": "Str0ng-Pass-77"})
        self.assertEqual(r.status_code, 200)
        self.assertFalse(U.objects.filter(username="x1").exists())

    def test_track_other_users_id_not_found(self):
        g = self.make(user=self.c2)
        self.client.force_login(self.c1)
        r = self.client.get(reverse("grievances:track"), {"tracking_id": g.tracking_id})
        self.assertContains(r, "No grievance with this ID")


class PageRenderTests(Base):
    def setUp(self):
        self.g = self.make()

    def test_public_pages(self):
        for name in ("core:home", "accounts:login", "accounts:register", "accounts:officer_login"):
            self.assertEqual(self.client.get(reverse(name)).status_code, 200, name)

    def test_citizen_pages(self):
        self.client.force_login(self.c1)
        for url in (reverse("grievances:dashboard"), reverse("grievances:submit"), reverse("grievances:track"),
                    reverse("grievances:detail", args=[self.g.tracking_id]),
                    reverse("grievances:submitted", args=[self.g.tracking_id])):
            self.assertEqual(self.client.get(url).status_code, 200, url)

    def test_officer_and_analytics_pages(self):
        self.client.force_login(self.off_mun)
        for url in (reverse("officers:dashboard"), reverse("officers:dashboard") + "?view=all&q=water&priority=Critical",
                    reverse("officers:manage", args=[self.g.pk]), reverse("officers:export"),
                    reverse("analytics:reports"), reverse("analytics:playground"), reverse("analytics:live")):
            self.assertEqual(self.client.get(url).status_code, 200, url)

    def test_admin_pages_and_apis(self):
        self.client.force_login(self.admin)
        self.assertEqual(self.client.get(reverse("analytics:admin_overview")).status_code, 200)
        self.assertEqual(self.client.get(reverse("officers:dashboard") + "?dept=DEMO-MUN").status_code, 200)
        self.assertEqual(self.client.get("/admin/").status_code, 200)
        r = self.client.get(reverse("analytics:api_fuzzy"), {"severity": 9, "impact": 9, "waiting": 30, "sla": 9})
        self.assertEqual(r.json()["label"], "Critical")
        self.assertEqual(self.client.get(reverse("analytics:api_fuzzy"), {"severity": "abc"}).status_code, 400)
        r = self.client.get(reverse("analytics:api_classify"), {"text": "my pension has not been credited"})
        self.assertEqual(r.json()["category"], "Pension")
        self.assertEqual(self.client.get(reverse("analytics:api_classify"), {"text": "hi"}).status_code, 400)

    def test_csv_export_contents(self):
        self.client.force_login(self.off_mun)
        body = self.client.get(reverse("officers:export")).content.decode()
        self.assertIn(self.g.tracking_id, body)
        self.assertTrue(body.startswith("Queue position"))

    def test_logout_requires_post(self):
        self.client.force_login(self.c1)
        self.assertEqual(self.client.get(reverse("accounts:logout")).status_code, 405)
        self.assertEqual(self.client.post(reverse("accounts:logout")).status_code, 302)

    def test_404_page_renders(self):
        self.client.force_login(self.c1)
        self.assertEqual(self.client.get("/grievances/GRV-NOPE/").status_code, 404)

from django.conf import settings
from django.db import models


class Profile(models.Model):
    """
    Extends Django's User with a role.  Citizen and Officer are both Profiles
    (role field); administrators are Django staff users.  An officer belongs to
    exactly one Department.
    """

    CITIZEN, OFFICER = "citizen", "officer"
    ROLES = [(CITIZEN, "Citizen"), (OFFICER, "Government Officer")]

    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="profile")
    role = models.CharField(max_length=10, choices=ROLES, default=CITIZEN)
    phone = models.CharField(max_length=15, blank=True)
    department = models.ForeignKey("core.Department", null=True, blank=True, on_delete=models.SET_NULL,
                                   related_name="officers")
    designation = models.CharField(max_length=80, blank=True)

    def __str__(self):
        return f"{self.user.get_username()} ({self.get_role_display()})"

    @property
    def is_officer(self):
        return self.role == self.OFFICER

from django.conf import settings


def site(request):
    role = None
    if request.user.is_authenticated:
        if request.user.is_staff:
            role = "admin"
        else:
            prof = getattr(request.user, "profile", None)
            role = prof.role if prof else None
    return {"SITE_NAME": settings.SITE_NAME, "SITE_TAGLINE": settings.SITE_TAGLINE, "user_role": role}

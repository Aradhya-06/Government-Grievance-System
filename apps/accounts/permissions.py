"""Role-based access control decorators ."""
from functools import wraps

from django.contrib import messages
from django.contrib.auth.views import redirect_to_login
from django.core.exceptions import PermissionDenied


def _role(user):
    if user.is_staff:
        return "admin"
    prof = getattr(user, "profile", None)
    return prof.role if prof else None


def role_required(*roles):
    """Allow only users whose role is in `roles`; anonymous users go to the login page."""
    def deco(view):
        @wraps(view)
        def wrapper(request, *a, **kw):
            if not request.user.is_authenticated:
                return redirect_to_login(request.get_full_path())
            if _role(request.user) not in roles:
                raise PermissionDenied("Your account role is not allowed to open this page.")
            return view(request, *a, **kw)
        return wrapper
    return deco


citizen_required = role_required("citizen")
officer_required = role_required("officer", "admin")
admin_required = role_required("admin")

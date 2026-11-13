from django.contrib import messages
from django.contrib.auth import login
from django.contrib.auth.views import LoginView
from django.shortcuts import redirect, render
from django.urls import reverse_lazy
from django.views.decorators.http import require_http_methods

from .forms import CitizenRegistrationForm
from .permissions import _role


@require_http_methods(["GET", "POST"])
def register(request):
    if request.user.is_authenticated:
        return redirect("accounts:redirect")
    form = CitizenRegistrationForm(request.POST or None)
    if request.method == "POST" and form.is_valid():
        user = form.save()
        login(request, user)
        messages.success(request, "Welcome! Your citizen account has been created.")
        return redirect("grievances:dashboard")
    return render(request, "accounts/register.html", {"form": form})


class CitizenLoginView(LoginView):
    template_name = "accounts/login.html"
    redirect_authenticated_user = True

    def get_success_url(self):
        return self.get_redirect_url() or str(reverse_lazy("accounts:redirect"))


class OfficerLoginView(LoginView):
    """Same authentication, but the account must be an officer (or admin)."""
    template_name = "accounts/officer_login.html"

    def form_valid(self, form):
        if _role(form.get_user()) not in ("officer", "admin"):
            form.add_error(None, "This account is not an authorised officer account.")
            return self.form_invalid(form)
        return super().form_valid(form)

    def get_success_url(self):
        return self.get_redirect_url() or str(reverse_lazy("accounts:redirect"))


def role_redirect(request):
    """Send each role to its own landing page after login."""
    if not request.user.is_authenticated:
        return redirect("accounts:login")
    role = _role(request.user)
    return redirect({"admin": "analytics:admin_overview", "officer": "officers:dashboard"}
                    .get(role, "grievances:dashboard"))

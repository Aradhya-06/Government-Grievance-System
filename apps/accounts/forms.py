import re

from django import forms
from django.contrib.auth import get_user_model
from django.contrib.auth.forms import UserCreationForm

from .models import Profile

User = get_user_model()


class CitizenRegistrationForm(UserCreationForm):
    first_name = forms.CharField(max_length=60, label="Full name")
    email = forms.EmailField(required=True)
    phone = forms.CharField(max_length=15, required=False, label="Mobile (optional)")

    class Meta:
        model = User
        fields = ("username", "first_name", "email", "phone", "password1", "password2")

    def clean_email(self):
        email = self.cleaned_data["email"].lower()
        if User.objects.filter(email__iexact=email).exists():
            raise forms.ValidationError("An account with this email already exists.")
        return email

    def clean_phone(self):
        phone = self.cleaned_data.get("phone", "").strip()
        if phone and not re.fullmatch(r"[6-9]\d{9}", phone):
            raise forms.ValidationError("Enter a valid 10-digit Indian mobile number.")
        return phone

    def save(self, commit=True):
        user = super().save(commit=False)
        user.email = self.cleaned_data["email"]
        user.first_name = self.cleaned_data["first_name"]
        if commit:
            user.save()
            Profile.objects.create(user=user, role=Profile.CITIZEN, phone=self.cleaned_data.get("phone", ""))
        return user

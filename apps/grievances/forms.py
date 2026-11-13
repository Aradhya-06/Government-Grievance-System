from django import forms

from .models import Grievance


class GrievanceForm(forms.ModelForm):
    class Meta:
        model = Grievance
        fields = ["subject", "description", "location", "affected_people"]
        widgets = {
            "subject": forms.TextInput(attrs={"placeholder": "e.g. No water supply in Sector 15 for 4 days"}),
            "description": forms.Textarea(attrs={"rows": 6, "placeholder":
                "Describe the problem: what happened, since when, who is affected, what you already tried."}),
            "location": forms.TextInput(attrs={"placeholder": "Area / ward / city"}),
            "affected_people": forms.NumberInput(attrs={"min": 1}),
        }
        labels = {"affected_people": "Approx. people affected"}

    def clean_subject(self):
        s = self.cleaned_data["subject"].strip()
        if len(s) < 8:
            raise forms.ValidationError("Please give a more descriptive subject (at least 8 characters).")
        return s

    def clean_description(self):
        d = self.cleaned_data["description"].strip()
        if len(d.split()) < 6:
            raise forms.ValidationError("Please describe the issue in at least a few words (6+).")
        return d

    def clean_affected_people(self):
        n = self.cleaned_data["affected_people"]
        if n < 1:
            raise forms.ValidationError("Must be at least 1.")
        if n > 5_000_000:
            raise forms.ValidationError("That number looks too large - please check.")
        return n


class TrackForm(forms.Form):
    tracking_id = forms.CharField(max_length=20, label="Tracking ID",
                                  widget=forms.TextInput(attrs={"placeholder": "GRV-2026-AB12CD"}))

    def clean_tracking_id(self):
        return self.cleaned_data["tracking_id"].strip().upper()

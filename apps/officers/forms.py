from django import forms

from apps.grievances.models import Grievance


class UpdateStatusForm(forms.Form):
    status = forms.ChoiceField(choices=Grievance.STATUS_CHOICES)
    remarks = forms.CharField(widget=forms.Textarea(attrs={"rows": 3, "placeholder":
                              "Action taken / reason for the status change (visible to the citizen)"}),
                              min_length=5, label="Officer remarks")

    def clean(self):
        data = super().clean()
        if data.get("status") in ("resolved", "rejected") and len((data.get("remarks") or "").split()) < 3:
            raise forms.ValidationError("Closing a grievance needs a short explanation (3+ words).")
        return data

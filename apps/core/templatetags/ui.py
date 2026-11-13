from django import template

register = template.Library()


@register.filter
def add_class(field, css):
    """{{ form.field|add_class:"form-control" }} - keeps the widget's own attrs."""
    existing = field.field.widget.attrs.get("class", "")
    return field.as_widget(attrs={"class": f"{existing} {css}".strip()})


@register.filter
def pct(value, digits=0):
    try:
        return f"{float(value) * 100:.{int(digits)}f}%"
    except (TypeError, ValueError):
        return ""


@register.filter
def days_text(value):
    """3.4 -> '3.4 d', negative -> 'overdue by 2.1 d'."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "-"
    return f"overdue by {abs(v):.1f} d" if v < 0 else f"{v:.1f} d"


@register.filter
def get_item(d, key):
    return d.get(key, "") if hasattr(d, "get") else ""

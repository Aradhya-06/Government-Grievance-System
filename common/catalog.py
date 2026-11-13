"""
common/catalog.py
=================
Single source of truth for everything that maps a grievance *category* to
department / severity / SLA / processing-time assumptions.

Every number here is an ASSUMPTION made for this system, not an
official CPGRAMS figure.  They are collected in one place so that they can be
documented, reviewed, and changed without touching algorithm code.

  base_severity   0-10  How harmful a delay is for the citizen in this category
                        (e.g. a live-wire / no-water complaint is more urgent
                        than a certificate-correction request).
  sla_days              Target days to resolve.  Assumed; verify the official
                        CPGRAMS timeline before quoting it in the report.
  process_hours         Assumed officer effort (hours) to actually resolve one
                        grievance of this category. Used only by the scheduler.
  department            Demo department that owns the category.  The names are
                        real Government of India department names, but the
                        org codes are DEMO codes, not real CPGRAMS org codes.
"""

from __future__ import annotations

# ---------------------------------------------------------------- departments
DEPARTMENTS = [
    # (demo_code, name, officers_available)
    ("DEMO-MUN", "Municipal Water & Sanitation Services", 3),
    ("DEMO-PWD", "Public Works & Roads Department", 2),
    ("DEMO-PWR", "Power & Electricity Department", 2),
    ("DEMO-PEN", "Department of Pension & Pensioners' Welfare", 2),
    ("DEMO-REV", "Department of Revenue (Taxation)", 2),
    ("DEMO-DOC", "Citizen Documentation Services", 2),
    ("DEMO-FPD", "Department of Food & Public Distribution", 2),
    ("DEMO-HLT", "Department of Health & Family Welfare", 2),
    ("DEMO-EDU", "Department of Higher Education", 2),
    ("DEMO-POL", "Public Safety & Police Facilitation", 2),
    ("DEMO-RLY", "Railways Passenger Services", 2),
    ("DEMO-TEL", "Telecom & Postal Services", 2),
    ("DEMO-FIN", "Department of Financial Services", 2),
    ("DEMO-GEN", "General Administration", 2),
]

# ------------------------------------------------------------------ categories
# 14 categories (matches the "14 categories" in the synopses).  If the real
# Kaggle file uses different names, only this table needs editing.
CATEGORIES = {
    "Water Supply":          dict(dept="DEMO-MUN", base_severity=8.0, sla_days=7,  process_hours=6),
    "Sanitation & Garbage":  dict(dept="DEMO-MUN", base_severity=6.5, sla_days=10, process_hours=5),
    "Roads & Infrastructure":dict(dept="DEMO-PWD", base_severity=6.0, sla_days=21, process_hours=10),
    "Electricity":           dict(dept="DEMO-PWR", base_severity=8.5, sla_days=5,  process_hours=6),
    "Pension":               dict(dept="DEMO-PEN", base_severity=7.0, sla_days=21, process_hours=4),
    "Taxation":              dict(dept="DEMO-REV", base_severity=4.5, sla_days=21, process_hours=4),
    "Documentation":         dict(dept="DEMO-DOC", base_severity=4.0, sla_days=15, process_hours=3),
    "Public Distribution":   dict(dept="DEMO-FPD", base_severity=7.0, sla_days=10, process_hours=3),
    "Healthcare":            dict(dept="DEMO-HLT", base_severity=8.5, sla_days=5,  process_hours=5),
    "Education":             dict(dept="DEMO-EDU", base_severity=4.5, sla_days=21, process_hours=4),
    "Police & Public Safety":dict(dept="DEMO-POL", base_severity=8.0, sla_days=5,  process_hours=6),
    "Railways":              dict(dept="DEMO-RLY", base_severity=5.0, sla_days=10, process_hours=3),
    "Telecom & Postal":      dict(dept="DEMO-TEL", base_severity=3.5, sla_days=15, process_hours=3),
    "Banking & Finance":     dict(dept="DEMO-FIN", base_severity=5.5, sla_days=15, process_hours=4),
}

CATEGORY_NAMES = list(CATEGORIES.keys())
FALLBACK_DEPT = "DEMO-GEN"          # used when the classifier is not confident
LOW_CONFIDENCE = 0.35               # below this the grievance goes to General Admin

# Words in the text that raise severity (documented, transparent keyword bump).
# Each hit adds +1.0 (capped so severity stays in 0-10).  Fairness note: the
# bump looks only at the complaint wording, never at location or person.
URGENT_KEYWORDS = [
    "fire", "accident", "injured", "hospital", "emergency", "live wire",
    "electrocution", "sparking", "flood", "sewage overflow", "no water",
    "contaminated", "died", "death", "threat", "unsafe", "collapsed",
    "ambulance", "medicine", "stopped", "not received for", "months",
]
KEYWORD_BUMP = 1.0
MAX_KEYWORD_BUMP = 2.0


def department_code_for(category: str) -> str:
    return CATEGORIES.get(category, {}).get("dept", FALLBACK_DEPT)


def keyword_hits(text: str) -> int:
    t = (text or "").lower()
    return sum(1 for k in URGENT_KEYWORDS if k in t)

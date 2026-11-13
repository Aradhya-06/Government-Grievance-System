"""
ga/fuzzy/engine.py
==============================
Mamdani fuzzy inference system for grievance urgency.


Pipeline implemented step by step (each step is a separate, readable function
so every score can be explained):

    crisp inputs
       -> 1. FUZZIFICATION      (membership degree of each input in each set)
       -> 2. RULE EVALUATION    (AND = min over the antecedents of a rule)
       -> 3. IMPLICATION        (clip the consequent set at the rule strength)
       -> 4. AGGREGATION        (max over all clipped consequent sets)
       -> 5. DEFUZZIFICATION    (centroid of the aggregated area)
       -> crisp Priority Score in [0, 100]

scikit-fuzzy supplies the membership-function primitives (trimf / trapmf),
`interp_membership` and `defuzz`.  The inference loop is written explicitly
(rather than hidden inside skfuzzy.control) so that we can return the list of
FIRED RULES with their strengths -> explainable prioritisation.
tests/test_fuzzy.py cross-checks this engine against skfuzzy.control.

INPUT UNIVERSES (all crisp inputs are normalised into these ranges)
    severity      0-10   category base severity + keyword bump
    public_impact 0-10   affected people (log scale) + recurrence bonus
    waiting       0-60   days since submission (capped at 60)
    sla_urgency   0-10   10 * fraction of the SLA window already used
                         (>= 1.0 means overdue -> 10)
OUTPUT UNIVERSE
    priority      0-100

MEMBERSHIP FUNCTION SHAPES (why):
  * Low / High at the edges use trapezoids that stay at 1.0 up to the end of
    the universe ("shoulders"), so extreme values are fully Low / fully High.
  * Medium uses a triangle centred in the universe -> gradual transitions.
  * Overlap of ~50% between neighbouring sets gives smooth output (no sudden
    jump in priority when an input moves slightly).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import skfuzzy as fuzz

# --------------------------------------------------------------------------- #
# Universes of discourse
# --------------------------------------------------------------------------- #
U_SEVERITY = np.linspace(0, 10, 101)
U_IMPACT = np.linspace(0, 10, 101)
U_WAITING = np.linspace(0, 60, 121)
U_SLA = np.linspace(0, 10, 101)
U_PRIORITY = np.linspace(0, 100, 201)

UNIVERSES = {
    "severity": U_SEVERITY,
    "public_impact": U_IMPACT,
    "waiting": U_WAITING,
    "sla_urgency": U_SLA,
    "priority": U_PRIORITY,
}


def _three_sets(u: np.ndarray, lo_end: float, med: tuple[float, float, float], hi_start: float):
    """Low (left shoulder), Medium (triangle), High (right shoulder)."""
    top = float(u[-1])
    return {
        "low": fuzz.trapmf(u, [float(u[0]), float(u[0]), lo_end * 0.5, lo_end]),
        "medium": fuzz.trimf(u, list(med)),
        "high": fuzz.trapmf(u, [hi_start, hi_start + (top - hi_start) * 0.5, top, top]),
    }


# Severity / impact / SLA universes are 0-10: Low 0-4, Medium 2-5-8, High 6-10.
MF = {
    "severity": _three_sets(U_SEVERITY, 4.0, (2.0, 5.0, 8.0), 6.0),
    "public_impact": _three_sets(U_IMPACT, 4.0, (2.0, 5.0, 8.0), 6.0),
    "sla_urgency": _three_sets(U_SLA, 4.0, (2.0, 5.0, 8.0), 6.0),
    # Waiting 0-60 days: Low 0-12 d, Medium 5-20-35 d, High from 25 d.
    "waiting": {
        "low": fuzz.trapmf(U_WAITING, [0, 0, 6, 12]),
        "medium": fuzz.trimf(U_WAITING, [5, 20, 35]),
        "high": fuzz.trapmf(U_WAITING, [25, 40, 60, 60]),
    },
    # Output: four sets so that "critical" is reachable only by strong evidence.
    "priority": {
        "low": fuzz.trapmf(U_PRIORITY, [0, 0, 15, 35]),
        "medium": fuzz.trimf(U_PRIORITY, [25, 45, 65]),
        "high": fuzz.trimf(U_PRIORITY, [55, 72, 88]),
        "critical": fuzz.trapmf(U_PRIORITY, [78, 90, 100, 100]),
    },
}

# --------------------------------------------------------------------------- #
# Rule base.  Each rule: (antecedents dict, consequent label, plain-English why)
# Antecedent variables not listed are "don't care".  AND = min.
# --------------------------------------------------------------------------- #
RULES: list[tuple[dict, str, str]] = [
    ({"severity": "high", "public_impact": "high", "sla_urgency": "high"}, "critical",
     "Severe, widespread and close to/over its deadline"),
    ({"severity": "high", "public_impact": "high", "waiting": "high"}, "critical",
     "Severe, widespread and has waited very long"),
    ({"severity": "high", "sla_urgency": "high"}, "critical",
     "Severe issue about to breach SLA"),
    ({"severity": "high", "public_impact": "high"}, "high",
     "Severe and affects many people"),
    ({"severity": "high", "public_impact": "medium"}, "high",
     "Severe issue with moderate public impact"),
    ({"severity": "high", "public_impact": "low"}, "medium",
     "Severe but affects few people"),
    ({"severity": "medium", "public_impact": "high"}, "high",
     "Moderate severity but many people affected"),
    ({"severity": "medium", "public_impact": "medium"}, "medium",
     "Moderate severity and impact"),
    ({"severity": "medium", "public_impact": "low"}, "low",
     "Moderate severity, few people affected"),
    ({"severity": "low", "public_impact": "high"}, "medium",
     "Minor issue but widespread"),
    ({"severity": "low", "public_impact": "medium"}, "low",
     "Minor issue, moderate spread"),
    ({"severity": "low", "public_impact": "low"}, "low",
     "Minor, isolated issue"),
    ({"sla_urgency": "high", "waiting": "high"}, "high",
     "Ageing grievance at the end of its SLA window (escalate)"),
    ({"sla_urgency": "high", "severity": "medium"}, "high",
     "Moderate issue nearly out of time"),
    ({"sla_urgency": "high", "severity": "low"}, "medium",
     "Minor issue nearly out of time"),
    ({"waiting": "high", "severity": "low"}, "medium",
     "Minor issue that has waited too long (anti-starvation)"),
    ({"waiting": "medium", "severity": "medium", "sla_urgency": "medium"}, "medium",
     "Mid-way through the SLA window"),
    ({"sla_urgency": "low", "waiting": "low", "severity": "low"}, "low",
     "Fresh, minor, plenty of time"),
    ({"sla_urgency": "low", "waiting": "low", "public_impact": "low"}, "low",
     "Fresh and isolated"),
    ({"waiting": "high", "public_impact": "high"}, "high",
     "Widespread and waited long"),
]


@dataclass
class FuzzyResult:
    score: float                                   # defuzzified priority 0-100
    label: str                                     # Low / Medium / High / Critical
    inputs: dict                                   # the crisp (clipped) inputs
    memberships: dict                              # fuzzification result
    fired_rules: list = field(default_factory=list)  # [{'rule','strength','then','why'}]

    def to_dict(self) -> dict:
        return {
            "score": round(self.score, 2),
            "label": self.label,
            "inputs": {k: round(v, 3) for k, v in self.inputs.items()},
            "memberships": {
                var: {s: round(d, 3) for s, d in sets.items()}
                for var, sets in self.memberships.items()
            },
            "fired_rules": self.fired_rules,
        }


# --------------------------------------------------------------------------- #
# Step 1: fuzzification
# --------------------------------------------------------------------------- #
def fuzzify(inputs: dict) -> dict:
    """Return {variable: {set_name: degree_of_membership}} for the 4 inputs."""
    out = {}
    for var in ("severity", "public_impact", "waiting", "sla_urgency"):
        x = inputs[var]
        out[var] = {
            name: float(fuzz.interp_membership(UNIVERSES[var], mf, x))
            for name, mf in MF[var].items()
        }
    return out


# --------------------------------------------------------------------------- #
# Steps 2-4: rule evaluation, implication, aggregation
# --------------------------------------------------------------------------- #
def infer(memberships: dict):
    """Evaluate every rule; return (aggregated_output_curve, fired_rules)."""
    aggregated = np.zeros_like(U_PRIORITY)
    fired = []
    for antecedents, consequent, why in RULES:
        strength = min(memberships[var][s] for var, s in antecedents.items())  # AND = min
        if strength <= 1e-6:
            continue
        clipped = np.fmin(strength, MF["priority"][consequent])    # implication (Mamdani clip)
        aggregated = np.fmax(aggregated, clipped)                  # aggregation (max)
        fired.append({
            "rule": " AND ".join(f"{v} is {s}" for v, s in antecedents.items()),
            "then": f"priority is {consequent}",
            "strength": round(strength, 3),
            "why": why,
        })
    fired.sort(key=lambda r: r["strength"], reverse=True)
    return aggregated, fired


# --------------------------------------------------------------------------- #
# Step 5: defuzzification
# --------------------------------------------------------------------------- #
def defuzzify(aggregated: np.ndarray) -> float:
    if aggregated.max() <= 0:
        return 0.0
    return float(fuzz.defuzz(U_PRIORITY, aggregated, "centroid"))


def label_for(score: float) -> str:
    if score >= 75:
        return "Critical"
    if score >= 58:
        return "High"
    if score >= 35:
        return "Medium"
    return "Low"


def compute_priority(severity: float, public_impact: float,
                     waiting_days: float, sla_urgency: float) -> FuzzyResult:
    """Public entry point used by the Django pipeline and the experiments."""
    inputs = {
        "severity": float(np.clip(severity, 0, 10)),
        "public_impact": float(np.clip(public_impact, 0, 10)),
        "waiting": float(np.clip(waiting_days, 0, 60)),
        "sla_urgency": float(np.clip(sla_urgency, 0, 10)),
    }
    memberships = fuzzify(inputs)
    aggregated, fired = infer(memberships)
    score = defuzzify(aggregated)
    return FuzzyResult(score=score, label=label_for(score), inputs=inputs,
                       memberships=memberships, fired_rules=fired)


# --------------------------------------------------------------------------- #
# Documented conversion of raw grievance facts into the fuzzy inputs
# --------------------------------------------------------------------------- #
def impact_from(affected_people: int, recurrence_count: int) -> float:
    """
    Public impact on 0-10.
      * affected people on a log10 scale: 1 person -> 0, 10 -> 1.75,
        100 -> 3.5, 1,000 -> 5.25, 10,000+ -> 7.0 (capped)
      * recurrence: every similar earlier complaint adds 0.75, max +3.0.
        Rationale: many separate citizens reporting the same issue is evidence
        that more people are affected than the form says.
    """
    people = max(1, int(affected_people or 1))
    base = min(7.0, math.log10(people) * 1.75)
    bonus = min(3.0, 0.75 * max(0, int(recurrence_count or 0)))
    return float(min(10.0, base + bonus))


def sla_urgency_from(elapsed_days: float, sla_days: float) -> float:
    """10 * fraction of the SLA window consumed (overdue -> 10)."""
    if sla_days <= 0:
        return 10.0
    return float(min(10.0, 10.0 * max(0.0, elapsed_days) / sla_days))


def severity_from(base_severity: float, keyword_hits: int,
                  bump: float = 1.0, max_bump: float = 2.0) -> float:
    return float(min(10.0, base_severity + min(max_bump, bump * keyword_hits)))

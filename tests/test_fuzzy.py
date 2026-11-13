import numpy as np
from django.test import SimpleTestCase
from skfuzzy import control as ctrl

from ga.fuzzy import engine as E


class FuzzyEngineTests(SimpleTestCase):
    def test_extremes(self):
        hi = E.compute_priority(10, 10, 60, 10)
        lo = E.compute_priority(0, 0, 0, 0)
        self.assertEqual(hi.label, "Critical")
        self.assertEqual(lo.label, "Low")
        self.assertGreater(hi.score, 75)
        self.assertLess(lo.score, 25)

    def test_monotonic_in_severity(self):
        scores = [E.compute_priority(s, 5, 10, 5).score for s in range(0, 11)]
        self.assertTrue(all(b >= a - 1.0 for a, b in zip(scores, scores[1:])), scores)

    def test_waiting_and_sla_raise_priority(self):
        base = E.compute_priority(4, 4, 1, 1).score
        aged = E.compute_priority(4, 4, 45, 9).score
        self.assertGreater(aged, base + 5)

    def test_inputs_are_clipped(self):
        r = E.compute_priority(99, -5, 1000, 50)
        self.assertTrue(0 <= r.score <= 100)

    def test_fired_rules_explainable(self):
        r = E.compute_priority(9, 9, 30, 9)
        self.assertTrue(r.fired_rules)
        self.assertTrue(all(0 < x["strength"] <= 1 and x["why"] for x in r.fired_rules))

    def test_membership_degrees_in_unit_interval(self):
        m = E.fuzzify(dict(severity=5, public_impact=5, waiting=20, sla_urgency=5))
        for sets in m.values():
            for d in sets.values():
                self.assertTrue(0 <= d <= 1)

    def test_matches_skfuzzy_control_system(self):
        """Cross-check our explicit Mamdani loop against skfuzzy.control for random inputs."""
        ants = {}
        for var in ("severity", "public_impact", "waiting", "sla_urgency"):
            a = ctrl.Antecedent(E.UNIVERSES[var], var)
            for name, mf in E.MF[var].items():
                a[name] = mf
            ants[var] = a
        out = ctrl.Consequent(E.UNIVERSES["priority"], "priority", defuzzify_method="centroid")
        for name, mf in E.MF["priority"].items():
            out[name] = mf
        rules = []
        for antecedents, cons, _ in E.RULES:
            cond = None
            for v, s in antecedents.items():
                term = ants[v][s]
                cond = term if cond is None else (cond & term)
            rules.append(ctrl.Rule(cond, out[cons]))
        sim = ctrl.ControlSystemSimulation(ctrl.ControlSystem(rules))
        rng = np.random.default_rng(0)
        checked = 0
        for _ in range(25):
            vals = dict(severity=rng.uniform(0, 10), public_impact=rng.uniform(0, 10),
                        waiting=rng.uniform(0, 60), sla_urgency=rng.uniform(0, 10))
            for k, v in vals.items():
                sim.input[k] = v
            try:
                sim.compute()
            except Exception:
                continue          # skfuzzy raises when no rule fires at all
            mine = E.compute_priority(vals["severity"], vals["public_impact"], vals["waiting"], vals["sla_urgency"])
            self.assertAlmostEqual(mine.score, float(sim.output["priority"]), delta=0.5)
            checked += 1
        self.assertGreater(checked, 10)

    def test_conversion_helpers(self):
        self.assertEqual(E.impact_from(1, 0), 0.0)
        self.assertGreater(E.impact_from(1000, 0), E.impact_from(10, 0))
        self.assertEqual(E.impact_from(10, 100), E.impact_from(10, 4))      # recurrence bonus capped
        self.assertEqual(E.sla_urgency_from(30, 10), 10.0)                   # overdue
        self.assertEqual(E.sla_urgency_from(5, 10), 5.0)
        self.assertEqual(E.severity_from(9, 10), 10.0)                       # capped

import random

from django.test import SimpleTestCase

from ga.scheduling import scheduler as G


def make_items(n, seed=1):
    r = random.Random(seed)
    return [G.QueueItem(id=100 + i, priority=r.uniform(0, 100), waited_days=r.uniform(0, 20),
                        deadline_days=r.uniform(-2, 15), hours=r.uniform(2, 10)) for i in range(n)]


class GATests(SimpleTestCase):
    def test_valid_permutation_output(self):
        items = make_items(25)
        res = G.run_ga(items, G.Capacity(officers=2), G.GAConfig(generations=30, seed=3))
        self.assertEqual(sorted(res.order), sorted(i.id for i in items))

    def test_operators_preserve_validity(self):
        rng = random.Random(0)
        for n in (2, 3, 10, 40):
            a, b = list(range(n)), list(range(n))
            rng.shuffle(a); rng.shuffle(b)
            for _ in range(50):
                c = G.order_crossover(a, b, rng)
                self.assertTrue(G.is_valid(c, n))
                self.assertTrue(G.is_valid(G.mutate(c, 1.0, rng), n))

    def test_invalid_chromosome_penalised(self):
        items = make_items(5)
        m = G.evaluate([0, 0, 1, 2, 3], items, G.Capacity())
        self.assertGreaterEqual(m.cost, G.P_INV)

    def test_ga_not_worse_than_seed_heuristics(self):
        items = make_items(30, seed=5)
        cap = G.Capacity(officers=2)
        res = G.run_ga(items, cap, G.GAConfig(generations=60, seed=9))
        for order in (G.order_by_priority(items), G.order_by_deadline(items)):
            self.assertLessEqual(res.metrics.cost, G.cost_of(order, items, cap) + 1e-9)

    def test_history_is_non_increasing(self):
        items = make_items(20)
        h = G.run_ga(items, G.Capacity(), G.GAConfig(generations=40, seed=1)).history
        self.assertTrue(all(b <= a + 1e-9 for a, b in zip(h, h[1:])))

    def test_reproducible_with_seed(self):
        items = make_items(20)
        a = G.run_ga(items, G.Capacity(), G.GAConfig(generations=20, seed=7)).order
        b = G.run_ga(items, G.Capacity(), G.GAConfig(generations=20, seed=7)).order
        self.assertEqual(a, b)

    def test_edge_cases(self):
        self.assertEqual(G.run_ga([], G.Capacity()).order, [])
        one = make_items(1)
        self.assertEqual(G.run_ga(one, G.Capacity()).order, [one[0].id])
        two = make_items(2)
        self.assertEqual(sorted(G.run_ga(two, G.Capacity(), G.GAConfig(generations=5)).order), sorted(i.id for i in two))

    def test_capacity_changes_schedule_length(self):
        items = make_items(20)
        order = list(range(20))
        slow = G.evaluate(order, items, G.Capacity(officers=1)).makespan_days
        fast = G.evaluate(order, items, G.Capacity(officers=4)).makespan_days
        self.assertGreater(slow, fast)

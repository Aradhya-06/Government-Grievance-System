"""
ga/experiments/run_comparison.py
============================================
Comparative evaluation (scheduling methods).

Methods compared on the SAME synthetic queues and the SAME capacity model:

  1. FIFO                      first come, first served (longest waiting first)
  2. Static priority (baseline) conventional rule: sort by the category's fixed
                               base severity, ties by earliest deadline.  No fuzzy
                               logic, no optimisation - this is the "conventional
                               priority-based processing" of the spec.
  3. Fuzzy sort (ablation)     sort by fuzzy priority score only (fuzzy, no GA)
  4. Fuzzy + GA (proposed)     GA search seeded with random permutations (+ the two
                               heuristic orders), minimising the scheduling cost.

Queues are generated randomly (seeded) from the catalog - they are SIMULATED
queues, not real grievances.  Results are saved as JSON/CSV/PNG and read by the
Django analytics page.  Nothing is hard-coded: run the file to regenerate.

Honesty note: the GA cost function uses the fuzzy score, so the "scheduling
cost" column favours methods that know the fuzzy score.  The metrics that do not
depend on the cost function (avg waiting time, SLA violations, waiting time of
high-priority cases, runtime) are therefore the fairer comparison.
"""

from __future__ import annotations

import json
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.catalog import CATEGORIES  # noqa: E402
from ga.fuzzy.engine import (compute_priority, impact_from,  # noqa: E402
                                         sla_urgency_from, severity_from)
from ga.scheduling.scheduler import (Capacity, GAConfig, QueueItem, evaluate,  # noqa: E402
                                         order_by_deadline, order_by_priority,
                                         order_fifo, run_ga)

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def generate_queue(n: int, rng: random.Random) -> list[QueueItem]:
    """Simulated pending queue of one department (mixed categories)."""
    cats = list(CATEGORIES.items())
    items = []
    for i in range(n):
        name, meta = rng.choice(cats)
        waited = rng.choice([0, 1, 2, 3, 5, 8, 12, 18, 25, 35])
        waited += rng.random() * 2
        sla_left = meta["sla_days"] - waited
        affected = int(10 ** rng.uniform(0, 3.8))
        recurrence = rng.choice([0, 0, 0, 1, 2, 4])
        hits = rng.choice([0, 0, 0, 1, 2])
        sev = severity_from(meta["base_severity"], hits)
        imp = impact_from(affected, recurrence)
        sla_u = sla_urgency_from(waited, meta["sla_days"])
        fz = compute_priority(sev, imp, waited, sla_u)
        items.append(QueueItem(id=i, priority=fz.score, waited_days=waited,
                               deadline_days=sla_left,
                               hours=meta["process_hours"] * rng.uniform(0.7, 1.4),
                               category=name))
    return items


def static_priority_order(items, base_sev):
    return sorted(range(len(items)),
                  key=lambda i: (-base_sev[items[i].category], items[i].deadline_days))


def evaluate_methods(items, cap, cfg):
    base_sev = {k: v["base_severity"] for k, v in CATEGORIES.items()}
    out = {}

    def timed(fn):
        t = time.perf_counter()
        order = fn()
        return order, time.perf_counter() - t

    for name, fn in [
        ("FIFO", lambda: order_fifo(items)),
        ("Static priority (baseline)", lambda: static_priority_order(items, base_sev)),
        ("Fuzzy sort (ablation)", lambda: order_by_priority(items)),
    ]:
        order, rt = timed(fn)
        m = evaluate(order, items, cap)
        out[name] = dict(metrics=m, runtime=rt, order=[items[i].id for i in order])

    res = run_ga(items, cap, cfg)
    pos = {it.id: i for i, it in enumerate(items)}
    out["Fuzzy + GA (proposed)"] = dict(metrics=res.metrics, runtime=res.runtime_s,
                                       order=res.order, history=res.history)
    return out


def run(n_queues=30, queue_size=40, officers=3, hours_per_day=6.0, seed=2026,
        generations=150, population=80, verbose=True):
    rng = random.Random(seed)
    cap = Capacity(officers=officers, hours_per_day=hours_per_day, horizon_days=14.0)
    cfg = GAConfig(population=population, generations=generations, seed=seed)
    methods = ["FIFO", "Static priority (baseline)", "Fuzzy sort (ablation)", "Fuzzy + GA (proposed)"]
    acc = {m: {k: [] for k in ("avg_wait", "sla_viol", "sla_rate", "hp_wait", "cost", "runtime")}
           for m in methods}
    sample_history = None
    for q in range(n_queues):
        items = generate_queue(queue_size, rng)
        res = evaluate_methods(items, cap, cfg)
        for m in methods:
            r, mt = res[m], res[m]["metrics"]
            acc[m]["avg_wait"].append(mt.avg_waiting_days)
            acc[m]["sla_viol"].append(mt.sla_violations)
            acc[m]["sla_rate"].append(mt.sla_violation_rate)
            acc[m]["hp_wait"].append(mt.high_priority_avg_wait_days)
            acc[m]["cost"].append(mt.cost)
            acc[m]["runtime"].append(r["runtime"])
        if q == 0:
            sample_history = res["Fuzzy + GA (proposed)"]["history"]
        if verbose:
            print(f"queue {q + 1}/{n_queues} done", flush=True)

    summary = {}
    for m in methods:
        summary[m] = {k: {"mean": statistics.mean(v), "stdev": statistics.pstdev(v)}
                      for k, v in acc[m].items()}
    # win-rate of GA vs baseline per queue (honest: includes ties / losses)
    base, ga = acc["Static priority (baseline)"], acc["Fuzzy + GA (proposed)"]
    fz = acc["Fuzzy sort (ablation)"]
    wins = {
        "ga_vs_baseline_cost": sum(g < b - 1e-9 for g, b in zip(ga["cost"], base["cost"])),
        "ga_vs_baseline_sla": [sum(g < b for g, b in zip(ga["sla_viol"], base["sla_viol"])),
                               sum(g == b for g, b in zip(ga["sla_viol"], base["sla_viol"])),
                               sum(g > b for g, b in zip(ga["sla_viol"], base["sla_viol"]))],
        "ga_vs_fuzzy_sort_cost": sum(g < f - 1e-9 for g, f in zip(ga["cost"], fz["cost"])),
        "ga_vs_fuzzy_sort_ties": sum(abs(g - f) <= 1e-9 for g, f in zip(ga["cost"], fz["cost"])),
        "queues": n_queues,
    }
    payload = {
        "config": dict(n_queues=n_queues, queue_size=queue_size, officers=officers,
                       hours_per_day=hours_per_day, seed=seed, generations=generations,
                       population=population,
                       note="Simulated queues generated from the category catalogue; not real grievances."),
        "summary": summary, "wins": wins, "ga_history_sample": sample_history,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "comparison.json").write_text(json.dumps(payload, indent=2))
    return payload


def print_table(payload):
    s = payload["summary"]
    print(f"\n{'Method':30s} {'AvgWait(d)':>11s} {'SLA viol':>9s} {'SLA %':>7s} {'HP wait':>8s} {'Cost':>10s} {'Time(s)':>8s}")
    for m, v in s.items():
        print(f"{m:30s} {v['avg_wait']['mean']:11.2f} {v['sla_viol']['mean']:9.2f} "
              f"{100 * v['sla_rate']['mean']:6.1f}% {v['hp_wait']['mean']:8.2f} "
              f"{v['cost']['mean']:10.0f} {v['runtime']['mean']:8.3f}")
    print("\nWin counts:", payload["wins"])


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    p = run(n_queues=n)
    print_table(p)

"""
ga/scheduling/scheduler.py
==============================
Genetic Algorithm that orders a department's pending grievance queue.


PROBLEM MODEL (all assumptions are documented in the report)
-------------------------------------------------------------
* A department has K officers working `hours_per_day` hours per working day.
* Each pending grievance i has
      priority_i     fuzzy score 0-100
      waited_i       days it has already waited
      deadline_i     days from NOW until its SLA deadline (negative = overdue)
      hours_i        effort needed to resolve it
* A *schedule* is an ordering of the grievances.  We simulate it with list
  scheduling: take grievances in that order and give each one to the officer
  who becomes free first.  So the order decides both WHO is served first and
  how the load is spread over officers (workload / capacity).
* completion_i = time (in days from now) when grievance i is finished.

CHROMOSOME   a permutation of the integers 0..n-1 (index of each grievance).
             Permutations are always valid schedules: no grievance is lost or
             repeated.  Operators below are permutation-preserving; `is_valid`
             verifies this and an invalid chromosome gets a huge penalty.

FITNESS      We MINIMISE a scheduling cost (lower = better); the reported
             `fitness` is 1 / (1 + cost) so that larger = better.

    cost = sum_i  w_i * (waited_i + start_i)                 priority-weighted waiting
         + P_SLA  * sum_i w_i * (1 + late_i)   [late_i > 0]  SLA violations, scaled by importance
         + P_HIGH * sum_{i high-priority} completion_i        extra delay penalty for urgent cases
         + P_CAP  * overflow_days                             work beyond the planning horizon
         + P_INV  * invalid                                   structural validity

    w_i = 1 + priority_i / 25        (priority 0 -> 1, priority 100 -> 5)
    late_i = completion_i - deadline_i

    Terms RETURN the "rewards" of the spec as reduced cost: processing
    high-priority items early, reducing waiting time, avoiding SLA violations
    and using capacity within the horizon all lower the cost.

GA OPERATORS
    initialisation  random permutations (+ optional heuristic seeds)
    selection       tournament (size 3)
    crossover       Order Crossover (OX) - preserves permutation validity
    mutation        swap, or inversion (reverse a segment) with probability pm
    replacement     generational with elitism (best `elite` individuals survive)
    termination     max generations OR no improvement for `patience` generations
"""

from __future__ import annotations

import heapq
import random
import time
from dataclasses import dataclass, field

# ---- penalty weights (documented, tunable) ----
P_SLA = 4.0
P_HIGH = 1.5
P_CAP = 20.0
P_INV = 1e9
HIGH_PRIORITY_THRESHOLD = 70.0   # fuzzy score from which a case counts as "high priority"


@dataclass
class QueueItem:
    id: int
    priority: float        # fuzzy score 0-100
    waited_days: float     # days already waited
    deadline_days: float   # days from now to SLA deadline (negative = already overdue)
    hours: float           # processing effort in hours
    category: str = ""


@dataclass
class Capacity:
    officers: int = 2
    hours_per_day: float = 6.0     # productive grievance-handling hours / officer / day
    horizon_days: float = 14.0     # planning horizon


@dataclass
class GAConfig:
    population: int = 60
    generations: int = 120
    crossover_rate: float = 0.9
    mutation_rate: float = 0.25
    tournament: int = 3
    elite: int = 2
    patience: int = 30
    seed_heuristics: bool = True   # put priority-sort / EDF into the initial population
    seed: int | None = 42


@dataclass
class ScheduleMetrics:
    avg_waiting_days: float
    sla_violations: int
    sla_violation_rate: float
    high_priority_avg_wait_days: float
    cost: float
    fitness: float
    makespan_days: float
    overflow_days: float


@dataclass
class GAResult:
    order: list            # list of item ids in processing order
    metrics: ScheduleMetrics
    runtime_s: float
    generations_run: int
    history: list = field(default_factory=list)   # best cost per generation


# --------------------------------------------------------------------------- #
# Simulation + fitness
# --------------------------------------------------------------------------- #
def is_valid(chrom: list[int], n: int) -> bool:
    return len(chrom) == n and sorted(chrom) == list(range(n))


def simulate(chrom: list[int], items: list[QueueItem], cap: Capacity):
    """List-schedule `chrom`; return (start_days, completion_days) arrays by item index."""
    n = len(items)
    free = [(0.0, k) for k in range(max(1, cap.officers))]
    heapq.heapify(free)
    start = [0.0] * n
    comp = [0.0] * n
    for idx in chrom:
        t, k = heapq.heappop(free)
        start[idx] = t
        done = t + items[idx].hours / cap.hours_per_day
        comp[idx] = done
        heapq.heappush(free, (done, k))
    return start, comp


def evaluate(chrom: list[int], items: list[QueueItem], cap: Capacity) -> ScheduleMetrics:
    n = len(items)
    if not is_valid(chrom, n):
        return ScheduleMetrics(1e9, n, 1.0, 1e9, P_INV, 0.0, 1e9, 1e9)
    start, comp = simulate(chrom, items, cap)
    wait_term = sla_term = high_term = 0.0
    waits, high_waits, violations = [], [], 0
    for i, it in enumerate(items):
        w = 1.0 + it.priority / 25.0
        waited_total = it.waited_days + start[i]
        waits.append(waited_total)
        wait_term += w * waited_total
        late = comp[i] - it.deadline_days
        if late > 0:
            violations += 1
            sla_term += w * (1.0 + late)
        if it.priority >= HIGH_PRIORITY_THRESHOLD:
            high_term += comp[i]
            high_waits.append(waited_total)
    makespan = max(comp) if comp else 0.0
    overflow = max(0.0, makespan - cap.horizon_days)
    cost = wait_term + P_SLA * sla_term + P_HIGH * high_term + P_CAP * overflow
    return ScheduleMetrics(
        avg_waiting_days=sum(waits) / n if n else 0.0,
        sla_violations=violations,
        sla_violation_rate=violations / n if n else 0.0,
        high_priority_avg_wait_days=(sum(high_waits) / len(high_waits)) if high_waits else 0.0,
        cost=cost, fitness=1.0 / (1.0 + cost),
        makespan_days=makespan, overflow_days=overflow,
    )


def cost_of(chrom, items, cap) -> float:
    return evaluate(chrom, items, cap).cost


# --------------------------------------------------------------------------- #
# GA operators
# --------------------------------------------------------------------------- #
def order_crossover(p1: list[int], p2: list[int], rng: random.Random) -> list[int]:
    """OX: copy a slice from p1, fill the rest in the order they appear in p2."""
    n = len(p1)
    a, b = sorted(rng.sample(range(n), 2)) if n > 1 else (0, 0)
    child = [None] * n
    child[a:b + 1] = p1[a:b + 1]
    taken = set(child[a:b + 1])
    fill = [g for g in p2 if g not in taken]
    j = 0
    for i in range(n):
        if child[i] is None:
            child[i] = fill[j]
            j += 1
    return child


def mutate(chrom: list[int], rate: float, rng: random.Random) -> list[int]:
    c = chrom[:]
    n = len(c)
    if n < 2 or rng.random() > rate:
        return c
    if rng.random() < 0.5:                       # swap mutation
        i, j = rng.sample(range(n), 2)
        c[i], c[j] = c[j], c[i]
    else:                                        # inversion mutation
        i, j = sorted(rng.sample(range(n), 2))
        c[i:j + 1] = reversed(c[i:j + 1])
    return c


def tournament(pop, costs, k, rng) -> list[int]:
    contenders = rng.sample(range(len(pop)), min(k, len(pop)))
    return pop[min(contenders, key=lambda i: costs[i])]


# --------------------------------------------------------------------------- #
# Heuristic orderings (also used as baselines elsewhere)
# --------------------------------------------------------------------------- #
def order_by_priority(items):      # highest fuzzy priority first
    return sorted(range(len(items)), key=lambda i: (-items[i].priority, items[i].deadline_days))


def order_by_deadline(items):      # earliest deadline first (EDF)
    return sorted(range(len(items)), key=lambda i: (items[i].deadline_days, -items[i].priority))


def order_fifo(items):             # first come first served = longest waiting first
    return sorted(range(len(items)), key=lambda i: -items[i].waited_days)


# --------------------------------------------------------------------------- #
# The Genetic Algorithm
# --------------------------------------------------------------------------- #
def run_ga(items: list[QueueItem], cap: Capacity | None = None,
           cfg: GAConfig | None = None) -> GAResult:
    cap = cap or Capacity()
    cfg = cfg or GAConfig()
    rng = random.Random(cfg.seed)
    t0 = time.perf_counter()
    n = len(items)
    if n == 0:
        return GAResult([], evaluate([], items, cap), 0.0, 0)
    if n == 1:
        m = evaluate([0], items, cap)
        return GAResult([items[0].id], m, time.perf_counter() - t0, 0, [m.cost])

    # 1. population initialisation
    pop = []
    if cfg.seed_heuristics:
        pop += [order_by_priority(items), order_by_deadline(items)]
    while len(pop) < cfg.population:
        c = list(range(n))
        rng.shuffle(c)
        pop.append(c)
    costs = [cost_of(c, items, cap) for c in pop]

    best_i = min(range(len(pop)), key=lambda i: costs[i])
    best, best_cost = pop[best_i][:], costs[best_i]
    history = [best_cost]
    stale = 0
    gen = 0
    for gen in range(1, cfg.generations + 1):
        # 6. replacement with elitism
        ranked = sorted(range(len(pop)), key=lambda i: costs[i])
        new_pop = [pop[i][:] for i in ranked[:cfg.elite]]
        while len(new_pop) < cfg.population:
            p1 = tournament(pop, costs, cfg.tournament, rng)        # 3. selection
            p2 = tournament(pop, costs, cfg.tournament, rng)
            child = order_crossover(p1, p2, rng) if rng.random() < cfg.crossover_rate else p1[:]  # 4.
            child = mutate(child, cfg.mutation_rate, rng)             # 5. mutation
            new_pop.append(child)
        pop = new_pop
        costs = [cost_of(c, items, cap) for c in pop]                 # 2. fitness
        gi = min(range(len(pop)), key=lambda i: costs[i])
        if costs[gi] < best_cost - 1e-9:
            best, best_cost, stale = pop[gi][:], costs[gi], 0
        else:
            stale += 1
        history.append(best_cost)
        if stale >= cfg.patience:                                     # 7. termination
            break

    metrics = evaluate(best, items, cap)
    return GAResult([items[i].id for i in best], metrics, time.perf_counter() - t0, gen, history)


def optimize_queue(items: list[QueueItem], cap: Capacity | None = None,
                   cfg: GAConfig | None = None) -> list[int]:
    """Convenience wrapper used by the Django pipeline -> list of item ids in order."""
    return run_ga(items, cap, cfg).order


def metrics_for_order(order_ids: list[int], items: list[QueueItem], cap: Capacity) -> ScheduleMetrics:
    pos = {it.id: i for i, it in enumerate(items)}
    return evaluate([pos[i] for i in order_ids], items, cap)

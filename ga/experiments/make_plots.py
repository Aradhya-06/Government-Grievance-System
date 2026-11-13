"""
ga/experiments/make_plots.py
========================================
Draws the soft-computing figures from the saved experiment results:
  fuzzy_membership.png   membership functions of the 4 inputs + output
  fuzzy_surface.png      priority surface for severity x SLA urgency
  ga_convergence.png     best cost per generation (sample queue)
  comparison_bars.png    FIFO / static priority / fuzzy sort / fuzzy+GA

Run after run_comparison.py:   python ga/experiments/make_plots.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ga.fuzzy.engine import MF, UNIVERSES, compute_priority  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"
COL = {"low": "#3a8f5c", "medium": "#d98b2b", "high": "#c0392b", "critical": "#6c2a8a"}
TITLES = {"severity": "Severity (0-10)", "public_impact": "Public impact (0-10)",
          "waiting": "Waiting time (days)", "sla_urgency": "SLA urgency (0-10)",
          "priority": "OUTPUT: Priority score (0-100)"}


def membership():
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.2))
    for ax, var in zip(axes.ravel(), ["severity", "public_impact", "waiting", "sla_urgency", "priority"]):
        for name, mf in MF[var].items():
            ax.plot(UNIVERSES[var], mf, color=COL[name], lw=2, label=name)
            ax.fill_between(UNIVERSES[var], mf, alpha=0.08, color=COL[name])
        ax.set_title(TITLES[var], fontsize=10); ax.set_ylim(-0.02, 1.05); ax.grid(alpha=.3)
        ax.legend(fontsize=7)
    axes.ravel()[-1].axis("off")
    fig.suptitle("Membership functions of the Mamdani priority system", fontsize=12)
    fig.tight_layout(); fig.savefig(OUT / "fuzzy_membership.png", dpi=140); plt.close(fig)


def surface():
    s = np.linspace(0, 10, 21)
    z = np.array([[compute_priority(a, 5, 10, b).score for b in s] for a in s])
    fig, ax = plt.subplots(figsize=(6.2, 5))
    c = ax.contourf(s, s, z, levels=14, cmap="YlOrRd")
    ax.set_xlabel("SLA urgency"); ax.set_ylabel("Severity")
    ax.set_title("Priority surface (public impact = 5, waiting = 10 days)", fontsize=10)
    fig.colorbar(c, label="priority"); fig.tight_layout()
    fig.savefig(OUT / "fuzzy_surface.png", dpi=140); plt.close(fig)


def comparison():
    p = json.loads((OUT / "comparison.json").read_text())
    S = p["summary"]
    methods = list(S)
    colors = ["#9aa5b1", "#d98b2b", "#5b8fb9", "#2f5d8a"]
    panels = [("avg_wait", "Average waiting time (days)"), ("sla_viol", "SLA violations per queue"),
              ("hp_wait", "Waiting time of high-priority cases (days)"), ("cost", "Scheduling cost (lower = better)")]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    for ax, (k, t) in zip(axes, panels):
        vals = [S[m][k]["mean"] for m in methods]
        err = [S[m][k]["stdev"] for m in methods]
        ax.bar(range(len(methods)), vals, yerr=err, color=colors, capsize=3)
        ax.set_xticks(range(len(methods)), ["FIFO", "Static\npriority", "Fuzzy\nsort", "Fuzzy\n+ GA"], fontsize=8)
        ax.set_title(t, fontsize=9); ax.grid(axis="y", alpha=.3)
    fig.suptitle(f"Scheduling comparison over {p['config']['n_queues']} simulated queues "
                 f"of {p['config']['queue_size']} grievances", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / "comparison_bars.png", dpi=140); plt.close(fig)
    h = p["ga_history_sample"]
    fig, ax = plt.subplots(figsize=(6.2, 4))
    ax.plot(h, color="#2f5d8a", lw=2); ax.set_xlabel("generation"); ax.set_ylabel("best cost")
    ax.set_title("GA convergence (sample queue)", fontsize=10); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / "ga_convergence.png", dpi=140); plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    membership(); surface()
    if (OUT / "comparison.json").exists():
        comparison()
    print("plots saved in", OUT)

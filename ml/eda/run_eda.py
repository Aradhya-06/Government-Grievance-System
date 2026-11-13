"""
ml/eda/run_eda.py
===========================
Exploratory data analysis. Each chart answers ONE question (printed in the title / JSON).
Saves PNGs in ml/results/eda_*.png and a captions JSON used by the Django page.

Charts that need data we do not have (state-wise statistics, pendency-age buckets,
redressal-time distributions) are listed in `unavailable` instead of being faked.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from ml.forecasting.monthly_data import is_simulated, load_monthly  # noqa: E402
from ml.preprocessing.load_complaints import is_synthetic, load_complaints  # noqa: E402

RESULTS = ROOT / "ml" / "results"
BLUE, ORANGE, GREEN, GREY = "#2f5d8a", "#d98b2b", "#3a8f5c", "#6b7280"
sns.set_theme(style="whitegrid", context="notebook")


def save(fig, name):
    fig.tight_layout()
    fig.savefig(RESULTS / name, dpi=140)
    plt.close(fig)


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    m = load_monthly()
    sim = is_simulated(m)
    tag = " (SIMULATED)" if sim else ""
    captions = []

    # 1 receipts vs disposals over time
    tot = m.groupby("month")[["receipts", "disposals"]].sum()
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(tot.index, tot["receipts"], color=BLUE, label="Receipts", lw=2)
    ax.plot(tot.index, tot["disposals"], color=ORANGE, label="Disposals", lw=2)
    ax.set_title("Are departments keeping pace with incoming grievances?" + tag, fontsize=11)
    ax.legend(); save(fig, "eda_monthly_trend.png")
    gap = (tot["disposals"].sum() / tot["receipts"].sum())
    captions.append(dict(file="eda_monthly_trend.png", question="Do disposals keep pace with receipts?",
                         finding=f"Overall disposal-to-receipt ratio across the period: {gap:.2f}."))

    # 2 department volume
    dv = m.groupby("department")["receipts"].sum().sort_values()
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.barh([d[:46] for d in dv.index], dv.values, color=BLUE)
    ax.set_title("Which departments receive the most grievances?" + tag, fontsize=11)
    save(fig, "eda_department_volume.png")
    captions.append(dict(file="eda_department_volume.png", question="Which departments carry the highest load?",
                         finding=f"Highest total receipts: {dv.index[-1]}."))

    # 3 disposal-rate boxplot per department
    m = m.assign(rate=m["disposals"] / m["receipts"])
    order = m.groupby("department")["rate"].median().sort_values().index
    fig, ax = plt.subplots(figsize=(9, 4.8))
    sns.boxplot(data=m, y="department", x="rate", order=order, color="#9db7d3", ax=ax)
    ax.axvline(1, color=ORANGE, ls="--"); ax.set_ylabel("")
    ax.set_yticklabels([t.get_text()[:40] for t in ax.get_yticklabels()], fontsize=8)
    ax.set_title("How consistent is each department's disposal rate? (disposals / receipts)" + tag, fontsize=11)
    save(fig, "eda_disposal_rate_box.png")
    captions.append(dict(file="eda_disposal_rate_box.png", question="Which departments dispose consistently?",
                         finding="Boxes right of the dashed line clear more than they receive."))

    # 4 heatmap: receipts by department x calendar month (seasonality)
    mm = m.assign(cal=m["month"].dt.month)
    hm = mm.pivot_table(index="department", columns="cal", values="receipts", aggfunc="mean")
    hm = hm.div(hm.mean(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    sns.heatmap(hm, cmap="YlGnBu", ax=ax, cbar_kws={"label": "relative to dept mean"})
    ax.set_yticklabels([t.get_text()[:34] for t in ax.get_yticklabels()], fontsize=8)
    ax.set_xlabel("calendar month"); ax.set_ylabel("")
    ax.set_title("Is there seasonality in grievance volume?" + tag, fontsize=11)
    save(fig, "eda_seasonality_heatmap.png")
    captions.append(dict(file="eda_seasonality_heatmap.png", question="Is volume seasonal?",
                         finding="Darker cells = months with above-average receipts for that department."))

    # 5 backlog vs disposal performance (correlation)
    last = m.sort_values("month").groupby("department").tail(12)
    agg = last.groupby("department").agg(pending=("pending", "mean"), rate=("rate", "mean"),
                                         rec=("receipts", "mean"))
    agg["backlog_months"] = agg["pending"] / agg["rec"]
    r = float(np.corrcoef(agg["backlog_months"], agg["rate"])[0, 1]) if len(agg) > 2 else float("nan")
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    ax.scatter(agg["backlog_months"], agg["rate"], color=BLUE, s=60)
    for d, row in agg.iterrows():
        ax.annotate(d[:16], (row["backlog_months"], row["rate"]), fontsize=7, xytext=(4, 3),
                    textcoords="offset points")
    ax.set_xlabel("average backlog (months of receipts)"); ax.set_ylabel("average disposal rate")
    ax.set_title(f"Does a bigger backlog slow disposal? r = {r:.2f}" + tag, fontsize=11)
    save(fig, "eda_backlog_vs_rate.png")
    captions.append(dict(file="eda_backlog_vs_rate.png", question="Does backlog age/size relate to disposal performance?",
                         finding=f"Pearson r between backlog size and disposal rate = {r:.2f} over {len(agg)} departments "
                                 "(small sample - indicative only; this is backlog SIZE, not age)."))

    # 6 complaint-category distribution (text corpus)
    c = load_complaints()
    cs = c["category"].value_counts()
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.barh(cs.index[::-1], cs.values[::-1], color=GREEN)
    ax.set_title("Category distribution of the complaint-text dataset"
                 + (" (SYNTHETIC)" if is_synthetic(c) else ""), fontsize=11)
    save(fig, "eda_category_distribution.png")
    captions.append(dict(file="eda_category_distribution.png", question="Is the training data balanced across categories?",
                         finding=f"{len(cs)} categories, {cs.min()}-{cs.max()} complaints each."))

    # 7 complaint length distribution
    lens = c["text"].str.split().str.len()
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    sns.histplot(lens, bins=20, color=BLUE, ax=ax)
    ax.set_xlabel("words per complaint")
    ax.set_title("How long are the complaints?", fontsize=11)
    save(fig, "eda_text_length.png")
    captions.append(dict(file="eda_text_length.png", question="How much text does the model see per complaint?",
                         finding=f"Median {int(lens.median())} words (min {lens.min()}, max {lens.max()})."))

    payload = dict(simulated_monthly=sim, synthetic_text=is_synthetic(c), charts=captions,
                   unavailable=["State-wise grievance statistics (needs DARPG States/UTs report tables)",
                                "Pendency-age distribution (needs the 'pendency by age' tables)",
                                "Redressal-time distribution (needs per-grievance timestamps; restricted data)"],
                   monthly_rows=int(len(m)), departments=int(m["department"].nunique()))
    (RESULTS / "eda_summary.json").write_text(json.dumps(payload, indent=2))
    print("EDA saved:", [x["file"] for x in captions])


if __name__ == "__main__":
    main()

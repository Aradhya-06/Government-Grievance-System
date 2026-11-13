"""
ml/forecasting/forecast.py
====================================
Department-level next-month receipts forecasting + evaluation.

Models (per department):
  * Naive             last observed value              (sanity baseline)
  * Moving average    mean of the last 3 months         (spec baseline)
  * Linear regression receipts ~ time index (last 24 months)  (spec baseline)
  * ARIMA(1,1,1)      only if the series has >= MIN_ARIMA_POINTS months
                      (spec: do not use methods the data cannot support)

Evaluation: ROLLING-ORIGIN (walk-forward) one-step-ahead forecasts over the last
`test_months` months - at each step the model sees only the past.  Metrics: MAE and RMSE
per department and averaged.  Then the next-month forecast of the best model is stored.

Run:  python ml/forecasting/forecast.py
Outputs: results/forecast_metrics.json, results/forecast_<dept>.png (top depts)
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402

from ml.forecasting.monthly_data import is_simulated, load_monthly  # noqa: E402

RESULTS = ROOT / "ml" / "results"
MIN_ARIMA_POINTS = 36
TEST_MONTHS = 6


def f_naive(y):
    return float(y[-1])


def f_ma(y, k=3):
    return float(np.mean(y[-k:]))


def f_reg(y, window=24):
    y = y[-window:]
    x = np.arange(len(y)).reshape(-1, 1)
    m = LinearRegression().fit(x, y)
    return float(m.predict([[len(y)]])[0])


def f_arima(y):
    from statsmodels.tsa.arima.model import ARIMA
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = ARIMA(np.asarray(y, float), order=(1, 1, 1)).fit()
        return float(fit.forecast(1)[0])


def walk_forward(y, fn, test_months):
    preds, actual = [], []
    for t in range(len(y) - test_months, len(y)):
        try:
            preds.append(max(0.0, fn(y[:t])))
        except Exception:
            preds.append(float(y[t - 1]))
        actual.append(float(y[t]))
    return np.array(actual), np.array(preds)


def mae(a, p):
    return float(np.mean(np.abs(a - p)))


def rmse(a, p):
    return float(np.sqrt(np.mean((a - p) ** 2)))


def run():
    df = load_monthly()
    sim = is_simulated(df)
    RESULTS.mkdir(parents=True, exist_ok=True)
    out, per_dept = {}, []
    for dept, g in df.groupby("department"):
        g = g.sort_values("month")
        y = g["receipts"].values.astype(float)
        if len(y) < 12 + TEST_MONTHS:
            per_dept.append(dict(department=dept, skipped=f"only {len(y)} months - too few"))
            continue
        models = {"Naive": f_naive, "Moving average (3)": f_ma, "Linear regression": f_reg}
        if len(y) >= MIN_ARIMA_POINTS:
            models["ARIMA(1,1,1)"] = f_arima
        res, series = {}, {}
        for name, fn in models.items():
            a, p = walk_forward(y, fn, TEST_MONTHS)
            res[name] = dict(mae=mae(a, p), rmse=rmse(a, p))
            series[name] = p
        best = min(res, key=lambda k: res[k]["rmse"])
        nxt = max(0, round(models[best](y)))
        next_month = (g["month"].max() + pd.offsets.MonthBegin(1)).strftime("%Y-%m")
        per_dept.append(dict(department=dept, n_months=len(y), models=res, best=best,
                             next_month=next_month, next_month_forecast=int(nxt),
                             last_actual=int(y[-1]), arima_used="ARIMA(1,1,1)" in res))
        out[dept] = (g, a, series, best)

    ok = [d for d in per_dept if "models" in d]
    names = sorted({m for d in ok for m in d["models"]})
    avg = {n: dict(mae=float(np.mean([d["models"][n]["mae"] for d in ok if n in d["models"]])),
                   rmse=float(np.mean([d["models"][n]["rmse"] for d in ok if n in d["models"]])),
                   n_departments=sum(n in d["models"] for d in ok)) for n in names}
    wins = {n: sum(d["best"] == n for d in ok) for n in names}
    payload = dict(simulated_data=sim, test_months=TEST_MONTHS, min_arima_points=MIN_ARIMA_POINTS,
                   note=("SIMULATED monthly data - for demonstration only" if sim else "Real aggregate data"),
                   average=avg, best_model_counts=wins, departments=per_dept)
    (RESULTS / "forecast_metrics.json").write_text(json.dumps(payload, indent=2))

    # charts: actual vs predicted for the 4 biggest departments
    top = sorted(out, key=lambda d: -out[d][0]["receipts"].mean())[:4]
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), sharex=False)
    for ax, d in zip(axes.ravel(), top):
        g, a, series, best = out[d]
        tail = g.tail(TEST_MONTHS + 12)
        ax.plot(tail["month"], tail["receipts"], color="#2f5d8a", marker="o", ms=3, label="actual")
        ax.plot(g.tail(TEST_MONTHS)["month"], series[best], color="#d98b2b", marker="s", ms=4,
                label=f"predicted ({best})")
        ax.set_title(d[:42], fontsize=9); ax.grid(alpha=.3); ax.tick_params(labelsize=7)
        ax.legend(fontsize=7)
    fig.suptitle("Actual vs predicted monthly receipts (walk-forward)" + (" - SIMULATED DATA" if sim else ""),
                 fontsize=11)
    fig.tight_layout(); fig.savefig(RESULTS / "forecast_actual_vs_pred.png", dpi=140); plt.close(fig)
    return payload


if __name__ == "__main__":
    p = run()
    print("simulated:", p["simulated_data"])
    for n, v in p["average"].items():
        print(f"{n:22s} MAE {v['mae']:8.1f}  RMSE {v['rmse']:8.1f}  ({v['n_departments']} depts)")
    print("best-model counts:", p["best_model_counts"])

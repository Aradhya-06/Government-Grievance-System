# GrievIQ — Government Grievance Intelligence & Prioritization System

A Django application that classifies public grievances, scores their urgency and schedules officer work queues.
Seeded demo complaints are synthetic.

```
Citizen submission → Django + DB → NLP classification → duplicate detection
   → Mamdani fuzzy priority → Genetic-algorithm queue → Officer dashboard
```

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
./run_all.sh            # trains models, runs experiments, migrates, seeds demo data, runs 54 tests
python manage.py runserver     # open http://127.0.0.1:8000
```
Without bash (Windows): run the commands inside `run_all.sh` one by one.

Demo logins (created by `seed_demo`, demo data only):

| Role | Username | Password |
|---|---|---|
| Admin (Django admin + overview) | `admin` | `Admin@123` |
| Officer (one per department) | `officer_mun`, `officer_pwd`, `officer_pwr`, `officer_pen`, … | `Officer@123` |
| Citizen | `citizen1` … `citizen5` | `Citizen@123` |

Officer pages: `/officer/` · Analytics: `/analytics/` · Playground (live fuzzy + classifier): `/analytics/playground/`

## Layout
```
apps/            Django apps: core, accounts, grievances, officers, analytics
  grievances/services/pipeline.py   <- THE integration point (ML -> fuzzy -> GA)
common/catalog.py                    category -> department / severity / SLA / effort assumptions
ml/    preprocessing, classification, similarity, forecasting, eda   (no Django imports)
ga/  fuzzy engine, GA scheduler, baselines, experiments           (no Django imports)
templates/ static/ tests/
```

## Real data vs. demo data
* **Complaint text.** Put the Kaggle CSV (`abhisheksingh016/citizen-grievance-dataset`) in `ml/data/raw/`
  and re-run `train.py`. Text/label columns are auto-detected; extend `CATEGORY_ALIASES` in
  `load_complaints.py` and the `CATEGORIES` table in `common/catalog.py` to match its label names.
  Until then a **synthetic** corpus is used and every page says so. The dataset was NOT verified by the author of this
  code (it could not be downloaded in the build sandbox).
* **Monthly department data.** Download DARPG monthly report PDFs yourself from the official DARPG / PIB pages into
  `ml/data/raw/darpg_pdfs/`, run `python ml/forecasting/monthly_data.py --from-pdfs`, **check the parsed
  CSV by hand**, save it as `monthly_department_grievances.csv`, then re-run forecasting + EDA. Until then the monthly series is
  **simulated** (banner shown). Report editions differ in table layout and some tables are cumulative year-to-date.
* **Restricted DARPG hackathon datasets** (grievance details, movement details) are NOT used.
* Severity per category, SLA days and processing hours in `common/catalog.py` are **assumptions** — check the official
  CPGRAMS timelines before quoting them.

## Honest results (as generated on this machine)
* Classification (synthetic text): ~99 % macro-F1 — optimistic because the text is templated.
* Duplicate detection: text-only F1 0.46 → 0.85 when location is also required to match.
* Forecasting (simulated series): ARIMA RMSE 130.8 vs moving average 137.1, naive 138.4, regression 149.6.
* Scheduling (30 simulated overloaded queues): Fuzzy+GA cut SLA violations 23.7 → 18.3 per queue vs static priority and won on
  cost in 30/30 queues, but did **not** beat plain fuzzy-sort on high-priority waiting time (14.3 vs 12.8 days) and costs ~0.24 s.

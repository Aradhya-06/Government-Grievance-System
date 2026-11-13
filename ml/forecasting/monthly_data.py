"""
ml/forecasting/monthly_data.py
========================================
Provides the department-wise MONTHLY receipts/disposals table used for EDA and
forecasting.  Columns:  month (YYYY-MM-01), department, receipts, disposals, pending, source

Where the numbers come from (in priority order):

 1. ml/data/raw/monthly_department_grievances.csv
    REAL data you prepare from DARPG monthly reports / data.gov.in
    (same columns as above; `pending` optional).  Use build_from_pdfs() below to
    extract tables from the DARPG PDFs you download, then review the output by hand.

 2. SIMULATED DEMO DATA generated here (source = "simulated").
    It exists only so the dashboard and the forecasting code can be demonstrated
    before the real files are collected.  It is NOT CPGRAMS data.  The analytics
    pages show a banner whenever this fallback is in use.

The DARPG PDFs are NOT downloaded by this project (the build sandbox could not
reach them).  Download them yourself from the official DARPG / PIB pages and put
the PDFs in ml/data/raw/darpg_pdfs/ , then run
    python ml/forecasting/monthly_data.py --from-pdfs
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

RAW = ROOT / "ml" / "data" / "raw"
PDF_DIR = RAW / "darpg_pdfs"
CSV = RAW / "monthly_department_grievances.csv"

# Demo departments with an assumed baseline monthly volume (invented scale, for the simulation only).
DEMO_DEPTS = {
    "Department of Posts": 1900, "Department of Telecommunications": 1500,
    "Department of Financial Services (Banking Division)": 1700, "Department of Revenue": 1600,
    "Department of Pension & Pensioners' Welfare": 1300, "Department of Agriculture & Farmers Welfare": 2100,
    "Department of Health & Family Welfare": 900, "Ministry of Railways": 1400,
    "Department of Food & Public Distribution": 500, "Department of Higher Education": 800,
}


def simulate(months: int = 48, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(end=pd.Timestamp.today().normalize().replace(day=1) - pd.offsets.MonthBegin(1),
                        periods=months, freq="MS")
    rows = []
    for dept, base in DEMO_DEPTS.items():
        trend = rng.uniform(-0.002, 0.006)
        season_phase = rng.uniform(0, 2 * np.pi)
        pend = base * rng.uniform(0.8, 1.6)
        for i, m in enumerate(idx):
            level = base * (1 + trend * i) * (1 + 0.10 * np.sin(2 * np.pi * i / 12 + season_phase))
            receipts = max(30, int(rng.normal(level, 0.07 * level)))
            disp = max(20, int(rng.normal(receipts * rng.uniform(0.92, 1.08), 0.05 * receipts)))
            pend = max(0, pend + receipts - disp)
            rows.append(dict(month=m, department=dept, receipts=receipts, disposals=disp,
                             pending=int(pend), source="simulated"))
    return pd.DataFrame(rows)


def load_monthly() -> pd.DataFrame:
    if CSV.exists():
        df = pd.read_csv(CSV, parse_dates=["month"])
        if "pending" not in df.columns:
            df = df.sort_values(["department", "month"])
            df["pending"] = (df.groupby("department")["receipts"].cumsum()
                             - df.groupby("department")["disposals"].cumsum()).clip(lower=0)
        df["source"] = df.get("source", "real")
        return df.sort_values(["department", "month"]).reset_index(drop=True)
    return simulate()


def is_simulated(df: pd.DataFrame) -> bool:
    return bool((df["source"] == "simulated").all())


# --------------------------------------------------------------------------- #
# PDF extraction helper (for REAL data).  Output must be reviewed manually.
# --------------------------------------------------------------------------- #
MONTH_RE = re.compile(r"(January|February|March|April|May|June|July|August|September|October|November|December)"
                      r"[,\s]+(20\d\d)", re.I)


def build_from_pdfs() -> pd.DataFrame:
    """
    Best-effort parser for DARPG 'Central Ministries/Departments' monthly report PDFs.
    It looks for the department table with columns
        Ministry/Department | Brought Forward | Receipts | Disposal | Pending
    on each page and records one row per department per report month.

    LIMITATIONS (be explicit in the report):
      * table layouts changed between report editions - always spot-check rows;
      * some editions show cumulative (year-to-date) figures; if so, difference
        consecutive months of the same year to obtain monthly values;
      * department names are standardised only by whitespace/case here.
    """
    import pdfplumber

    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs found in {PDF_DIR}")
    rows = []
    for pdf_path in pdfs:
        with pdfplumber.open(pdf_path) as pdf:
            text0 = (pdf.pages[0].extract_text() or "") + " " + pdf_path.name
            m = MONTH_RE.search(text0)
            if not m:
                print("skip (month not found):", pdf_path.name)
                continue
            month = pd.to_datetime(f"{m.group(1)} 1 {m.group(2)}")
            for page in pdf.pages:
                for table in page.extract_tables():
                    for r in table:
                        cells = [(c or "").replace("\n", " ").strip() for c in r]
                        nums = [c.replace(",", "") for c in cells if re.fullmatch(r"[\d,]+", c)]
                        names = [c for c in cells if re.search(r"[A-Za-z]{4,}", c)]
                        if len(nums) >= 4 and names:
                            bf, rec, dis, pen = map(int, nums[:4])
                            rows.append(dict(month=month, department=" ".join(names[0].split()),
                                             receipts=rec, disposals=dis, pending=pen,
                                             source=f"pdf:{pdf_path.name}"))
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No tables parsed - inspect the PDFs and adapt build_from_pdfs().")
    df.to_csv(RAW / "monthly_department_grievances_PARSED_REVIEW_ME.csv", index=False)
    print(f"Parsed {len(df)} rows. Review the _PARSED_REVIEW_ME.csv file, fix it, then save as "
          f"monthly_department_grievances.csv")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-pdfs", action="store_true")
    a = ap.parse_args()
    if a.from_pdfs:
        build_from_pdfs()
    else:
        d = load_monthly()
        print(d.head(), "\nsimulated:", is_simulated(d), d.shape)

"""
ml/preprocessing/load_complaints.py
=============================================
Loads the complaint-text dataset into a DataFrame with the columns
    text, category, cluster_id (may be NaN), location (may be empty), source

Priority:
  1. a real CSV in ml/data/raw/ (e.g. the Kaggle file you downloaded)
     - the text and label columns are auto-detected (see _guess_columns)
     - label names are mapped to our 14 catalogue categories via CATEGORY_ALIASES,
       unknown labels are kept as they are (the classifier works with any labels)
  2. otherwise the SYNTHETIC illustrative corpus (clearly flagged source="synthetic")

Preprocessing performed here :
  missing values  -> rows without text or label are dropped (cannot be used to learn)
  duplicates      -> exact duplicate (text, category) rows are dropped once
  text normalise  -> column `clean` added via clean_text()
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml.preprocessing.synthetic_corpus import generate
from ml.preprocessing.text_cleaner import clean_text

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
TEXT_HINTS = ["complaint", "text", "description", "grievance", "message", "content", "body"]
LABEL_HINTS = ["category", "label", "department", "class", "type", "topic"]
CLUSTER_HINTS = ["cluster", "duplicate", "group", "dup"]

# Extend when you see the real label names of the Kaggle file.
CATEGORY_ALIASES: dict[str, str] = {}


def _guess_columns(df: pd.DataFrame):
    low = {c: c.lower() for c in df.columns}

    def pick(hints, exclude=()):
        for h in hints:
            for c, l in low.items():
                if h in l and c not in exclude:
                    return c
        return None

    text = pick(TEXT_HINTS)
    label = pick(LABEL_HINTS, exclude=(text,))
    cluster = pick(CLUSTER_HINTS, exclude=(text, label))
    return text, label, cluster


def load_raw_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    t, l, c = _guess_columns(df)
    if not t or not l:
        raise ValueError(f"Could not detect text/label columns in {path.name}: {list(df.columns)}")
    out = pd.DataFrame({"text": df[t], "category": df[l]})
    out["cluster_id"] = df[c] if c else float("nan")
    loc = next((x for x in df.columns if "location" in x.lower() or "area" in x.lower()
                or "city" in x.lower()), None)
    out["location"] = df[loc] if loc else ""
    out["source"] = f"file:{path.name}"
    return out


def load_complaints(prefer_real: bool = True) -> pd.DataFrame:
    frames = []
    if prefer_real and RAW_DIR.exists():
        for p in sorted(RAW_DIR.glob("*.csv")):
            if p.name.startswith(("monthly", "darpg")):
                continue   # aggregate monthly files, not complaint text
            try:
                frames.append(load_raw_csv(p))
            except ValueError:
                continue
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(generate())
    n0 = len(df)
    if "location" not in df.columns:
        df["location"] = ""
    df["location"] = df["location"].fillna("")
    df = df.dropna(subset=["text", "category"])
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 5]
    df["category"] = df["category"].astype(str).str.strip().replace(CATEGORY_ALIASES)
    df = df.drop_duplicates(subset=["text", "category"]).reset_index(drop=True)
    df["clean"] = df["text"].map(clean_text)
    df = df[df["clean"].str.len() > 0].reset_index(drop=True)
    df.attrs["dropped"] = n0 - len(df)
    return df


def is_synthetic(df: pd.DataFrame) -> bool:
    return bool((df["source"] == "synthetic").all())


if __name__ == "__main__":
    d = load_complaints()
    print(d.shape, "synthetic" if is_synthetic(d) else "real file", "| dropped:", d.attrs["dropped"])
    print(d.head(3).to_string())

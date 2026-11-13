"""
ml/similarity/duplicates.py
=====================================
Similarity-based duplicate / recurring complaint detection.

Method: TF-IDF vectors + cosine similarity.
  * Complaints are NEVER rejected for being similar - we only surface the most
    similar earlier grievances so an officer / citizen can see them, and the
    similar-count feeds the fuzzy "public impact" input (recurrence).
  * Threshold semantics:  sim >= DUPLICATE_T -> "likely duplicate"
                          sim >= RELATED_T   -> "related"

Evaluate with:  python ml/similarity/duplicates.py
  (uses cluster_id ground truth when present: a pair is a TRUE duplicate if both
  complaints share a cluster_id; precision/recall/F1 are computed over pairs).
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.metrics.pairwise import cosine_similarity  # noqa: E402

from ml.preprocessing.text_cleaner import clean_text  # noqa: E402

DUPLICATE_T = 0.50
RELATED_T = 0.35
RESULTS_DIR = ROOT / "ml" / "results"


@dataclass
class Match:
    id: int
    similarity: float
    kind: str        # "duplicate" | "related"


def _norm_loc(x) -> str:
    return " ".join(str(x or "").lower().split())


def find_similar(text: str, corpus: list[tuple], top_k: int = 5,
                 related_t: float = RELATED_T, duplicate_t: float = DUPLICATE_T,
                 location: str | None = None) -> list[Match]:
    """
    corpus = [(grievance_id, raw_text) or (grievance_id, raw_text, location), ...]
    A match is labelled "duplicate" only when the text is very similar AND the
    location agrees (or is unknown); similar text from a different place is at
    most "related" - the same wording in two wards is usually two incidents.
    """
    if not corpus:
        return []
    ids = [c[0] for c in corpus]
    locs = [_norm_loc(c[2]) if len(c) > 2 else "" for c in corpus]
    qloc = _norm_loc(location)
    docs = [clean_text(c[1]) for c in corpus]
    q = clean_text(text)
    if not q or not any(docs):
        return []
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
    mat = vec.fit_transform(docs + [q])
    sims = cosine_similarity(mat[-1], mat[:-1]).ravel()
    out = []
    for i in np.argsort(-sims)[:top_k]:
        s = float(sims[i])
        same_place = (not qloc) or (not locs[i]) or qloc == locs[i]
        if s >= related_t:
            kind = "duplicate" if (s >= duplicate_t and same_place) else "related"
            out.append(Match(ids[i], round(s, 3), kind))
    return out


def pairwise_evaluation(texts: list[str], clusters: list, thresholds=(0.3, 0.4, 0.5, 0.55, 0.6, 0.7),
                        locations: list | None = None):
    """Pair-level P/R/F1 for each threshold using shared cluster ids as ground truth.
    If `locations` is given, a pair is predicted duplicate only when text AND location agree."""
    docs = [clean_text(t) for t in texts]
    mat = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit_transform(docs)
    sim = cosine_similarity(mat)
    n = len(texts)
    iu = np.triu_indices(n, k=1)
    s = sim[iu]
    cl = np.array(clusters)
    truth = (cl[iu[0]] == cl[iu[1]])
    rows = []
    same_loc = None
    if locations is not None:
        loc = np.array([_norm_loc(x) for x in locations])
        same_loc = (loc[iu[0]] == loc[iu[1]])
    for t in thresholds:
        pred = s >= t
        if same_loc is not None:
            pred = pred & same_loc
        tp = int((pred & truth).sum()); fp = int((pred & ~truth).sum()); fn = int((~pred & truth).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        rows.append(dict(threshold=t, precision=round(p, 4), recall=round(r, 4), f1=round(f, 4),
                         tp=tp, fp=fp, fn=fn))
    return rows


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ml.preprocessing.load_complaints import is_synthetic, load_complaints
    df = load_complaints()
    if df["cluster_id"].isna().all():
        print("No duplicate-cluster labels in this dataset - evaluation skipped.")
        return
    texts, clusters = df["text"].tolist(), df["cluster_id"].tolist()
    rows_text = pairwise_evaluation(texts, clusters)
    locs = df["location"].tolist()
    has_loc = sum(bool(str(l).strip()) for l in locs) > 0.5 * len(locs)
    rows = pairwise_evaluation(texts, clusters, locations=locs) if has_loc else rows_text
    print("text only :", max(rows_text, key=lambda r: r["f1"]))
    for r in rows:
        print(r)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    best = max(rows, key=lambda r: r["f1"])
    (RESULTS_DIR / "duplicate_metrics.json").write_text(json.dumps(
        dict(rows=rows, rows_text_only=rows_text, location_aware=has_loc, best=best,
             best_text_only=max(rows_text, key=lambda r: r["f1"]), n=len(df),
             synthetic_data=is_synthetic(df),
             note="Ground truth = shared cluster_id. A pair is a predicted duplicate when cosine >= threshold "
                  "AND the location matches. Synthetic paraphrases are easier than real duplicates."),
        indent=2))
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ts = [r["threshold"] for r in rows]
    for k, c in [("precision", "#2f5d8a"), ("recall", "#d98b2b"), ("f1", "#3a8f5c")]:
        ax.plot(ts, [r[k] for r in rows], marker="o", label=k, color=c)
    ax.set_xlabel("cosine threshold"); ax.set_ylim(0, 1.05); ax.legend(); ax.grid(alpha=.3)
    ax.set_title("Duplicate detection vs threshold")
    fig.tight_layout(); fig.savefig(RESULTS_DIR / "duplicate_thresholds.png", dpi=140)


if __name__ == "__main__":
    main()

"""
ml/classification/train.py
====================================
Trains and COMPARES several text classifiers, then saves the winner.

Run from the project root:
    python ml/classification/train.py

Method
  * features: TF-IDF on unigrams+bigrams of the cleaned text
  * models  : Multinomial Naive Bayes, Logistic Regression, Linear SVM (calibrated),
              MLP neural network (the "ANN" baseline)
  * protocol: stratified 80/20 hold-out for the headline numbers AND 5-fold
              stratified cross-validation (mean +- std) because with only a few
              hundred texts a single split is noisy
  * selection: best CROSS-VALIDATED macro-F1 (never chosen on the test split)
  * outputs  : artifacts/classifier.joblib, results/classification_metrics.json,
               results/confusion_matrix.png, results/model_comparison.png

Honesty note: results on synthetic or very templated text are optimistic and do
not predict performance on real citizen complaints.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import joblib  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.calibration import CalibratedClassifierCV  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (accuracy_score, confusion_matrix,  # noqa: E402
                             precision_recall_fscore_support)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split  # noqa: E402
from sklearn.naive_bayes import MultinomialNB  # noqa: E402
from sklearn.neural_network import MLPClassifier  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.svm import LinearSVC  # noqa: E402

from ml.preprocessing.load_complaints import is_synthetic, load_complaints  # noqa: E402

ARTIFACT_DIR = ROOT / "ml" / "artifacts"
RESULTS_DIR = ROOT / "ml" / "results"
SEED = 42


def make_models():
    def tfidf():
        return TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)

    return {
        "Naive Bayes": Pipeline([("tfidf", tfidf()), ("clf", MultinomialNB(alpha=0.3))]),
        "Logistic Regression": Pipeline([("tfidf", tfidf()),
                                         ("clf", LogisticRegression(max_iter=2000, C=10))]),
        "Linear SVM": Pipeline([("tfidf", tfidf()),
                                ("clf", CalibratedClassifierCV(LinearSVC(C=1.0), cv=3))]),
        "MLP (ANN)": Pipeline([("tfidf", tfidf()),
                               ("clf", MLPClassifier(hidden_layer_sizes=(64,), max_iter=400,
                                                     random_state=SEED, early_stopping=False))]),
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_complaints()
    synthetic = is_synthetic(df)
    X, y = df["clean"].values, df["category"].values
    classes = sorted(set(y))
    print(f"Loaded {len(df)} complaints, {len(classes)} classes, synthetic={synthetic}")

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    report, fitted = {}, {}
    for name, model in make_models().items():
        t0 = time.perf_counter()
        scores = cross_validate(model, X, y, cv=cv, scoring=["accuracy", "f1_macro"])
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        p, r, f, _ = precision_recall_fscore_support(yte, pred, average="macro", zero_division=0)
        report[name] = dict(
            cv_accuracy_mean=float(scores["test_accuracy"].mean()),
            cv_accuracy_std=float(scores["test_accuracy"].std()),
            cv_f1_macro_mean=float(scores["test_f1_macro"].mean()),
            cv_f1_macro_std=float(scores["test_f1_macro"].std()),
            test_accuracy=float(accuracy_score(yte, pred)),
            test_precision_macro=float(p), test_recall_macro=float(r), test_f1_macro=float(f),
            train_time_s=round(time.perf_counter() - t0, 2),
        )
        fitted[name] = (model, pred)
        print(f"{name:22s} CV F1 {report[name]['cv_f1_macro_mean']:.3f} "
              f"+- {report[name]['cv_f1_macro_std']:.3f} | test acc {report[name]['test_accuracy']:.3f}")

    best = max(report, key=lambda k: report[k]["cv_f1_macro_mean"])
    best_model, best_pred = fitted[best]
    print("Selected:", best)

    # confusion matrix of the selected model (hold-out split)
    cm = confusion_matrix(yte, best_pred, labels=classes)
    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(classes)), classes, fontsize=8)
    for i in range(len(classes)):
        for j in range(len(classes)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix - {best} (hold-out 20%)")
    fig.colorbar(im, fraction=0.046); fig.tight_layout()
    fig.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=140); plt.close(fig)

    # model comparison chart
    fig, ax = plt.subplots(figsize=(7.5, 4))
    names = list(report)
    means = [report[n]["cv_f1_macro_mean"] for n in names]
    stds = [report[n]["cv_f1_macro_std"] for n in names]
    ax.bar(names, means, yerr=stds, color="#2f5d8a", capsize=4)
    ax.set_ylim(0, 1.05); ax.set_ylabel("5-fold CV macro-F1")
    ax.set_title("Classifier comparison"); plt.xticks(rotation=15)
    fig.tight_layout(); fig.savefig(RESULTS_DIR / "model_comparison.png", dpi=140); plt.close(fig)

    # per-class report for the selected model
    p, r, f, s = precision_recall_fscore_support(yte, best_pred, labels=classes, zero_division=0)
    per_class = [dict(category=c, precision=float(a), recall=float(b), f1=float(d), support=int(e))
                 for c, a, b, d, e in zip(classes, p, r, f, s)]

    # refit the selected model on ALL data for deployment
    deploy = make_models()[best].fit(X, y)
    joblib.dump({"model": deploy, "classes": classes, "name": best}, ARTIFACT_DIR / "classifier.joblib")

    payload = dict(selected=best, n_samples=len(df), n_classes=len(classes), synthetic_data=synthetic,
                   data_note=("SYNTHETIC illustrative corpus - optimistic estimate" if synthetic
                              else "Real labelled complaint file"),
                   models=report, per_class=per_class)
    (RESULTS_DIR / "classification_metrics.json").write_text(json.dumps(payload, indent=2))
    print("Saved artifacts and results.")


if __name__ == "__main__":
    main()

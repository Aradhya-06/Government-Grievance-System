"""
ml/classification/predict.py
======================================
Loads the saved classifier ONCE (module-level cache) and predicts a category.
Django calls classify(text) for every new grievance - no retraining.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

import joblib

from ml.preprocessing.text_cleaner import clean_text

ARTIFACT = Path(__file__).resolve().parents[1] / "artifacts" / "classifier.joblib"
_lock = threading.Lock()
_bundle = None


@dataclass
class ClassificationResult:
    category: str
    confidence: float
    top3: list          # [(category, probability), ...]
    model_name: str


def _load():
    global _bundle
    if _bundle is None:
        with _lock:
            if _bundle is None:
                if not ARTIFACT.exists():
                    raise FileNotFoundError(
                        "classifier.joblib missing - run: python ml/classification/train.py")
                _bundle = joblib.load(ARTIFACT)
    return _bundle


def reload_model():
    global _bundle
    _bundle = None


def classify(text: str) -> ClassificationResult:
    b = _load()
    cleaned = clean_text(text)
    if not cleaned:
        return ClassificationResult("Unclassified", 0.0, [], b["name"])
    model = b["model"]
    proba = model.predict_proba([cleaned])[0]
    classes = list(model.classes_)
    order = proba.argsort()[::-1][:3]
    top3 = [(classes[i], float(proba[i])) for i in order]
    return ClassificationResult(top3[0][0], top3[0][1], top3, b["name"])

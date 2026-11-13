"""
ml/preprocessing/text_cleaner.py
==========================================
Text normalisation used by BOTH training and prediction (so they can never
drift apart).

Steps and why:
  1. lowercase            - "Water" and "water" are the same word for a classifier
  2. strip URLs/emails/numbers - ids and phone numbers carry no category signal
  3. tokenise (regex)     - no external NLTK data needed, so it runs offline
  4. stop-word removal    - high-frequency words ("the", "is") are noise for TF-IDF
                            BUT negations/urgency words are kept (see KEEP)
  5. stemming (Porter)    - "leaking", "leaks", "leaked" -> "leak"; shrinks vocabulary,
                            which matters with only a few hundred training texts
"""

from __future__ import annotations

import re

from nltk.stem import PorterStemmer   # algorithmic, needs no downloaded corpus

_STEM = PorterStemmer()

STOPWORDS = set("""
a about above after again all am an and any are as at be because been before being below
between both but by can could did do does doing down during each few for from further had
has have having he her here hers him his how i if in into is it its itself just me more
most my myself of off on once only or other our ours out over own same she should so some
such than that the their theirs them then there these they this those through to too under
until up very was we were what when where which while who whom why will with would you your
yours please kindly sir madam respected regarding request kindly pls
""".split())

# Words that look like stop-words but carry meaning for grievances.
KEEP = {"no", "not", "nor", "against", "without"}
STOPWORDS -= KEEP

_URL = re.compile(r"https?://\S+|www\.\S+")
_EMAIL = re.compile(r"\S+@\S+")
_NUM = re.compile(r"\d+")
_TOKEN = re.compile(r"[a-z]+")


def tokenize(text: str) -> list[str]:
    t = (text or "").lower()
    t = _URL.sub(" ", t)
    t = _EMAIL.sub(" ", t)
    t = _NUM.sub(" ", t)
    return _TOKEN.findall(t)


def clean_text(text: str, stem: bool = True) -> str:
    """Full pipeline: raw complaint -> space-separated normalised tokens."""
    toks = [w for w in tokenize(text) if w not in STOPWORDS and len(w) > 1]
    if stem:
        toks = [_STEM.stem(w) for w in toks]
    return " ".join(toks)


if __name__ == "__main__":
    demo = "Respected Sir, there is NO water supply in my area since 5 days!! Pipes are leaking."
    print(clean_text(demo))   # -> "no water suppli area sinc day pipe leak"

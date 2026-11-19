"""Model definition: canonical() -> TF-IDF over character n-grams -> Logistic Regression.

Why this model (see README for the full argument):
  * Kazakh is agglutinative: one root, many suffixes. Character n-grams catch the root
    ("котак" inside "котакбастар") whatever suffix follows.
  * Typos, masks (б*ять) and mixed Cyrillic/Latin share most n-grams with the clean form,
    so unseen spellings still get a score. There is no out-of-vocabulary problem.
  * Logistic Regression gives a calibrated-ish probability and one weight per n-gram,
    so any prediction can be traced back to the n-grams that caused it.
"""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.pipeline import Pipeline

from .text import canonical


def build_pipeline(features: dict, classifier: dict, seed: int) -> Pipeline:
    vec = TfidfVectorizer(
        preprocessor=canonical,                 # same normalisation as the lexicon
        analyzer=features["analyzer"],          # char_wb: n-grams never cross word boundaries
        ngram_range=tuple(features["ngram_range"]),
        min_df=features["min_df"],              # drop n-grams seen in a single comment (noise, names)
        sublinear_tf=features["sublinear_tf"],  # 1+log(tf): "бля бля бля" is not 3x worse
        max_features=features["max_features"],
        lowercase=False,                        # canonical() already lowercases
    )
    clf = LogisticRegression(
        C=classifier["C"],
        class_weight=classifier["class_weight"],  # classes are ~70/30
        max_iter=classifier["max_iter"],
        solver="liblinear",
        random_state=seed,
    )
    return Pipeline([("tfidf", vec), ("clf", clf)])


def choose_threshold(y_true, proba, strategy: str = "max_f1", min_recall: float = 0.95) -> tuple[float, dict]:
    """Pick the decision threshold on VALIDATION data."""
    prec, rec, thr = precision_recall_curve(y_true, proba)
    prec, rec = prec[:-1], rec[:-1]            # last point has no threshold
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    if strategy == "min_recall":
        ok = np.where(rec >= min_recall)[0]
        i = ok[np.argmax(prec[ok])] if len(ok) else int(np.argmax(f1))
    else:
        i = int(np.argmax(f1))
    return float(thr[i]), {"precision": float(prec[i]), "recall": float(rec[i]), "f1": float(f1[i])}


def explain(pipeline: Pipeline, text: str, top: int = 8) -> dict:
    """Trace one comment: canonical text -> active n-grams -> their contribution to the logit."""
    vec, clf = pipeline.named_steps["tfidf"], pipeline.named_steps["clf"]
    x = vec.transform([text])
    names = vec.get_feature_names_out()
    idx = x.indices
    contrib = x.data * clf.coef_[0][idx]
    order = np.argsort(contrib)
    fmt = lambda ids: [(repr(names[idx[i]]), round(float(contrib[i]), 3)) for i in ids]
    logit = float(contrib.sum() + clf.intercept_[0])
    return {
        "canonical": canonical(text),
        "n_active_ngrams": int(len(idx)),
        "intercept": round(float(clf.intercept_[0]), 3),
        "logit": round(logit, 3),
        "probability": round(float(1 / (1 + np.exp(-logit))), 4),
        "pushes_to_profane": fmt(order[::-1][:top]),
        "pushes_to_clean": fmt(order[:top]),
    }

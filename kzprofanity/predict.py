"""Inference with the saved model - no retraining needed.

Command line:
    python -m kzprofanity.predict "Бұл тауар өте нашар"
    python -m kzprofanity.predict "қотақ бас сатушы" --explain        # trace raw text -> prediction
    python -m kzprofanity.predict --interactive                      # type comments one by one
    python -m kzprofanity.predict --file comments.csv --out preds.csv # column "text"
    python -m kzprofanity.predict "..." --threshold 0.5               # override the tuned threshold

Python:
    from kzprofanity.predict import predict
    predict("Бұл тауар өте нашар")   # -> {"label": 0, "probability": 0.02, ...}
"""
from __future__ import annotations

import argparse
import sys
from functools import lru_cache

import joblib
import pandas as pd

from .config import load_config, path
from .model import explain


@lru_cache(maxsize=1)
def load_model(model_path: str | None = None) -> dict:
    p = model_path or path(load_config(), "model_file")
    return joblib.load(p)


def predict(text: str | list[str], threshold: float | None = None, model_path: str | None = None):
    """Return {"text", "label", "probability", "threshold"} for one text, or a list for many."""
    bundle = load_model(model_path)
    thr = bundle["threshold"] if threshold is None else threshold
    texts = [text] if isinstance(text, str) else list(text)
    proba = bundle["pipeline"].predict_proba(texts)[:, 1]
    out = [{"text": t, "label": int(p >= thr), "probability": round(float(p), 4), "threshold": round(thr, 4)}
           for t, p in zip(texts, proba)]
    return out[0] if isinstance(text, str) else out


def _show(text: str, threshold, do_explain: bool) -> None:
    r = predict(text, threshold)
    tag = "1 PROFANE" if r["label"] else "0 clean"
    print(f"{tag:10s} p={r['probability']:.3f} (threshold {r['threshold']:.3f}) | {text}")
    if do_explain:
        e = explain(load_model()["pipeline"], text)
        print(f"  1. canonical text : {e['canonical']!r}")
        print(f"  2. TF-IDF vector  : {e['n_active_ngrams']} non-zero character n-grams")
        print(f"  3. logit          : intercept {e['intercept']} + sum(tfidf * weight) = {e['logit']}")
        print(f"  4. probability    : sigmoid({e['logit']}) = {e['probability']}")
        print(f"  5. label          : {int(e['probability'] >= r['threshold'])}  (p >= {r['threshold']})")
        print("     -> profane :", ", ".join(f"{g}{w:+.2f}" for g, w in e["pushes_to_profane"]))
        print("     -> clean   :", ", ".join(f"{g}{w:+.2f}" for g, w in e["pushes_to_clean"]))


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):  # Windows consoles default to cp1251
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stdin.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Kazakh profanity classifier")
    ap.add_argument("texts", nargs="*", help="comments to classify")
    ap.add_argument("--file", help="CSV with a 'text' column")
    ap.add_argument("--out", help="where to write predictions for --file")
    ap.add_argument("--threshold", type=float, help="override the threshold chosen on validation")
    ap.add_argument("--explain", action="store_true", help="show the n-grams behind each prediction")
    ap.add_argument("--interactive", action="store_true", help="read comments from the keyboard")
    a = ap.parse_args()

    for t in a.texts:
        _show(t, a.threshold, a.explain)
    if a.file:
        df = pd.read_csv(a.file)
        res = pd.DataFrame(predict(df["text"].astype(str).tolist(), a.threshold))
        df["label"], df["probability"] = res["label"].values, res["probability"].values
        if a.out:
            df.to_csv(a.out, index=False, encoding="utf-8")
            print(f"wrote {a.out} ({len(df)} rows, {int(df.label.sum())} profane)")
        else:
            print(df.to_string(index=False))
    if a.interactive:
        print("Type a comment and press Enter (empty line to quit).")
        for line in sys.stdin:
            if not line.strip():
                break
            _show(line.strip(), a.threshold, a.explain)
    if not (a.texts or a.file or a.interactive):
        ap.print_help()


if __name__ == "__main__":
    main()

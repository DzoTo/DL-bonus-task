"""Kazakh profanity classifier for product comments (DL bonus task).

Pipeline (run from the repo root):
    python -m kzprofanity.prepare_data   # download KazSAnDRA, scrub PII, dedupe, pre-label
    python -m kzprofanity.make_splits    # merge manual labels, train/val/test split
    python -m kzprofanity.train          # tune on val, save models/model.joblib
    python -m kzprofanity.evaluate       # one-time test evaluation -> results/
    python -m kzprofanity.predict "текст"
"""

__version__ = "1.0.0"

"""Step 3: tune on validation, fit on train, save the model.

    python -m kzprofanity.train
    python -m kzprofanity.train --no-search      # just use the defaults in config.yaml

The test set is NOT touched here. Model selection and the threshold use validation only.
"""
from __future__ import annotations

import argparse
import itertools
import time
from copy import deepcopy

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from . import __version__
from .config import load_config, path, save_json, set_seed
from .model import build_pipeline, choose_threshold


def mask_augment(df: pd.DataFrame, frac: float, seed: int) -> pd.DataFrame:
    """Copies of a random `frac` of rows from EACH class with one inner letter of some words -> '*'.

    Teaches the model that "б**ть" looks like "блять". Both classes are augmented in the same
    proportion, so '*' by itself does not become a profanity signal. Train split only.
    """
    if frac <= 0:
        return df
    rng = np.random.default_rng(seed)

    def mask(text: str) -> str:
        words = text.split()
        for i, w in enumerate(words):
            if len(w) >= 4 and w.isalpha() and rng.random() < 0.5:
                j = int(rng.integers(1, len(w) - 1))
                words[i] = w[:j] + "*" + w[j + 1:]
        return " ".join(words)

    extra = df.groupby("label", group_keys=False).sample(frac=frac, random_state=seed).copy()
    extra["text"] = extra["text"].map(mask)
    return pd.concat([df, extra], ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-search", action="store_true", help="skip the grid search")
    args = ap.parse_args()

    cfg = load_config()
    seed = cfg["seed"]
    set_seed(seed)
    m = cfg["model"]
    proc = path(cfg, "processed_dir")
    train = pd.read_csv(proc / "train.csv")
    val = pd.read_csv(proc / "val.csv")
    print(f"train {len(train)} rows ({train.label.mean():.1%} profane), val {len(val)} rows")
    n_real = len(train)
    train = mask_augment(train, m.get("augment_mask_frac", 0.0), seed)
    if len(train) > n_real:
        print(f"mask augmentation: +{len(train) - n_real} masked copies (train only)")

    grid = [(m["classifier"]["C"], tuple(m["features"]["ngram_range"]))]
    if not args.no_search:
        grid = list(itertools.product(m["search"]["C"], [tuple(r) for r in m["search"]["ngram_range"]]))

    rows, best = [], None
    for C, ngr in grid:
        feats, clf = deepcopy(m["features"]), deepcopy(m["classifier"])
        feats["ngram_range"], clf["C"] = list(ngr), C
        t0 = time.time()
        pipe = build_pipeline(feats, clf, seed).fit(train.text, train.label)
        proba = pipe.predict_proba(val.text)[:, 1]
        thr, at = choose_threshold(val.label, proba, m["threshold"]["strategy"], m["threshold"]["min_recall"])
        row = {"C": C, "ngram_range": str(ngr), "val_avg_precision": average_precision_score(val.label, proba),
               "threshold": thr, **{f"val_{k}": v for k, v in at.items()},
               "n_features": len(pipe.named_steps["tfidf"].vocabulary_), "fit_seconds": round(time.time() - t0, 1)}
        rows.append(row)
        print(f"  C={C:<5} ngrams={ngr}  val F1={at['f1']:.4f}  AP={row['val_avg_precision']:.4f}  thr={thr:.3f}")
        key = (round(at["f1"], 4), row["val_avg_precision"])
        if best is None or key > best[0]:
            best = (key, pipe, feats, clf, thr, row)

    _, pipe, feats, clf, thr, row = best
    print(f"best: C={clf['C']} ngrams={tuple(feats['ngram_range'])} threshold={thr:.3f} val F1={row['val_f1']:.4f}")

    model_file = path(cfg, "model_file")
    model_file.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "pipeline": pipe,
        "threshold": thr,
        "features": feats,
        "classifier": clf,
        "seed": seed,
        "trained_on_rows": len(train),
        "augment_mask_frac": m.get("augment_mask_frac", 0.0),
        "package_version": __version__,
    }, model_file, compress=3)
    res = path(cfg, "results_dir")
    res.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values("val_f1", ascending=False).to_csv(res / "val_search.csv", index=False)
    save_json({"best": row, "threshold_strategy": m["threshold"]}, res / "train_summary.json")
    print(f"saved {model_file} ({model_file.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()

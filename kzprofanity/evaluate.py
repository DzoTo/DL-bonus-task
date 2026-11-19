"""Step 4: ONE evaluation on the held-out test set with the saved model and its saved threshold.

    python -m kzprofanity.evaluate

Writes to results/:
    metrics.json            precision / recall / F1 for class 1, confusion matrix, n_test, baselines
    metrics.md              the same as a Markdown table (pasted into the README)
    confusion_matrix.png
    pr_curve.png            precision-recall curve on test, threshold marked
    test_predictions.csv    every test row (id, label, probability, prediction) - no text
    errors.csv              all false positives / false negatives WITH text (local only, git-ignored)
    top_ngrams.csv          the n-grams with the largest positive / negative weights
    challenge.csv           predictions on the hand-written challenge set (synthetic, not test data)
"""
from __future__ import annotations

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score, average_precision_score, confusion_matrix, precision_recall_curve,
    precision_recall_fscore_support, roc_auc_score,
)

from .config import load_config, path, save_json  # noqa: E402
from .lexicon import predict_lexicon  # noqa: E402


def scores(y, pred) -> dict:
    p, r, f, _ = precision_recall_fscore_support(y, pred, average="binary", pos_label=1, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4),
            "accuracy": round(accuracy_score(y, pred), 4), "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}


def plot_confusion(cm: np.ndarray, out) -> None:
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center", fontsize=14,
                color="white" if v > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["0 clean", "1 profane"])
    ax.set_yticks([0, 1], ["0 clean", "1 profane"])
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"Test confusion matrix (n={cm.sum()})")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_pr(y, proba, thr: float, out) -> None:
    prec, rec, t = precision_recall_curve(y, proba)
    i = int(np.argmin(np.abs(t - thr)))
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    ax.plot(rec, prec)
    ax.scatter([rec[i]], [prec[i]], color="red", zorder=3, label=f"threshold {thr:.2f}")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title("Test precision-recall curve")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def top_ngrams(pipe, k: int = 30) -> pd.DataFrame:
    names = pipe.named_steps["tfidf"].get_feature_names_out()
    coef = pipe.named_steps["clf"].coef_[0]
    pos, neg = np.argsort(coef)[::-1][:k], np.argsort(coef)[:k]
    return pd.DataFrame({
        "profane_ngram": [repr(names[i]) for i in pos], "profane_weight": coef[pos].round(3),
        "clean_ngram": [repr(names[i]) for i in neg], "clean_weight": coef[neg].round(3),
    })


def main() -> None:
    cfg = load_config()
    bundle = joblib.load(path(cfg, "model_file"))
    pipe, thr = bundle["pipeline"], bundle["threshold"]
    test = pd.read_csv(path(cfg, "processed_dir") / "test.csv")
    res = path(cfg, "results_dir")
    res.mkdir(parents=True, exist_ok=True)

    proba = pipe.predict_proba(test.text)[:, 1]
    pred = (proba >= thr).astype(int)
    y = test.label.values
    m = scores(y, pred)
    m.update({"n_test": len(test), "n_test_profane": int(y.sum()), "threshold": round(thr, 4),
              "roc_auc": round(roc_auc_score(y, proba), 4),
              "average_precision": round(average_precision_score(y, proba), 4)})

    # Baselines: what you get without learning anything.
    lex = test.text.map(predict_lexicon).values
    baselines = {"majority_class_0": scores(y, np.zeros_like(y)), "lexicon_only": scores(y, lex)}
    # Per-group metrics: are the hard (manual) rows harder than the automatic ones?
    test = test.assign(proba=proba.round(4), pred=pred)
    by_source = {s: {"n": len(g), **scores(g.label, g.pred)} for s, g in test.groupby("label_source")}
    by_domain = {s: {"n": len(g), **scores(g.label, g.pred)} for s, g in test.groupby("domain")}

    plot_confusion(confusion_matrix(y, pred, labels=[0, 1]), res / "confusion_matrix.png")
    plot_pr(y, proba, thr, res / "pr_curve.png")
    # committed file: ids only (no comment text in the repository); errors.csv keeps text and is git-ignored
    test.drop(columns="text").to_csv(res / "test_predictions.csv", index=False, encoding="utf-8")
    err = test[test.pred != test.label].copy()
    err["error"] = np.where(err.label == 1, "FN", "FP")
    err["confidence"] = (err.proba - 0.5).abs()
    err.sort_values("confidence", ascending=False).to_csv(res / "errors.csv", index=False, encoding="utf-8")
    top_ngrams(pipe).to_csv(res / "top_ngrams.csv", index=False, encoding="utf-8")

    challenge = {}
    ch_file = path(cfg, "challenge_file")
    if ch_file.exists():
        ch = pd.read_csv(ch_file)
        ch["proba"] = pipe.predict_proba(ch.text)[:, 1].round(4)
        ch["pred"] = (ch.proba >= thr).astype(int)
        ch["lexicon"] = ch.text.map(predict_lexicon)
        ch.to_csv(res / "challenge.csv", index=False, encoding="utf-8")
        challenge = {"model": {"n": len(ch), **scores(ch.label, ch.pred)},
                     "lexicon_only": {"n": len(ch), **scores(ch.label, ch.lexicon)}}

    out = {"test": m, "baselines_on_test": baselines, "by_label_source": by_source,
           "by_domain": by_domain, "challenge_set": challenge,
           "model": {k: bundle[k] for k in ("features", "classifier", "seed", "trained_on_rows")}}
    save_json(out, res / "metrics.json")

    md = ["| | precision | recall | F1 | TP | FP | FN | TN |", "|---|---|---|---|---|---|---|---|"]
    rows = [("**model (test)**", m), ("lexicon only (test)", baselines["lexicon_only"]),
            ("always 0 (test)", baselines["majority_class_0"])]
    if challenge:
        rows += [("model (challenge set)", challenge["model"]), ("lexicon only (challenge set)", challenge["lexicon_only"])]
    md += [f"| {n} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} | {r['tp']} | {r['fp']} | {r['fn']} | {r['tn']} |"
           for n, r in rows]
    (res / "metrics.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"TEST  n={m['n_test']} (profane {m['n_test_profane']})  threshold={thr:.3f}")
    print(f"      precision={m['precision']}  recall={m['recall']}  F1={m['f1']}  AP={m['average_precision']}")
    print(f"      confusion [[TN FP][FN TP]] = [[{m['tn']} {m['fp']}] [{m['fn']} {m['tp']}]]")
    print(f"      lexicon-only F1={baselines['lexicon_only']['f1']}   errors: {len(err)} -> results/errors.csv")
    if challenge:
        print(f"CHALLENGE n={challenge['model']['n']}  model F1={challenge['model']['f1']}  "
              f"lexicon F1={challenge['lexicon_only']['f1']}")


if __name__ == "__main__":
    main()

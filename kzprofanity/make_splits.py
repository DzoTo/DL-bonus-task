"""Step 2: candidates + manual labels -> final dataset and train/val/test splits.

    python -m kzprofanity.make_splits

Label priority: a manual label in data/labeling/to_review.csv always wins over the automatic one.
Rows that need review but have no manual label yet are left out (not guessed).
Optional synthetic rows (data/synthetic/*.csv with columns text,label) go to TRAIN only, so
val/test contain real comments only.

Near-duplicates: exact duplicates were already removed in prepare_data (same letters after
canonical()). Here comments with the same SET of words ("котак бас ойын" / "ойын котак бас",
"керемет" / "керемет керемет") form one group, and a whole group always goes to one split.

Outputs
    data/processed/{dataset,train,val,test}.csv   full text, git-ignored (rebuilt by the scripts)
    data/splits/split_ids.csv                      id,label,label_source,split - committed, no text
    data/labeling/manual_labels.csv                id,label,note of your manual labels - committed
"""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import load_config, path, save_json, set_seed
from .prepare_data import read_review
from .text import canonical, dedupe_key

COLS = ["id", "text", "label", "label_source", "is_synthetic", "stars", "domain"]


def load_labeled(cfg: dict, stats: dict) -> pd.DataFrame:
    cand = pd.read_csv(path(cfg, "interim_dir") / "candidates.csv", dtype={"id": str})
    review = read_review(path(cfg, "labeling_dir") / "to_review.csv")
    review["label"] = review["label"].astype(str).str.strip()
    bad = review[~review["label"].isin(["", "0", "1"])]
    if len(bad):
        raise SystemExit(f"to_review.csv: labels must be 0, 1 or empty. Bad rows:\n{bad[['id', 'label']]}")
    manual = review[review["label"] != ""].set_index("id")["label"].astype(int)
    stats["manual_labels"] = len(manual)
    stats["manual_unlabeled_left_out"] = int((cand["review_reason"].eq("ambiguous_or_borderline")
                                              & ~cand["id"].isin(manual.index)).sum())

    # agreement between automatic labels and the human audit -> estimate of label noise
    audit = cand[cand["review_reason"].fillna("").str.startswith("audit") & cand["id"].isin(manual.index)]
    if len(audit):
        agree = (audit["auto_label"].astype(int).values == manual.loc[audit["id"]].values)
        stats["audit_rows_labeled"] = len(audit)
        stats["audit_agreement"] = round(float(agree.mean()), 4)

    df = cand.copy()
    df["label"] = df["auto_label"]
    has_manual = df["id"].isin(manual.index)
    df.loc[has_manual, "label"] = df.loc[has_manual, "id"].map(manual)
    df.loc[has_manual, "label_source"] = "manual"
    df = df[df["label"].notna()].copy()
    df["label"] = df["label"].astype(int)
    df["is_synthetic"] = 0
    return df[COLS]


def load_synthetic(cfg: dict) -> pd.DataFrame:
    files = sorted(path(cfg, "synthetic_dir").glob("*.csv"))
    if not files:
        return pd.DataFrame(columns=COLS)
    syn = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    syn["id"] = [f"syn{i:05d}" for i in range(len(syn))]
    syn["label_source"] = "synthetic"
    syn["is_synthetic"] = 1
    syn["stars"] = pd.NA
    syn["domain"] = "synthetic"
    return syn[COLS]


def near_dup_key(text: str) -> str:
    return " ".join(sorted(set(canonical(text).split())))


def group_split(df: pd.DataFrame, cfg: dict, seed: int):
    """Stratified split of near-duplicate GROUPS (label x manual/auto), then expand to rows."""
    df = df.assign(group=df["text"].map(near_dup_key))
    g = df.groupby("group").agg(label=("label", "max"),
                                manual=("label_source", lambda s: (s == "manual").any()))
    strat = g["label"].astype(str) + "_" + g["manual"].astype(str)
    s = cfg["split"]
    g_train, g_rest = train_test_split(g.index, test_size=s["val"] + s["test"], stratify=strat, random_state=seed)
    g_val, g_test = train_test_split(g_rest, test_size=s["test"] / (s["val"] + s["test"]),
                                     stratify=strat.loc[g_rest], random_state=seed)
    pick = lambda ids: df[df["group"].isin(set(ids))].drop(columns="group")
    return pick(g_train), pick(g_val), pick(g_test), int((df.groupby("group").size() > 1).sum())


def main() -> None:
    cfg = load_config()
    seed = cfg["seed"]
    set_seed(seed)
    stats: dict = {}
    real = load_labeled(cfg, stats)

    train, val, test, n_groups = group_split(real, cfg, seed)
    stats["near_duplicate_groups"] = n_groups

    syn = load_synthetic(cfg)
    if len(syn):
        test_keys = set(pd.concat([val, test])["text"].map(dedupe_key))
        syn = syn[~syn["text"].map(dedupe_key).isin(test_keys)]
        train = pd.concat([train, syn], ignore_index=True)
    stats["synthetic_rows_in_train"] = len(syn)

    # leakage check: no normalised text / word set may appear in two splits
    keys = {n: set(d["text"].map(dedupe_key)) | set(d["text"].map(near_dup_key))
            for n, d in [("train", train), ("val", val), ("test", test)]}
    overlap = len(keys["train"] & keys["test"]) + len(keys["train"] & keys["val"]) + len(keys["val"] & keys["test"])
    assert overlap == 0, f"{overlap} duplicated texts across splits"

    out = path(cfg, "processed_dir")
    out.mkdir(parents=True, exist_ok=True)
    full = pd.concat([train.assign(split="train"), val.assign(split="val"), test.assign(split="test")])
    full.to_csv(out / "dataset.csv", index=False, encoding="utf-8")
    for name, d in [("train", train), ("val", val), ("test", test)]:
        d.to_csv(out / f"{name}.csv", index=False, encoding="utf-8")
        stats[name] = {"rows": len(d), "label_0": int((d["label"] == 0).sum()),
                       "label_1": int((d["label"] == 1).sum())}
    stats["label_source_counts"] = full["label_source"].value_counts().to_dict()
    save_json(stats, out / "split_stats.json")

    # committed, text-free artefacts (no personal data in the repository)
    split_dir = out.parent / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    full[["id", "label", "label_source", "is_synthetic", "split"]].to_csv(split_dir / "split_ids.csv", index=False)
    save_json(stats, split_dir / "split_stats.json")
    review = read_review(path(cfg, "labeling_dir") / "to_review.csv")
    labeled = review[review["label"].astype(str).str.strip() != ""][["id", "label", "note"]]
    labeled.to_csv(path(cfg, "labeling_dir") / "manual_labels.csv", index=False, encoding="utf-8")
    for k, v in stats.items():
        print(f"  {k:26s} {v}")


if __name__ == "__main__":
    main()

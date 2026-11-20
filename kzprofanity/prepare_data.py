"""Step 1: KazSAnDRA -> candidate rows with automatic labels + a manual review queue.

    python -m kzprofanity.prepare_data

Outputs
    data/interim/candidates.csv   every selected row with its automatic label (or empty = needs review)
    data/interim/prepare_stats.json
    data/labeling/to_review.csv   rows a human must label (label column empty). Labels you already
                                  typed are carried over when this script is re-run.
"""
from __future__ import annotations

import io
import zipfile

import pandas as pd
import requests

from .config import load_config, path, save_json, set_seed
from .lexicon import find_hits, lexicon_label
from .text import canonical, dedupe_key, letter_count, scrub_pii

REVIEW_COLUMNS = ["id", "text", "stars", "domain", "hits", "review_reason", "suggested", "why", "label", "note"]


def download(cfg: dict) -> pd.DataFrame:
    raw_dir = path(cfg, "raw_dir")
    raw_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for name in cfg["source"]["files"]:
        zpath = raw_dir / f"{name}.zip"
        if not zpath.exists():
            url = f"{cfg['source']['base_url']}/{name}.zip"
            print(f"downloading {url}")
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            zpath.write_bytes(r.content)
        with zipfile.ZipFile(zpath) as z:
            frames.append(pd.read_csv(io.BytesIO(z.read(f"{name}.csv"))))
    df = pd.concat(frames, ignore_index=True)
    # KazSAnDRA score labels are 0..4 -> stars 1..5
    return pd.DataFrame({
        "id": df["custom_id"].astype(str),
        "text": df["text"].astype(str),
        "stars": df["label"].astype(int) + 1,
        "domain": df["domain"].astype(str),
    })


def clean(df: pd.DataFrame, cfg: dict, stats: dict) -> pd.DataFrame:
    stats["loaded"] = len(df)
    df = df.copy()
    df["text"] = df["text"].map(scrub_pii)
    stats["pii_rows_scrubbed"] = int(df["text"].str.contains(r"<(?:URL|EMAIL|PHONE|USER)>").sum())
    df = df[df["text"].map(letter_count) >= cfg["prepare"]["min_letters"]]
    df["canon"] = df["text"].map(canonical)
    df["key"] = df["text"].map(dedupe_key)
    df = df[df["key"].str.len() > 0]
    before = len(df)
    df = df.drop_duplicates("key", keep="first")
    stats["duplicates_removed"] = before - len(df)
    stats["after_clean_dedupe"] = len(df)
    return df.reset_index(drop=True)


def scan(df: pd.DataFrame) -> pd.DataFrame:
    hits = df["canon"].map(lambda s: find_hits(s, already_canonical=True))
    df = df.copy()
    df["hits"] = hits.map(lambda hs: "; ".join(f"{h.rule}:{h.match}" for h in hs))
    df["hit_groups"] = hits.map(lambda hs: ",".join(sorted({h.group for h in hs})))
    df["lex_label"] = hits.map(lexicon_label)          # 1 / None / 0
    df["has_insult"] = df["hit_groups"].str.contains("insult")
    return df


def select(df: pd.DataFrame, cfg: dict, stats: dict) -> pd.DataFrame:
    p, seed = cfg["prepare"], cfg["seed"]
    strong = df[df["lex_label"] == 1].assign(auto_label=1, label_source="auto_lexicon", review_reason="")
    review = df[df["lex_label"].isna()].assign(auto_label=pd.NA, label_source="manual",
                                               review_reason="ambiguous_or_borderline")
    clean_pool = df[df["lex_label"] == 0]

    hard = clean_pool[clean_pool["has_insult"]]
    hard = hard.sample(min(len(hard), p["n_hard_negatives"]), random_state=seed)
    rest = clean_pool.drop(hard.index)
    angry = rest[rest["stars"] <= 2]
    angry = angry.sample(min(len(angry), p["n_angry_negatives"]), random_state=seed)
    other = rest[rest["stars"] >= 3]
    other = other.sample(min(len(other), p["n_other_negatives"]), random_state=seed)
    negatives = pd.concat([
        hard.assign(label_source="auto_hard_negative"),
        angry.assign(label_source="auto_clean_angry"),
        other.assign(label_source="auto_clean"),
    ]).assign(auto_label=0, review_reason="")

    # Audit: a random sample of automatic labels also goes to a human, to measure label noise.
    n_audit = p["n_audit_per_class"]
    audit_pos = strong.sample(min(n_audit, len(strong)), random_state=seed).index
    audit_neg = negatives.sample(min(n_audit, len(negatives)), random_state=seed).index
    strong.loc[audit_pos, "review_reason"] = "audit_auto_1"
    negatives.loc[audit_neg, "review_reason"] = "audit_auto_0"

    out = pd.concat([strong, review, negatives], ignore_index=True)
    stats.update({
        "lexicon_strong_hits": len(strong),
        "lexicon_review_hits": len(review),
        "clean_pool": len(clean_pool),
        "insult_pool": int(clean_pool["has_insult"].sum()),
        "sampled_hard_negatives": len(hard),
        "sampled_angry_negatives": len(angry),
        "sampled_other_negatives": len(other),
        "audit_rows": len(audit_pos) + len(audit_neg),
        "candidates": len(out),
    })
    return out


def write_review_queue(cand: pd.DataFrame, cfg: dict, stats: dict) -> None:
    lab_dir = path(cfg, "labeling_dir")
    lab_dir.mkdir(parents=True, exist_ok=True)
    out_file = lab_dir / "to_review.csv"
    queue = cand[cand["review_reason"] != ""][["id", "text", "stars", "domain", "hits", "review_reason"]].copy()
    for c in ("suggested", "why", "label", "note"):
        queue[c] = ""
    # optional pre-filled suggestions (columns suggested/why) survive a re-run; they are hints only,
    # make_splits uses nothing but the "label" column you fill in
    if out_file.exists():
        prev = read_review(out_file).set_index("id")
        for c in ("suggested", "why"):
            if c in prev:
                queue[c] = queue["id"].map(prev[c]).fillna("")
    # carry over labels already typed in a previous run (to_review.csv locally, or the committed
    # manual_labels.csv after a fresh clone)
    sources = [f for f in (lab_dir / "manual_labels.csv", out_file) if f.exists()]
    if sources:
        old = pd.concat([read_review(f) for f in sources]).drop_duplicates("id", keep="last")
        old = old[old["label"].astype(str).str.strip() != ""].set_index("id")
        mask = queue["id"].isin(old.index)
        queue.loc[mask, "label"] = queue.loc[mask, "id"].map(old["label"])
        queue.loc[mask, "note"] = queue.loc[mask, "id"].map(old["note"]).fillna("")
        stats["review_labels_carried_over"] = int(mask.sum())
    order = {"ambiguous_or_borderline": 0, "audit_auto_1": 1, "audit_auto_0": 2}
    queue = queue.sort_values(["review_reason", "hits"], key=lambda s: s.map(order) if s.name == "review_reason" else s)
    # utf-8-sig so Excel shows Kazakh letters correctly
    queue[REVIEW_COLUMNS].to_csv(out_file, index=False, encoding="utf-8-sig")
    stats["review_queue"] = len(queue)
    print(f"wrote {out_file} ({len(queue)} rows to label)")


def read_review(p) -> pd.DataFrame:
    """Read to_review.csv whether it was saved by pandas, Excel (; separator) or Google Sheets."""
    df = pd.read_csv(p, sep=None, engine="python", encoding="utf-8-sig", dtype=str, keep_default_na=False)
    for c in ("suggested", "why", "label", "note"):
        if c not in df:
            df[c] = ""
    return df


def main() -> None:
    cfg = load_config()
    set_seed(cfg["seed"])
    stats: dict = {}
    df = clean(download(cfg), cfg, stats)
    df = scan(df)
    cand = select(df, cfg, stats)
    interim = path(cfg, "interim_dir")
    interim.mkdir(parents=True, exist_ok=True)
    cols = ["id", "text", "stars", "domain", "hits", "hit_groups", "auto_label", "label_source", "review_reason"]
    cand[cols].to_csv(interim / "candidates.csv", index=False, encoding="utf-8")
    write_review_queue(cand, cfg, stats)
    save_json(stats, interim / "prepare_stats.json")
    for k, v in stats.items():
        print(f"  {k:28s} {v}")


if __name__ == "__main__":
    main()

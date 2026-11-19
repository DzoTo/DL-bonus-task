# Kazakh Profanity Detection

DSSC5113 bonus project (KBTU, Fall 2026). An individual project by **`<FULL NAME>`**, student ID `<STUDENT_ID>`.

The model reads one Kazakh product review or comment and returns:

* **label** `0` = no profanity, `1` = profanity present
* **probability** of class 1, and the decision **threshold** that was tuned on validation data

```
$ python -m kzprofanity.predict "Тауар өте нашар, ақша босқа кетті" "Қ*тақ бас сатушы"
0 clean    p=0.003 (threshold 0.415) | Тауар өте нашар, ақша босқа кетті
1 PROFANE  p=0.971 (threshold 0.415) | Қ*тақ бас сатушы
```

> **Status:** the pipeline runs end to end. The numbers below come from the automatic labels. The
> manual review of 372 hard and audit rows (`data/labeling/to_review.csv`) is still in progress.
> After it is done, the test set is evaluated **once** and this README is updated.

---

## 1. Quick start

```bash
git clone <this repo> && cd DL-bonus-task
python -m venv .venv
.venv\Scripts\activate            # Windows  (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt

# use the saved model, no retraining needed
python -m kzprofanity.predict "сатушы далбайоп"
python -m kzprofanity.predict "сатушы далбайоп" --explain    # trace: text -> n-grams -> probability -> label
python -m kzprofanity.predict --interactive                  # type your own comments
python -m kzprofanity.predict --file my.csv --out preds.csv  # CSV with a "text" column
```

From Python:

```python
from kzprofanity.predict import predict
predict("Бұл тауар өте нашар")   # {'text': ..., 'label': 0, 'probability': 0.01, 'threshold': 0.41}
```

The trained model is `models/model.joblib` (0.6 MB). It is committed to the repository and holds the
fitted TF-IDF vocabulary, the classifier, the threshold and the settings. Nothing else is needed for
inference. A Colab version is in [`notebooks/colab.ipynb`](notebooks/colab.ipynb).

## 2. Reproduce everything (about 1 minute on a laptop)

```bash
python -m kzprofanity.prepare_data   # download KazSAnDRA, scrub personal data, dedupe, pre-label
python -m kzprofanity.make_splits    # apply manual labels, group near-duplicates, 70/15/15 split
python -m kzprofanity.train          # grid search on validation, pick threshold, save model
python -m kzprofanity.evaluate       # single test evaluation -> results/
python -m pytest -q                  # 160+ checks: lexicon rules, normalisation, PII, predict()
```

All settings are in [`config.yaml`](config.yaml): seed `42`, split ratios, sample sizes, the
n-gram/C grid and the threshold strategy. Package versions are pinned in `requirements.txt`
(Python 3.10).

## 3. Task definition and labeling rules

**1 = the comment contains profanity (мат)**, Kazakh or Russian, in any spelling. **0 = everything else**,
including very negative or rude reviews.

| Case | Rule | Example → label |
|---|---|---|
| Kazakh obscene words | 1 | `қотақбас ойын` → 1, `шешеңнің амы` → 1, `сігейін` → 1 |
| Mixed language | KazSAnDRA comments are code-switched Kazakh/Russian. Russian мат *inside a Kazakh comment* counts as profanity, because readers see it the same way. | `ойын пиздец екен` → 1 |
| Masked words | A mask (`*`, `#`, `@`, digits) that hides a letter of a profane word still counts. | `б*ять`, `к0так`, `сі#ейын` → 1 |
| Spelling variants | Latin script, look-alike letters, repeated letters and spaced letters all count. | `qotaq`, `cyka`, `котаааак`, `с у к а` → 1 |
| Negative but clean | Anger, complaints and low ratings are **not** profanity. | `Тауар өте нашар, ақша босқа кетті` → 0 |
| Insults that are not мат | Rude, but not obscene. | `ақымақ`, `есек`, `оңбаған`, `мал`, `дебил`, `идиот` → 0 |
| Innocent look-alikes | Same letters, different word. | `көтеру`, `адам`, `аман`, `қатынас`, `педиатр`, `сукно`, `бильярд` → 0 |
| Borderline vulgar words | `жопа, хер, қатын, жезөкше, боқ, говно, сайқал, долбан`: **decided per comment** by a human. Default 1 if used as swearing, 0 if used literally. | `қатын` (wife, literal) → 0 |
| Ambiguous / typos | Decided by a human. Typos that only *look* profane are 0. | `сіге` (typo of `сізге`) → 0, `қаямын` (typo of `қоямын`) → 0 |

Labeled examples from the dataset:

| Comment (from KazSAnDRA) | Label | Why |
|---|---|---|
| `Қотағымды жегер ойын мынау` | 1 | Kazakh obscene word with a suffix |
| `Сөз жоқ пиздес приложение` | 1 | Russian мат in a Kazakh sentence |
| `Шшс пздц кызык ойын` | 1 | abbreviations of obscene phrases |
| `Шешең аман болсын осы, ойынды шығарған адам` | 0 | "may your mother be well": `шешең` + `аман` is not `шешең амы` |
| `Мен мына ақымақтарға айтайын ойнай алмасаңдар...` | 0 | insult, not obscene |
| `Тез желінеді баллон, не стоит, качество так себе` | 0 | negative review, clean |

## 4. Dataset

**Source:** [KazSAnDRA](https://github.com/IS2AI/KazSAnDRA) (ISSAI, Nazarbayev University, CC BY 4.0).
It contains 175,158 real Kazakh reviews from app stores, online markets (`market`), maps and bookstores, with
1-5 star ratings. The rows are reviews of products, apps and services, so the data matches the task domain.
`prepare_data.py` downloads the score files `06_sc_train_ib`, `09_sc_valid` and `10_sc_test` automatically.

**Its labels do not fit this task.** KazSAnDRA is labeled for *sentiment*, and a 1-star review is not
profane. The star ratings were ignored. Every row in this project got a new profanity label:

```
KazSAnDRA 175,158 reviews
  │  scrub personal data (URL/e-mail/phone/@user -> tags)          406 rows changed
  │  remove duplicates (same letters after normalisation)       -12,010
  ▼  162,989 unique reviews
  │  scan with the lexicon (kzprofanity/lexicon.py, 45 rules)
  ├── strong hit (unambiguous мат)            -> label 1 automatically          ~2,240
  ├── only ambiguous/borderline hit           -> MANUAL label                    ~310
  ├── no hit, contains a non-obscene insult   -> label 0 (hard negative)         ~380
  ├── no hit, 1-2 stars (angry)               -> label 0, random 2,500
  ├── no hit, 3-5 stars                       -> label 0, random 2,500
  └── audit: 30 random auto-1 + 30 random auto-0 -> MANUAL label, to measure label noise
```

The lexicon only *finds candidates*. It is **not** the classifier and **not** a model feature. It was
built from my own list of Kazakh swear words, extended with variants found in the corpus. Its
**148 self-tests** (`python -m kzprofanity.lexicon`) cover each rule and the innocent look-alikes.
I checked random samples of every rule by hand. Several rules were narrowed after the checks
because they caught typos (`сіге`, `сиқты`, `қаямын`, `сігіспей`) or innocent words (`бильярд`,
`шешең аман`, `ámın` = "amen").

**Synthetic data:** none in train, validation or test. The only synthetic file is
`data/challenge/challenge.csv`: 40 sentences I wrote to stress-test masks, spelling variants and
look-alikes. It is reported separately and never used for training or model selection.

**Splits** (seed 42, stratified by label × manual/automatic). Current numbers, before the manual labels:

| split | rows | label 0 | label 1 |
|---|---|---|---|
| train | 5,3xx | ~3,770 | ~1,570 |
| validation | 1,14x | ~806 | ~336 |
| test | 1,14x | ~807 | ~334 |

The exact numbers are in `data/splits/split_stats.json`.

* **Duplicates** are removed before splitting. Two comments count as duplicates if they have the same letters after
  `canonical()`, so case, punctuation, emoji and repeated letters are ignored.
* **Near-duplicates** (the same *set* of words, e.g. `керемет ойын` / `ойын керемет керемет`) form a
  group, and the whole group goes to one split. `make_splits.py` asserts that no text or word set occurs
  in two splits.
* **Vocabulary** (TF-IDF) is fitted on the training split only, inside the sklearn `Pipeline`.

**Personal data.** The repository does **not** contain comment texts. `data/raw`, `data/interim` and
`data/processed` are git-ignored and rebuilt by the scripts. What is committed is
`data/splits/split_ids.csv` (id, label, split) and `data/labeling/manual_labels.csv` (id, label, note),
which are enough to rebuild the exact dataset from KazSAnDRA. Links, e-mails, phone numbers and @usernames
are replaced with `<URL>`, `<EMAIL>`, `<PHONE>` and `<USER>` before anything is saved. Comment examples in this
README were picked so that they contain no names.

## 5. Model

**TF-IDF over character n-grams + Logistic Regression** (`kzprofanity/model.py`). The reasons for this choice:

* **Kazakh is agglutinative.** `қотақ`, `қотақбас`, `қотағымның` and `қотақбастарға` share the root's n-grams, so
  unseen suffixes still score high. A word-level model would treat each form as a new word.
* **Spelling noise is the norm.** Masks, Latin letters and typos change one or two characters, and most n-grams survive.
* **No unknown-token problem.** Any new word is split into n-grams, and n-grams that were never seen in
  training simply get no weight.
* **Explainable.** Each prediction is a sum of `tfidf(n-gram) × weight` (see `--explain`). That's the
  "trace one comment" defense question. Trains in 2 s, model is 0.6 MB.
* A fine-tuned transformer (e.g. `xlm-roberta-base`) might handle context better. But it needs a GPU and is
  much harder to trace, and the task asks for one well-understood model.

**How text becomes model input:**

1. `canonical()` (`kzprofanity/text.py`): lowercase → Kazakh Latin and Latin→Cyrillic (`q→к`, `sh→ш`) →
   digits/symbols inside words (`0→о`, `@→а`, `#→*`) → `қ→к, і→и, ң→н, ө→о, ұ/ү→у, ә→а, ғ→г, щ→ш`, drop `ь/ъ`
   → glue spaced letters (`с у к а→сука`) → punctuation and emoji → space (`*` is kept as a mask
   signal) → collapse repeats (`жаааман→жаман`). These are fixed rules with nothing learned, so there is no leakage.
2. `TfidfVectorizer(analyzer="char_wb", ngram_range=(2,5), min_df=2, sublinear_tf=True)`. `char_wb` means the
   n-grams stay inside words (padded with spaces).
3. `LogisticRegression(C=100, class_weight="balanced", solver="liblinear")`. The objective is L2-regularised
   log-loss, with class weights for the 70/30 imbalance.

**Long comments:** there is no truncation, because the whole comment becomes one vector. The side effect is that TF-IDF
L2-normalisation *dilutes* one swear word in a long comment (see errors below).
**Punctuation:** removed, except `*`. **Repeated punctuation and emoji:** removed.

**Training-time augmentation** (`augment_mask_frac: 0.3`): 30% of training rows *of each class* get a
copy where some inner letters are replaced by `*`. This teaches that `б**ть` ≈ `блять`. Both classes are
masked, so `*` alone does not mean "profane". It is applied to the training split only.

**Model selection (validation only):** grid `C ∈ {0.1, 1, 10, 30, 100}` × n-grams `{(1,3), (2,5), (3,6)}`.
The best pair by validation F1 was chosen, with ties broken by average precision. The **threshold** is the one that maximises F1 on
validation (`results/val_search.csv`, `results/train_summary.json`).

## 6. Results (test set, evaluated once)

<!-- results/metrics.md is regenerated by evaluate.py; paste it here after the final run -->
| | precision | recall | F1 | TP | FP | FN | TN |
|---|---|---|---|---|---|---|---|
| **model (test)** | 0.997 | 0.958 | 0.977 | 320 | 1 | 14 | 806 |
| lexicon only (test) | 1.000 | 1.000 | 1.000 | 334 | 0 | 0 | 807 |
| always 0 (test) | 0.000 | 0.000 | 0.000 | 0 | 0 | 334 | 807 |
| model (challenge set) | 1.000 | 0.700 | 0.824 | 14 | 0 | 6 | 20 |
| lexicon only (challenge set) | 1.000 | 0.850 | 0.919 | 17 | 0 | 3 | 20 |

n_test = 1,141 (334 profane). Confusion matrix: `results/confusion_matrix.png`; PR curve: `results/pr_curve.png`;
all predictions (ids only): `results/test_predictions.csv`; errors with text: `results/errors.csv` (created locally by `evaluate`, git-ignored); predictions on the challenge set: `results/challenge.csv`; most important n-grams: `results/top_ngrams.csv`.

![confusion matrix](results/confusion_matrix.png)

**How to read the baselines.** "Lexicon only" scores 1.0 on test *by construction*: most test labels were
produced by the same lexicon. This is the main limitation of the dataset. The test set therefore measures
how well the model *reproduces* the labeling procedure, while the challenge set and the error analysis show
how it *generalises*. The manually labeled rows are the honest part of the test set. `results/metrics.json` →
`by_label_source` reports them separately.

**False positive vs false negative in moderation:**

* **FP** (clean comment flagged): a customer's legitimate, possibly angry, opinion is hidden. This is unfair to
  the customer and to the seller's reputation. It is costly if comments are auto-deleted.
* **FN** (profanity published): an obscene comment is shown to all buyers, which hurts the platform.
* For a store, a sensible setup is **auto-hide at high probability and send the middle band to a human**.
  If comments are only hidden pending review, recall matters more, so set `threshold.strategy: min_recall` in
  `config.yaml`.

## 7. Error analysis

*Draft, based on development runs. Final version after the last evaluation: see `results/errors.csv`.*

1. **Label noise: the model is right and the label is wrong.** `Шешең аман болсын` ("may your mother be
   well") was auto-labeled 1, because the rule `шешең + ам` had no word-end check. The model gave it p≈0.14.
   Similar cases: `ámın` ("amen" in Kazakh Latin) matched `амың`, and `*здравствуйте` matched the masked `п*зд`.
   *Fix applied:* tighter rules and a self-test for each case. *Lesson:* lexicon labels need a human audit
   (the 60 audit rows estimate how often this happens).
2. **The model catches profanity the lexicon missed.** Examples labeled 0 but predicted 1: `Шешенамы` (two
   words glued), `сі#ейын`, `За#бал` (`#` as a mask), `питараздар` (`д→т`), `е блан`. These are the strongest evidence
   that the model learned sub-word patterns and not a word list. *Fix:* added `#` as a mask and glued
   forms, and the remaining cases go to manual labeling.
3. **One swear word inside a long comment.** E.g. `Жетылми калган гондондар кашан дамиды ...` (long review,
   p≈0.09). TF-IDF averages over all n-grams, so one strong word is diluted by many clean ones.
   *Improvement:* score each sentence or window of words separately and take the maximum, or give the
   model the maximum n-gram weight as an extra feature.
4. **Innocent words that contain a profane n-gram.** `АЛЛАХУ АКБАР АМИН` contains `ху` and `ам`.
   `dalbasalap` ("flailing about") shares `далб` with `далбаеб`. *Improvement:* more hard negatives with
   these n-grams; a word-level feature next to the char n-grams.
5. **Masks the model has rarely seen** (challenge set): `б**ть` (two masks), `х*йня`. The mask augmentation raised challenge
   F1 from 0.82 to 0.86-0.89 in development runs, but double masks are still weak. *Improvement:* augment with 1-3
   masks per word.

## 8. Limitations

* Most positive labels come from a lexicon, so test metrics are optimistic (see §6). The manual labels and the
  challenge set are the honest measurements.
* The lexicon reflects my knowledge of Kazakh/Russian slang. Rare regional words are missing, and so are words
  written in a way that no rule normalises (`пидоp` with a Latin `p` becomes `пидоп`).
* The domain is mostly app-store reviews (about 89% of test). There is little data from `market`, the closest domain to
  "product reviews".
* `canonical()` maps `қ→к`, `і→и` and so on, so a few Kazakh/Russian word pairs become identical (`көт`/`кот`).
* The test split was inspected during development to find **labeling** bugs (items 1-2 above). Model settings
  were never tuned on it. The final evaluation is run once after labeling is complete.

## 9. Repository layout

```
kzprofanity/        package: text.py (normalisation, PII), lexicon.py, prepare_data.py, make_splits.py,
                    model.py, train.py, evaluate.py, predict.py, config.py
config.yaml         every setting + seed
data/challenge/     40 hand-written stress-test sentences (synthetic, not used for training)
data/labeling/      manual_labels.csv (committed); to_review.csv (local, git-ignored)
data/splits/        split_ids.csv + split_stats.json (committed, no text)
models/model.joblib trained model (vectorizer + classifier + threshold)
results/            metrics, confusion matrix, PR curve, predictions, errors, top n-grams
tests/              pytest suite
notebooks/          Colab notebook
```

## 10. AI assistance and my contribution

AI tools were allowed for this project. I used an AI coding assistant (a large language model): it wrote most of
the Python code, the regex patterns for the lexicon and the drafts of this README, following my instructions.

**My own work:**
* the initial list of Kazakh swear words and the decision to count Russian мат in Kazakh comments
* the labeling rules (§3) and the manual labels in `data/labeling/manual_labels.csv`
* checking the lexicon samples, and choosing what counts as borderline
* reviewing and running all code, the final evaluation, and the error analysis

I can explain every step of the pipeline at the defense.

## References

* Yeshpanov, R., Varol, H. A. *KazSAnDRA: Kazakh Sentiment Analysis Dataset of Reviews and Attitudes*. LREC-COLING 2024. Dataset: <https://github.com/IS2AI/KazSAnDRA> (CC BY 4.0).
* scikit-learn: `TfidfVectorizer`, `LogisticRegression`.

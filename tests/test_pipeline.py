"""Fast checks: python -m pytest -q"""
from pathlib import Path

import pytest

from kzprofanity.lexicon import SELF_TEST, find_hits, lexicon_label
from kzprofanity.text import canonical, dedupe_key, scrub_pii

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("text,expected", SELF_TEST)
def test_lexicon(text, expected):
    assert lexicon_label(find_hits(text)) == expected


@pytest.mark.parametrize("raw,canon", [
    ("Қотақ", "котак"), ("qotaq", "котак"), ("к0так", "котак"), ("КОТАААК", "котак"),
    ("с у к а", "сука"), ("б*ять!!!", "б*ят"), ("100 тг", "тг"), ("Шешеңнің", "шешенин"),
    ("ба7амын", "бамын"),
])
def test_canonical(raw, canon):
    assert canonical(raw) == canon


def test_dedupe_key_ignores_case_spacing_and_repeats():
    assert dedupe_key("Керемет!!! ойын") == dedupe_key("кереемет ойын") == dedupe_key("КЕРЕМЕТ  ОЙЫН")


def test_email_fully_removed():
    assert scrub_pii("test.user@mail.ru") == "<EMAIL>"


@pytest.mark.parametrize("raw", [
    "звоните +7 701 234 56 78", "пишите на test.user@mail.ru", "сайт https://kaspi.kz/shop", "@user_name айтты",
])
def test_scrub_pii(raw):
    out = scrub_pii(raw)
    assert any(tag in out for tag in ("<PHONE>", "<EMAIL>", "<URL>", "<USER>")), out


@pytest.mark.skipif(not (ROOT / "models" / "model.joblib").exists(), reason="model not trained yet")
def test_predict_roundtrip():
    from kzprofanity.predict import predict
    r = predict("Тауар өте нашар, ақша босқа кетті")
    assert set(r) == {"text", "label", "probability", "threshold"}
    assert r["label"] == 0
    assert predict("қотақбас сатушы")["label"] == 1

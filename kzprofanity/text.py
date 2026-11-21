"""Text normalisation shared by the lexicon, the dataset builder and the model.

canonical() maps the many ways people write the same word onto one form:
    "Қотақ", "qotaq", "к0так", "котаааак"  ->  "котак"
Steps (order matters):
    1. lowercase + Unicode NFC
    2. Kazakh Latin diacritics and Latin multi-letter transliterations (sh, zh, ch, ya ...)
    3. single Latin letters -> Cyrillic (both look-alikes like c/y/k and transliteration like q)
    4. digits/symbols inside words -> letters (0->о, 3->з, @->а ...), other digits in words dropped
    5. Kazakh-specific letters -> nearest Russian letter (қ->к, і->и, ң->н ...), щ->ш, drop ь/ъ
    6. glue letters that were spaced out ("с у к а", "с.у.к.а" -> "сука")
    7. everything that is not a letter or a mask "*" becomes a space
    8. collapse repeated letters ("жаааман" -> "жаман")

Note: step 8 also collapses legitimate double letters ("шешеңнің" -> "шешенин"),
so lexicon patterns are written against the collapsed form.
"""
from __future__ import annotations

import re
import unicodedata

# --- step 2: multi-character Latin sequences (checked before single letters) ---------------
_LATIN_MULTI = [
    ("shch", "ш"), ("sch", "ш"), ("sh", "ш"), ("ch", "ч"), ("zh", "ж"), ("kh", "х"),
    ("ts", "ц"), ("ya", "я"), ("ja", "я"), ("yu", "ю"), ("ju", "ю"), ("yo", "е"),
]
_LATIN_DIACRITICS = str.maketrans({
    "ä": "а", "á": "а", "ö": "о", "ó": "о", "ü": "у", "ú": "у", "ū": "у", "ı": "ы",
    "í": "и", "ğ": "г", "ǵ": "г", "ş": "ш", "ç": "ч", "ñ": "н", "ń": "н",
})
# --- step 3: single Latin letters ----------------------------------------------------------
# Look-alikes (a, c, e, k, o, x, y, m, t) and Kazakh/Russian chat transliteration (q, w, j).
_LATIN_SINGLE = str.maketrans({
    "a": "а", "b": "б", "c": "с", "d": "д", "e": "е", "f": "ф", "g": "г", "h": "х",
    "i": "и", "j": "ж", "k": "к", "l": "л", "m": "м", "n": "н", "o": "о", "p": "п",
    "q": "к", "r": "р", "s": "с", "t": "т", "u": "у", "v": "в", "w": "ш", "x": "х",
    "y": "у", "z": "з",
})
# --- step 4: digits / symbols used as letters (only inside tokens that contain letters) ----
_LEET = str.maketrans({"0": "о", "3": "з", "4": "ч", "6": "б", "@": "а", "$": "с", "#": "*"})  # '#' is a mask too
# --- step 5: Kazakh -> Russian, plus щ/ё/ь/ъ ------------------------------------------------
_KAZ_TO_RUS = str.maketrans({
    "ә": "а", "ғ": "г", "қ": "к", "ң": "н", "ө": "о", "ұ": "у", "ү": "у", "һ": "х",
    "і": "и", "ё": "е", "щ": "ш", "ь": None, "ъ": None,
})

_HAS_LETTER = re.compile(r"[^\W\d_]")
_DIGIT = re.compile(r"\d")
_SPACED = re.compile(r"(?<![а-я*])(?:[а-я*][\s.\-_]+){2,}[а-я*](?![а-я*])")
_NON_LETTER = re.compile(r"[^а-я*]+")
_REPEAT = re.compile(r"([а-я])\1+")
_LONE_STARS = re.compile(r"(?<![а-я])\*+(?![а-я])")


def _latin_multi(s: str) -> str:
    for src, dst in _LATIN_MULTI:
        s = s.replace(src, dst)
    return s


def canonical(text: str) -> str:
    """Normalise text for lexicon matching and for the model (see module docstring)."""
    if not isinstance(text, str):
        return ""
    s = unicodedata.normalize("NFC", text).lower()
    s = s.translate(_LATIN_DIACRITICS)
    s = _latin_multi(s)
    s = s.translate(_LATIN_SINGLE)
    # digits/@/$ become letters only inside words: "к0так" -> "котак", but "100 тг" stays
    # other digits inside a word are dropped ("ба7амын" -> "бамын", not "ба амын")
    s = " ".join(_DIGIT.sub("", t.translate(_LEET)) if _HAS_LETTER.search(t) else t for t in s.split())
    s = s.translate(_KAZ_TO_RUS)
    s = _SPACED.sub(lambda m: re.sub(r"[\s.\-_]+", "", m.group(0)), s)
    s = _NON_LETTER.sub(" ", s)
    s = _LONE_STARS.sub(" ", s)
    s = _REPEAT.sub(r"\1", s)
    return " ".join(s.split())


def dedupe_key(text: str) -> str:
    """Key for exact + near-duplicate detection: canonical form with spaces and masks removed."""
    return re.sub(r"[^а-я]", "", canonical(text))


# --- personal information -------------------------------------------------------------------
_URL = re.compile(r"(?:https?://|www\.)\S+|\b[\w-]+\.(?:kz|ru|com|org|net)\b\S*", re.I)
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
_PHONE = re.compile(r"(?<!\d)(?:\+?[78][\s\-(]*)?\d{3}[\s\-)]*\d{3}[\s\-]*\d{2}[\s\-]*\d{2}(?!\d)")
_USER = re.compile(r"(?<![\w*])@[A-Za-z0-9_.]{2,}")


def scrub_pii(text: str) -> str:
    """Replace links, emails, phone numbers and @usernames with placeholder tags."""
    if not isinstance(text, str):
        return ""
    text = _EMAIL.sub("<EMAIL>", text)   # before URL: "name@mail.ru" must not become "name@<URL>"
    text = _URL.sub("<URL>", text)
    text = _PHONE.sub("<PHONE>", text)
    text = _USER.sub("<USER>", text)
    return text.strip()


def letter_count(text: str) -> int:
    return len(_HAS_LETTER.findall(text or ""))

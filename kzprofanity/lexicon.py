"""Profanity lexicon used ONLY to find candidate comments for labelling.

It is a search tool, not the classifier and not a model feature:
  * STRONG hits (groups "kaz", "rus")      -> auto label 1 (audited by hand on a sample)
  * AMBIGUOUS / BORDERLINE hits only       -> sent to manual review (data/labeling/to_review.csv)
  * INSULT hits (rude but not profane)     -> sampled as hard negatives, label 0

All patterns are written against canonical() text (see kzprofanity/text.py):
lowercase Cyrillic, қ->к, і->и, ң->н, ө->о, ұ/ү->у, ә->а, ғ->г, щ->ш, no ь/ъ,
repeated letters collapsed. Every Cyrillic letter in a pattern automatically
also matches a mask "*", so "к*так" and "б*ять" are caught; a match must keep
at least two real letters, no more masks than letters, and must not start with a mask.

Run `python -m kzprofanity.lexicon` to execute the self-test below.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .text import canonical

B = r"(?<![а-я*])"   # word start
E = r"(?![а-я*])"    # word end

# name -> regex (canonical form).  Keep the comment = original spelling.
KAZ = {
    "котак": B + r"(?:кота[кг]|котабас)",                                   # қотақ, қотақбас, қотағым
    "сиг_verb": B + r"(?:сиге[мйи]|сыге[йи]|сиги[плн]|сигу" + E + r"|сикти|сиктир|сикен|сг(?:ей|еи|ейи)н|сгем)",
    # сігем, сігейін, сігіп, сігіл, сігін, сікті, сіккен, сгейн (vowel dropped)
    "relative_sig": B + r"(?:(?:ана|аке|шеше|апа|ага|карындасы|кызы|катыны|аже|ата|баба|аузы|коти)н[дт][ыи]"
                        r"|(?:шешеси|анаси|апаси)н|сени)\s+(?:[а-я*]+\s+)?сиг",  # анаңды сіг..., шешесін сіг...
    "коти": B + r"коти[нм]",                                    # көтің, көтіңді, көтіме
    "кая": B + r"кая(?:м|йын|йн|ин|ийн)" + E,                   # қаям, қаяйын
    "шешес": B + r"(?:шс|шешес)" + E,                           # щщс, щещес
    "relative_am": B + r"(?:шешен|шешенин|анан|ананын|апан|апанын|катынын|кызынын|карындасынын)\s*ам(?:[ыи][а-я]*)?" + E,  # шешеңнің амы, glued шешенамы
    "амын": B + r"амын(?:ды|ди|а|га|нан)?" + E,                 # амың, амыңды (also Latin "ámın"=amen, see README)
    "амшык": B + r"(?:амшык|амдар)",                            # амшық, амдарың
    "каншык": B + r"каншык",                                    # қаншық (bitch)
    "жалеп": B + r"жалеп",                                      # жәлеп (whore)
}

RUS = {
    "хуй": r"ху[йяеию](?!ав)",                                  # хуй, нахуй, похуй, хуево, хуесос, охуел
    "пизд": r"п[иеи]?зд",                                       # пизда, пиздец, пздц
    "еб": B + r"(?:за|вы|по|у|до|от|раз|рас|на|о|при|вз|из|недо|про|под|пере|ни|съ)?"
              r"(?:еба|ебл|ебн|ебу|ебо[кшщ]|ебуч|ебыв|ебис|ебет|ебеш)",   # ебать, уебок, наебал
    "еб_тв": B + r"[её][бп]\s+тво",                             # ёб твою
    "себал": B + r"себа[лт]",                                   # съебал (ъ is dropped)
    "долбоеб": B + r"д[ао]лб[ао](?:[йи]?[еоюи]?[бп])",          # долбоёб, далбаеб, далбайоп, долбаюб
    "бля": B + r"(?:бл[яэ]|б[иы]ля(?!р))(?!ха|ш)",              # бля, блять, блядь, біля, блэать (not бильярд)
    "заебал": B + r"за[йи]б(?:а[лн]|ис)",                       # зайбал, заибал, заибись
    "иебан": B + r"(?:[йи]е|[уи]и|й)ба[нлт]",                   # иебани, yibani
    "нах": B + r"(?:на|по)ху?" + E,                             # нах, наху, пох
    "сука": B + r"(?:сук(?:а|и|е|у|ой|ам|ами|ах|ин)" + E + r"|сучк|сучар|суч[ае]?ка)",
    "мудак": r"муда[кч]|мудил",
    "пидор": r"п[еи]д[оа]р|пидр|п[еи]дераст|" + B + r"педик(?!юр)",  # пидор, пидараз, педик
    "гандон": r"г[ао]ндон",
    "шлюха": r"шлюх|шлюш",
    "залупа": r"залуп",
    "мандавошка": r"мандавош",
    "шалава": r"шалав",
    "abbrev": B + r"(?:пнх|нх|пздц)" + E,
}

# Hits from these groups never auto-label anything: a human decides (see README rules).
AMBIGUOUS = {
    "ам": B + r"ам[ыи]?" + E,                                   # амы (vulgar) vs "Ам ым", "ам-ам"
    "сиг_other": B + r"сиг(?!нал|арет|ар|ма|е[мйи]|и[плн]|у" + E + r")",  # сіге (typo of сізге), сиген (urinated)
    "сгин": B + r"сгин",                                        # сгін (сігін) vs Russian "сгинь"
    "сикты": B + r"сикты",                                      # сиқты = typo of сияқты
    "кая_other": B + r"кая(?!(?:м|йын|йн|ин|ийн)" + E + r")[а-я]",  # қаямын = typo of қоямын
    "жалап": B + r"жалап",                                      # whore (slang) vs "licking"
}
BORDERLINE = {
    "жопа": B + r"жоп",
    "хер": B + r"(?:на|по)?хер(?:ня|ов|а|у|ом|ово)?" + E,
    "катын": B + r"катын(?!ас|ау|ыс)",                          # қатын (vulgar 'woman'), not қатынас
    "жезокше": r"жезокше",
    "говно": B + r"г[ао]вн",
    "долбан": B + r"д[ао]лбан",                                 # долбанный (mild)
    "сайкал": B + r"са[йи]кал",                                 # сайқал: 'harlot' but often just 'vile, cunning'
    "бок": B + r"бок(?:ка|тан|ты|тай|пен)?" + E,                # боқ (shit) vs Russian бок (side)
}
# Rude, but NOT profanity -> label 0. Used to sample hard negatives.
INSULTS = {
    "insult": B + r"(?:акымак|есек|онбаган|малгун|дебил|идиот|тупо[йг]|чмо|твар[ия]|мраз|дура[кч]|урод|лох)",
}

STRONG_GROUPS = ("kaz", "rus")
REVIEW_GROUPS = ("ambiguous", "borderline")


def _maskable(pattern: str) -> str:
    """Make every Cyrillic letter in a regex also accept '*' (outside and inside [...])."""
    out, in_class, i = [], False, 0
    while i < len(pattern):
        ch = pattern[i]
        if ch == "\\":
            out.append(pattern[i:i + 2])
            i += 2
            continue
        if in_class:
            if ch == "]":
                out.append("*]")
                in_class = False
            else:
                out.append(ch)
        elif ch == "[":
            in_class = True
            out.append(ch)
        elif "а" <= ch <= "я":
            out.append(f"[{ch}*]")
        else:
            out.append(ch)
        i += 1
    return "".join(out)


@dataclass(frozen=True)
class Rule:
    name: str
    group: str
    regex: re.Pattern


def _compile(table: dict, group: str) -> list[Rule]:
    return [Rule(n, group, re.compile(_maskable(p))) for n, p in table.items()]


RULES: list[Rule] = (
    _compile(KAZ, "kaz") + _compile(RUS, "rus") + _compile(AMBIGUOUS, "ambiguous")
    + _compile(BORDERLINE, "borderline") + _compile(INSULTS, "insult")
)


def _valid(span: str) -> bool:
    stars = span.count("*")
    letters = sum(c != "*" and not c.isspace() for c in span)
    # a mask must hide a letter INSIDE a word: "*здравствуйте" is decoration, not "п*зд"
    return letters >= 2 and stars <= letters and not span.startswith("*")


@dataclass(frozen=True)
class Hit:
    rule: str
    group: str
    match: str


def find_hits(text: str, *, already_canonical: bool = False) -> list[Hit]:
    s = text if already_canonical else canonical(text)
    hits = []
    for r in RULES:
        for m in r.regex.finditer(s):
            if _valid(m.group(0)):
                hits.append(Hit(r.name, r.group, m.group(0)))
                break  # one hit per rule is enough
    return hits


def lexicon_label(hits: list[Hit]) -> int | None:
    """1 = strong profanity, None = needs a human, 0 = nothing (insults alone are 0)."""
    groups = {h.group for h in hits}
    if groups & set(STRONG_GROUPS):
        return 1
    if groups & set(REVIEW_GROUPS):
        return None
    return 0


def predict_lexicon(text: str) -> int:
    """Lexicon used as a naive classifier (baseline in evaluate.py): ambiguous -> 0."""
    return int(lexicon_label(find_hits(text)) == 1)


# ------------------------------------------------------------------------------------------
# Self-test: every phrase below must behave as written. Add a line here for every new rule.
SELF_TEST = [
    # (text, expected lexicon_label)
    ("Қотақ бас сатушы", 1), ("qotaq", 1), ("к0так", 1), ("котаааак", 1), ("к*так", 1),
    ("Сігем сені", 1), ("сігейін", 1), ("сігейн", 1), ("сіктім", 1),
    ("анаңды сігейн", 1), ("шешеңді сіг", 1), ("әкеңді сігейін", 1), ("қарындасыңды сігем", 1),
    ("көтіңді сігем", 1), ("қаям жеме", 1), ("қаяйын", 1), ("щщс", 1), ("щещес", 1),
    ("шешеңнің амы", 1), ("щещен амы", 1), ("амыңды", 1),
    ("сука", 1), ("cyka", 1), ("с у к а", 1), ("с.у.к.а", 1), ("сучара", 1), ("сукин сын", 1),
    ("хуй", 1), ("похуй", 1), ("нахуй", 1), ("иди нахуй", 1), ("хуево", 1), ("хуесос", 1),
    ("пизда", 1), ("пиздец", 1), ("пздц", 1), ("пнх", 1),
    ("блять", 1), ("бл*ть", 1), ("б*ять", 1), ("блядь", 1),
    ("ебать", 1), ("уёбок", 1), ("наебал", 1), ("долбоёб", 1), ("ёб твою мать", 1), ("съебал", 1),
    ("мудак", 1), ("пидор", 1), ("пидараз", 1), ("педик", 1), ("гандон", 1), ("шлюха", 1),
    ("залупа", 1), ("мандавошка", 1), ("шалава", 1), ("xyй", 1),
    ("амшық", 1), ("қаншық", 1), ("жәлеп", 1), ("сайқал адамдар", None), ("ба7амын", 0), ("сігіліп", 1), ("сігін", 1),
    ("сыгейн", 1), ("сгейн", 1), ("сіккенім", 1), ("амдарың", 1), ("зайбал", 1), ("заебал", 1),
    ("иебани", 1), ("yibani", 1), ("наху", 1), ("иди нах", 1), ("далбайоп", 1), ("долбаюб", 1),
    ("далбаебтар", 1), ("біля", 1), ("шешенамы", 1), ("қотабас", 1), ("сі#ейын", 1), ("за#бал", 1), ("блэать", 1),
    # needs a human
    ("ам ым", None), ("сізге сіге жаздым", None), ("сиқты", None), ("жалап", None),
    ("қаямын", None), ("жопа", None), ("қатын", None), ("боқ", None), ("говно", None),
    ("нахер", None), ("сгин", None), ("сігіспей қалады", None), ("далбан", None),
    # clean look-alikes and rude-but-not-profane
    ("адам", 0), ("амал", 0), ("аман", 0), ("амбар", 0), ("қайын ата", 0), ("көтеру", 0),
    ("көтерме баға", 0), ("ебі жоқ", 0), ("ебедейсіз", 0), ("шешен сөйлейді", 0),
    ("сукно", 0), ("мандарин", 0), ("педиатр", 0), ("педикюр", 0), ("сигнал жоқ", 0),
    ("сигарет", 0), ("қатынас", 0), ("ақымақ", 0), ("есек", 0), ("оңбаған", 0),
    ("дебил", 0), ("чмо", 0), ("тварь", 0), ("мразь", 0), ("идиот", 0), ("гей", 0),
    ("себебі", 0), ("небо", 0), ("ребята", 0), ("хуавей", 0), ("huawei", 0), ("тауар өте нашар", 0),
    ("100 тг", 0), ("бильярд", 0), ("Шешең аман болсын", 0), ("анаң амандық", 0), ("*здравствуйте *", 0), ("*уй", 0), ("шешенама", 0), ("сығып", 0), ("заиыбы", 0), ("наурыз", 0), ("нахождение", 0), ("ху", 0), ("сигма", 0), ("бляха", 0), ("котик", 0), ("коттедж", 0), ("жаман", 0), ("себет", 0),
    ("ам", None), ("Ам ым", None),
]


def self_test(verbose: bool = True) -> int:
    failures = 0
    for text, expected in SELF_TEST:
        hits = find_hits(text)
        got = lexicon_label(hits)
        if got != expected:
            failures += 1
            print(f"FAIL {text!r}: expected {expected}, got {got} "
                  f"(canonical={canonical(text)!r}, hits={[(h.rule, h.match) for h in hits]})")
    if verbose:
        n = len(SELF_TEST)
        print(f"lexicon self-test: {n - failures}/{n} passed, {len(RULES)} rules")
    return failures


if __name__ == "__main__":
    raise SystemExit(1 if self_test() else 0)

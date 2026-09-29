"""Step 5a - Text preprocessing shared by every model and by the real-SMS test.

    from src.preprocess import preprocess
    p = preprocess("URGENT! Your BOC a/c is blocked. Verify: bit.ly/x1")
    p["text_model"]   -> 'urgent! your boc a/c is blocked. verify: <URL>'
    p["urls"]         -> ['bit.ly/x1']
    p["language"]     -> 'en'

What it does
    1. normalise  - HTML entities, Unicode NFC, invisible characters (keeps the
                    zero-width joiner Sinhala/Tamil spelling needs), operator
                    warning prefixes ("DIALOG WARNING - SPAM :") that would leak the label
    2. extract    - URLs, e-mails, phone numbers, money amounts, emoji
    3. language   - script tag: en, si, ta, si-en, ta-en, si-ta (+ romanised
                    Singlish / Tanglish detected from common words)
    4. mask       - text_model: URLs -> <URL>, phones -> <PHONE>, money -> <MONEY>,
                    long numbers -> <NUM>, emoji -> <EMOJI>, Latin lower-cased.
                    Masking lets the model learn "has a link" instead of memorising
                    one scam domain, and keeps personal numbers out of the model.
"""
from __future__ import annotations

import html
import re
import unicodedata

# ----------------------------------------------------------------- patterns
INVISIBLE = re.compile("[​‎‏‪-‮⁠-⁤﻿­]")
STRAY_JOINER = re.compile("(?<![඀-෿஀-௿])[‌‍]+")
TELCO_PREFIX = re.compile(r"^\s*(?:DIALOG|MOBITEL|HUTCH|AIRTEL|SLT)?\s*WARNING\s*-\s*SPAM\s*:\s*", re.I)

SI = re.compile("[඀-෿]")
TA = re.compile("[஀-௿]")
LATIN = re.compile(r"[A-Za-z]")

EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
URL = re.compile(
    r"(?:https?://[^\s<>\"']+"                                   # explicit scheme
    r"|www\.[^\s<>\"']+"                                         # www.
    r"|\b(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,6}"   # bare host.tld
    r"(?:/[^\s<>\"']*)?)",
    re.I,
)
NOT_URL = re.compile(r"^(?:rs|no|mr|mrs|dr|st|pvt|ltd|co|vs|etc|e\.g|i\.e|a\.m|p\.m)\.", re.I)
PHONE = re.compile(
    r"(?:(?:\+|00)?94[\s-]?|\b0)(?:7\d|[1-9]\d)[\s-]?\d{3}[\s-]?\d{4}\b"  # Sri Lankan mobile/landline
    r"|\b\d{9,12}\b"                                                      # other long numeric senders
)
MONEY = re.compile(
    r"(?:\b(?:rs|lkr|inr|usd)\.?\s?|රු\.?\s?|ரூ\.?\s?|\$|£|€)\s?\d[\d,\s]*(?:\.\d+)?(?:/-)?"
    r"|\b\d[\d,]*(?:\.\d+)?\s?(?:rupees|rupiyal|rs\b|lkr\b|/-)",
    re.I,
)
LONG_NUM = re.compile(r"\b\d{4,}\b")
EMOJI = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F000-\U0001F2FF\U0001F1E6-\U0001F1FF⭐⭕‼⁉™ℹ]"
    "[️‍\U0001F3FB-\U0001F3FF]*"
)
SPECIAL = re.compile(r"<(url|email|phone|money|num|emoji|acc|nic|card|name|otp|link)>", re.I)

SINGLISH = {"eka", "ekak", "ekata", "karanna", "oyage", "oyata", "wenawa", "ganna", "labaganna", "nam",
            "mata", "mage", "api", "apen", "naya", "salli", "mudal", "gewanna", "witharai", "epa", "puluwan",
            "kiyala", "danna", "ewanna", "nattam", "ada", "dan", "hari", "machan", "awa", "una", "wela", "karala",
            "thiyenawa", "enawa", "yanawa", "dakwa", "dakva", "gewala", "saha", "pahasu", "ikman", "poli"}
TANGLISH = {"unga", "ungalukku", "pannunga", "irukku", "vandhu", "vandhirukku", "illa", "illana", "inniku",
            "naan", "podunga", "kattunga", "venum", "aagum", "sollu", "vaanga", "panna", "enna", "eppo",
            "nee", "neenga", "enakku", "ungal", "mattum", "ippove", "anuppunga", "aayiduchu", "kitta", "da"}


# ----------------------------------------------------------------- steps
def normalise(text: str) -> tuple[str, int]:
    """Return (clean text, telco_flag). telco_flag=1 if an operator spam warning was stripped."""
    t = html.unescape(str(text if text is not None else ""))
    t = t.replace("Â£", "£").replace("Ã¼", "ü")          # common mojibake in UCI
    t = unicodedata.normalize("NFC", t)
    t = STRAY_JOINER.sub("", INVISIBLE.sub("", t))
    flag = int(bool(TELCO_PREFIX.match(t)))
    t = TELCO_PREFIX.sub("", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t, flag


def extract_urls(text: str) -> list[str]:
    out = []
    for m in URL.finditer(EMAIL.sub(" ", text)):
        u = m.group(0).rstrip(".,;:!?)]}'\"*")
        if NOT_URL.match(u) or re.fullmatch(r"[\d.]+", u):
            continue
        out.append(u)
    return out


def host_of(url: str) -> str:
    h = re.sub(r"^https?://", "", url.lower())
    h = h.split("/")[0].split("?")[0].split(":")[0]
    return h[4:] if h.startswith("www.") else h


def detect_language(text: str) -> str:
    si, ta, la = bool(SI.search(text)), bool(TA.search(text)), bool(LATIN.search(text))
    if si and ta:
        return "si-ta"
    if si:
        return "si-en" if la and len(LATIN.findall(text)) > 0.5 * len(SI.findall(text)) else "si"
    if ta:
        return "ta-en" if la and len(LATIN.findall(text)) > 0.5 * len(TA.findall(text)) else "ta"
    words = set(re.findall(r"[a-z]+", text.lower()))
    s, t = len(words & SINGLISH), len(words & TANGLISH)
    if max(s, t) >= 2:
        return "si-en" if s >= t else "ta-en"
    return "en"


def to_model_text(text: str) -> str:
    t = EMAIL.sub(" <EMAIL> ", text)
    for u in sorted(extract_urls(t), key=len, reverse=True):
        t = t.replace(u, " <URL> ")
    t = MONEY.sub(" <MONEY> ", t)
    t = PHONE.sub(" <PHONE> ", t)
    t = LONG_NUM.sub(" <NUM> ", t)
    t = EMOJI.sub(" <EMOJI> ", t)
    t = t.lower()
    t = SPECIAL.sub(lambda m: f"<{m.group(1).upper()}>", t)
    return re.sub(r"\s+", " ", t).strip()


def preprocess(text: str) -> dict:
    clean, flag = normalise(text)
    urls = extract_urls(clean)
    return {
        "text_clean": clean,
        "text_model": to_model_text(clean),
        "language": detect_language(clean),
        "urls": urls,
        "hosts": [host_of(u) for u in urls],
        "emails": EMAIL.findall(clean),
        "phones": [p.strip() for p in PHONE.findall(EMAIL.sub(" ", clean))],
        "money": [m.strip() for m in MONEY.findall(clean)],
        "n_emoji": len(EMOJI.findall(clean)),
        "telco_flag": flag,
    }

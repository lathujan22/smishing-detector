"""Step 3 - Import Google Form responses into the real-SMS test set.

    python -m src.import_form                     # form_responses.csv -> real_test.csv
    python -m src.import_form --file other.csv

Inputs  (data/real_sms/)
    form_responses.csv  Google Forms "Download responses (.csv)"
    relabel.csv         researcher annotation, keyed by form Timestamp:
                        timestamp,label,language,action(keep|drop|split),note
Output
    real_test.csv       rebuilt every run (FORM-xxxx rows) + any manual rows kept

Cleaning
    - removes invisible characters and the operator warning prefix
      "DIALOG WARNING - SPAM :" (kept as telco_flag=1 so the model can't cheat on it)
    - applies researcher labels; keeps the respondent's label in label_user
    - split: several SMS pasted in one answer (blank-line separated) -> separate rows
    - masks phone / NIC / account / card numbers and e-mails
    - removes duplicates (counted in times_reported) and anything already in train.csv
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REAL_DIR = ROOT / "data" / "real_sms"
OUT = REAL_DIR / "real_test.csv"
RELABEL = REAL_DIR / "relabel.csv"
TRAIN = ROOT / "data" / "processed" / "train.csv"
COLS = ["id", "text", "label", "language", "label_user", "language_user", "telco_flag", "times_reported",
        "sender_type", "source", "date_received", "notes", "annotator_note", "screenshot"]

TYPE_MAP = {"scam": "smishing", "promotion": "spam", "normal": "ham"}
LANG_MAP = {"english": "en", "sinhala": "si", "tamil": "ta", "singlish": "si-en", "tanglish": "ta-en", "mixed": "mixed"}

INVISIBLE = re.compile("[\u200b\u200e\u200f\u202a-\u202e\u2060-\u2064\ufeff]")
# ZWJ/ZWNJ are needed inside Sinhala/Tamil words (e.g. al-lakuna + ZWJ) - drop them only elsewhere
STRAY_JOINER = re.compile("(?<![\u0D80-\u0DFF\u0B80-\u0BFF])[\u200c\u200d]+")
TELCO_PREFIX = re.compile(r"^\s*(?:DIALOG|MOBITEL|HUTCH|AIRTEL)?\s*WARNING\s*-\s*SPAM\s*:\s*", re.I)
MASKS = [
    (re.compile(r"\S+@\S+\.\w+"), "<EMAIL>"),
    (re.compile(r"(?:\+?94|\b0)\s?7\d[\s-]?\d{3}[\s-]?\d{4}\b"), "<PHONE>"),        # SL mobile
    (re.compile(r"(?:\+?94|\b0)\s?\d{2}[\s-]?\d{3}[\s-]?\d{4}\b"), "<PHONE>"),      # SL landline
    (re.compile(r"\b\d{9}[VvXx]\b|\b(?:19|20)\d{10}\b"), "<NIC>"),                  # old / new NIC
    (re.compile(r"\b(?:\d{4}[\s-]?){3}\d{4}\b"), "<CARD>"),                         # 16-digit card
    (re.compile(r"\b\d{10,15}\b"), "<ACC>"),                                         # long account nos
]


def col(df: pd.DataFrame, name: str) -> pd.Series:
    """Find a form column whose title contains `name` (Google may add 'Title:' / descriptions)."""
    for c in df.columns:
        if name.lower() in c.strip().lower():
            return df[c]
    return pd.Series([""] * len(df), index=df.index)


def map_start(value, mapping) -> str | None:
    v = str(value).strip().lower()
    return next((out for k, out in mapping.items() if v.startswith(k)), None)


def clean(text: str) -> tuple[str, int]:
    t = STRAY_JOINER.sub("", INVISIBLE.sub("", str(text)))
    flag = int(bool(TELCO_PREFIX.match(t)))
    t = TELCO_PREFIX.sub("", t)
    t = re.sub(r"\s+", " ", t).strip()
    for rx, repl in MASKS:
        t = rx.sub(repl, t)
    return t, flag


def key(t: str) -> str:
    return re.sub(r"\s+", " ", str(t)).strip().lower()


def cohen_kappa(a: pd.Series, b: pd.Series) -> float:
    labels = sorted(set(a) | set(b))
    n = len(a)
    po = (a == b).mean()
    pe = sum((a == l).sum() / n * (b == l).sum() / n for l in labels)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def main(path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"{path} not found - download the responses CSV from Google Forms first.")
    raw = pd.read_csv(path, encoding="utf-8")
    base = pd.DataFrame({
        "timestamp": col(raw, "Timestamp").astype(str),
        "consent": col(raw, "Consent").astype(str).str.len() > 3,
        "raw_text": col(raw, "Message text").fillna("").astype(str),
        "label_user": col(raw, "Message type").map(lambda v: map_start(v, TYPE_MAP)),
        "language_user": col(raw, "Language").map(lambda v: map_start(v, LANG_MAP)),
        "sender_type": col(raw, "Sender").fillna("").astype(str),
        "date_received": col(raw, "Month received").fillna("").astype(str),
        "notes": col(raw, "How did you know").fillna("").astype(str),
        "screenshot": col(raw, "screen shot").fillna("").astype(str),
    })
    rel = pd.read_csv(RELABEL, encoding="utf-8", dtype=str).fillna("") if RELABEL.exists() else pd.DataFrame(
        columns=["timestamp", "label", "language", "action", "note"])
    base = base.merge(rel.rename(columns={"label": "label_r", "language": "language_r", "note": "annotator_note"}),
                      on="timestamp", how="left").fillna({"action": "", "label_r": "", "language_r": "", "annotator_note": ""})

    rep = {"responses": len(base)}
    base = base[base["consent"]];                                   rep["no consent"] = rep["responses"] - len(base)
    n = len(base); base = base[base["action"] != "drop"];            rep["dropped by annotator"] = n - len(base)
    pending = base[(base["action"] == "")]
    if len(pending):
        print(f"! {len(pending)} responses have no researcher label yet (add them to relabel.csv):")
        for _, r in pending.iterrows():
            print(f"   {r['timestamp']} | user={r['label_user']} | {r['raw_text'][:70]!r}")

    rows = []
    for _, r in base.iterrows():
        parts = re.split(r"\n\s*\n", r["raw_text"].strip()) if r["action"] == "split" else [r["raw_text"]]
        for p in parts:
            text, flag = clean(p)
            rows.append({**r.to_dict(), "text": text, "telco_flag": flag,
                         "label": r["label_r"] or r["label_user"], "language": r["language_r"] or r["language_user"] or "mixed"})
    df = pd.DataFrame(rows)
    n = len(df); df = df[df["text"].str.len() >= 10];               rep["empty/too short"] = n - len(df)
    n = len(df); df = df[df["label"].notna() & (df["label"] != "")]; rep["unlabelled ('Not sure', no relabel)"] = n - len(df)

    # duplicates: keep first report, count how many people reported it
    df["k"] = df["text"].map(key)
    df["times_reported"] = df.groupby("k")["k"].transform("size")
    df["telco_flag"] = df.groupby("k")["telco_flag"].transform("max")
    n = len(df); df = df.drop_duplicates("k", keep="first");        rep["duplicates merged"] = n - len(df)
    if TRAIN.exists():
        seen = set(pd.read_csv(TRAIN, encoding="utf-8")["text"].map(key))
        n = len(df); df = df[~df["k"].isin(seen)];                  rep["already in train"] = n - len(df)

    df = df.reset_index(drop=True)
    df["id"] = [f"FORM-{i:04d}" for i in range(1, len(df) + 1)]
    df["source"] = "google form"
    manual = pd.read_csv(OUT, encoding="utf-8") if OUT.exists() else pd.DataFrame(columns=COLS)
    manual = manual[~manual["id"].astype(str).str.startswith(("FORM-", "EX-"))]
    out = pd.concat([manual, df[COLS]], ignore_index=True)
    out.to_csv(OUT, index=False, encoding="utf-8")

    print("Import report:", rep)
    print(f"-> real_test.csv: {len(out)} unique messages ({len(df)} from the form)\n")
    print(pd.crosstab(out["language"], out["label"], margins=True))
    both = df[df["label_user"].notna()]
    if len(both):
        agree = (both["label_user"] == both["label"]).mean()
        print(f"\nRespondent vs researcher label agreement: {agree:.0%}  (Cohen's kappa = {cohen_kappa(both['label_user'], both['label']):.2f}, n={len(both)})")
    print(f"Operator already flagged as spam: {int(df['telco_flag'].sum())} messages")
    print("\nNext: python -m src.data_prep --check-real")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", type=Path, default=REAL_DIR / "form_responses.csv")
    main(ap.parse_args().file)

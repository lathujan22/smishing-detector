"""Step 3b - Bulk-add real SMS from your own / family phones to the real-SMS test set.

Two ways to collect (use either):

A) Text file  data/real_sms/my_messages.txt  - paste messages, blank line between them.
   Put a header line before each group:  ## <label> <language>
        ## ham si
        ඔබගේ ගිණුමට රු. 5,000.00ක් බැර විය. - BOC

        ඔබගේ OTP අංකය 123456 ...

        ## smishing en
        SLKPOST: Your shipment is pending ...
   label: ham | spam | smishing      language: en | si | ta | si-en | ta-en | mixed

B) Whole inbox  (Android app "SMS Backup & Restore" -> Back up -> Messages -> .xml file)
        python -m src.import_bulk inbox.xml       -> data/real_sms/to_label.csv
   Open to_label.csv in Excel, fill the `label` column (ham/spam/smishing), DELETE rows you
   don't want to share (private chats), save, then run --apply.

Commands
    python -m src.import_bulk                       # reads my_messages.txt, appends labelled rows
    python -m src.import_bulk inbox.xml             # makes to_label.csv (label column empty)
    python -m src.import_bulk --apply               # appends labelled rows of to_label.csv

Everything is cleaned with src.preprocess (operator warning stripped, invisible chars),
numbers/e-mails are masked, and duplicates of real_test.csv or train.csv are skipped.
Rows get ids OWN-0001 ... and survive re-running src.import_form.
"""
from __future__ import annotations

import argparse
import re
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.import_form import COLS, MASKS, key
from src.preprocess import detect_language, normalise

ROOT = Path(__file__).resolve().parents[1]
REAL = ROOT / "data" / "real_sms"
OUT = REAL / "real_test.csv"
TXT = REAL / "my_messages.txt"
TO_LABEL = REAL / "to_label.csv"
TRAIN = ROOT / "data" / "processed" / "train.csv"
LABELS = {"ham", "spam", "smishing"}
LANGS = {"en", "si", "ta", "si-en", "ta-en", "si-ta", "mixed"}


def clean(text: str) -> tuple[str, int]:
    t, flag = normalise(text)
    for rx, repl in MASKS:
        t = rx.sub(repl, t)
    return t, flag


def seen_keys() -> set:
    s = set()
    for p, col in ((OUT, "text"), (TRAIN, "text")):
        if p.exists():
            s |= set(pd.read_csv(p, encoding="utf-8")[col].map(key))
    return s


def parse_txt(path: Path) -> pd.DataFrame:
    rows, label, lang = [], None, None
    blocks = re.split(r"\n\s*\n", path.read_text(encoding="utf-8-sig").replace("\r\n", "\n"))
    for b in blocks:
        lines = b.strip().split("\n")
        while lines and lines[0].startswith("##"):
            parts = lines.pop(0).lstrip("#").split()
            label = parts[0].lower() if parts else None
            lang = parts[1].lower() if len(parts) > 1 else None
        body = "\n".join(lines).strip()
        if body:
            rows.append({"raw": body, "label": label, "language": lang})
    return pd.DataFrame(rows)


def parse_xml(path: Path) -> pd.DataFrame:
    rows = []
    for el in ET.parse(path).getroot().iter("sms"):
        if el.get("type") not in (None, "1"):          # 1 = received (skip sent messages)
            continue
        ms = int(el.get("date", "0") or 0)
        rows.append({"raw": el.get("body", ""), "sender": el.get("address", ""),
                     "date_received": datetime.fromtimestamp(ms / 1000).strftime("%Y-%m") if ms else ""})
    return pd.DataFrame(rows)


def finish(df: pd.DataFrame, source: str) -> None:
    """Append labelled rows (label + language set) to real_test.csv."""
    df = df[df["label"].isin(LABELS)].copy()
    cleaned = df["raw"].map(clean)
    df["text"], df["telco_flag"] = cleaned.str[0], cleaned.str[1]
    df["language"] = [l if l in LANGS else detect_language(t) for l, t in zip(df["language"], df["text"])]
    df = df[df["text"].str.len() >= 10]
    seen = seen_keys()
    before = len(df)
    df = df[~df["text"].map(key).duplicated() & ~df["text"].map(key).isin(seen)]
    skipped = before - len(df)

    existing = pd.read_csv(OUT, encoding="utf-8") if OUT.exists() else pd.DataFrame(columns=COLS)
    n0 = existing["id"].astype(str).str.extract(r"OWN-(\d+)")[0].dropna().astype(int)
    start = int(n0.max()) + 1 if len(n0) else 1
    df["id"] = [f"OWN-{i:04d}" for i in range(start, start + len(df))]
    df["source"], df["times_reported"] = source, 1
    for c in COLS:
        if c not in df:
            df[c] = ""
    out = pd.concat([existing, df[COLS]], ignore_index=True)
    out.to_csv(OUT, index=False, encoding="utf-8")
    print(f"added {len(df)} messages ({skipped} duplicates skipped) -> real_test.csv now {len(out)} rows\n")
    print(pd.crosstab(out["language"], out["label"], margins=True))


def make_to_label(df: pd.DataFrame) -> None:
    cleaned = df["raw"].map(clean)
    df = df.assign(text=cleaned.str[0])
    df = df[df["text"].str.len() >= 10]
    seen = seen_keys()
    df = df[~df["text"].map(key).duplicated() & ~df["text"].map(key).isin(seen)].copy()
    df["language"] = df["text"].map(detect_language)
    df["suggested"] = ""
    model = ROOT / "models" / "baselines" / "SVM.joblib"
    if model.exists():                                  # optional hint from the Step 6 SVM
        import joblib
        from src.preprocess import to_model_text
        pred = joblib.load(model).predict(df["text"].map(to_model_text))
        df["suggested"] = ["spam/smishing?" if p else "ham?" for p in pred]
    df["label"] = ""
    cols = ["label", "language", "suggested", "sender", "date_received", "text"]
    df[[c for c in cols if c in df]].to_csv(TO_LABEL, index=False, encoding="utf-8-sig")
    print(f"{len(df)} messages -> {TO_LABEL}\nFill 'label' (ham / spam / smishing), delete private rows, save, "
          f"then run:  python -m src.import_bulk --apply")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", nargs="?", type=Path, help="SMS Backup & Restore .xml (or a .txt in the ## format)")
    ap.add_argument("--apply", action="store_true", help="import the labelled rows of to_label.csv")
    a = ap.parse_args()
    if a.apply:
        t = pd.read_csv(TO_LABEL, encoding="utf-8-sig", dtype=str).fillna("")
        t["label"] = t["label"].str.strip().str.lower()
        finish(t.rename(columns={"text": "raw"}).assign(sender_type=t.get("sender", "")), "own phone (inbox export)")
    elif a.file and a.file.suffix.lower() == ".xml":
        make_to_label(parse_xml(a.file))
    else:
        path = a.file or TXT
        if not path.exists():
            raise SystemExit(f"{path} not found - create it (see the format at the top of this file).")
        finish(parse_txt(path), "own phone / family")

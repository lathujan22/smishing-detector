"""Step 2 - Dataset preparation.

Loads the UCI SMS Spam Collection and the Mendeley SMS Phishing Dataset
(plus an optional multilingual training file), merges them, removes
duplicates, maps labels and writes a fixed stratified 80/20 split.

Machine-translated rows (data/raw/multilingual_mt.csv, made by
notebooks/01_translate_nllb.ipynb from train.csv) are added to TRAIN ONLY,
after the split, so no translation of a test message can leak into training.

Run from the project root:
    python -m src.data_prep                 # build train/test
    python -m src.data_prep --check-real    # validate data/real_sms/real_test.csv

Outputs (data/processed/):
    train.csv, test.csv   columns: text, label, label3, language, source
    data_report.txt       counts before/after cleaning, per class, per language

Label scheme
    label3 : ham | spam | smishing      (kept for analysis)
    label  : 0 = ham, 1 = spam or smishing  (the model target)
"""
from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
REAL = ROOT / "data" / "real_sms" / "real_test.csv"

SEED = 42
TEST_SIZE = 0.20
LABEL3_ORDER = ["ham", "spam", "smishing"]
VALID_LANGS = {"en", "si", "ta", "si-en", "ta-en", "si-ta", "mixed"}

# Unicode blocks
_SI = re.compile(r"[඀-෿]")
_TA = re.compile(r"[஀-௿]")
_LATIN = re.compile(r"[A-Za-z]")


# ----------------------------------------------------------------- helpers
def detect_language(text: str) -> str:
    """Script-based language tag: en, si, ta, si-en, ta-en, si-ta.

    Romanised Sinhala/Tamil ("Singlish"/"Tanglish") is written in Latin
    script and will be tagged 'en' here - tag those by hand in your CSVs.
    """
    t = str(text)
    si, ta, la = bool(_SI.search(t)), bool(_TA.search(t)), bool(_LATIN.search(t))
    if si and ta:
        return "si-ta"
    if si:
        return "si-en" if la else "si"
    if ta:
        return "ta-en" if la else "ta"
    return "en"


def normalise_label(value) -> str | None:
    v = str(value).strip().lower()
    if v in {"ham", "legit", "legitimate", "0"}:
        return "ham"
    if v in {"spam"}:
        return "spam"
    if v in {"smishing", "phishing", "smish", "scam", "1"}:
        return "smishing"
    return None


def dedupe_key(text: str) -> str:
    """Key used to spot duplicates: lower-case, collapsed whitespace."""
    return re.sub(r"\s+", " ", str(text)).strip().lower()


def _find(pattern_list: list[str]) -> Path | None:
    for pat in pattern_list:
        hits = sorted(RAW.glob(pat))
        if hits:
            return hits[0]
    return None


# ----------------------------------------------------------------- loaders
def load_uci() -> pd.DataFrame:
    path = _find(["SMSSpamCollection", "SMSSpamCollection.txt", "**/SMSSpamCollection*"])
    if path is None:
        print("! UCI file not found in data/raw (expected 'SMSSpamCollection')")
        return pd.DataFrame(columns=["text", "label3", "source"])
    df = pd.read_csv(path, sep="\t", header=None, names=["label3", "text"],
                     quoting=3, encoding="utf-8", on_bad_lines="skip")
    df["label3"] = df["label3"].map(normalise_label)
    df["source"] = "uci"
    print(f"  UCI       {path.name:<28} {len(df):>6} rows")
    return df[["text", "label3", "source"]]


def load_mendeley() -> pd.DataFrame:
    path = _find(["Dataset_5971.csv", "*5971*.csv", "**/Dataset_5971.csv", "**/*5971*.csv"])
    if path is None:
        print("! Mendeley file not found in data/raw (expected 'Dataset_5971.csv')")
        return pd.DataFrame(columns=["text", "label3", "source"])
    df = None
    for enc in ("utf-8", "latin-1"):
        try:
            df = pd.read_csv(path, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    cols = {c.lower().strip(): c for c in df.columns}
    text_col = cols.get("text") or cols.get("message") or cols.get("sms")
    label_col = cols.get("label") or cols.get("class") or cols.get("category")
    if not text_col or not label_col:
        sys.exit(f"Mendeley columns not recognised: {list(df.columns)}")
    out = pd.DataFrame({"text": df[text_col], "label3": df[label_col].map(normalise_label)})
    out["source"] = "mendeley"
    print(f"  Mendeley  {path.name:<28} {len(out):>6} rows")
    return out


def load_multilingual() -> pd.DataFrame:
    path = RAW / "multilingual_train.csv"
    if not path.exists():
        print("  (no data/raw/multilingual_train.csv yet - English-only for now)")
        return pd.DataFrame(columns=["text", "label3", "source", "language"])
    df = pd.read_csv(path, encoding="utf-8")
    out = pd.DataFrame({"text": df["text"], "label3": df["label"].map(normalise_label)})
    if "language" in df.columns:
        out["language"] = df["language"]
    out["group"] = df["template_id"] if "template_id" in df.columns else [f"ml-{i}" for i in range(len(df))]
    out["source"] = "multilingual"
    print(f"  Multi     {path.name:<28} {len(out):>6} rows")
    return out


def load_mt(test: pd.DataFrame) -> pd.DataFrame:
    """Machine-translated rows -> train only. Drops any whose English source is in test."""
    path = RAW / "multilingual_mt.csv"
    cols = ["text", "label", "label3", "language", "source"]
    if not path.exists():
        return pd.DataFrame(columns=cols)
    df = pd.read_csv(path, encoding="utf-8")
    df["label3"] = df["label"].map(normalise_label)
    df = df[df["label3"].notna() & df["text"].notna()]
    if "src_text" in df.columns:  # safety: never keep a translation of a test message
        df = df[~df["src_text"].map(dedupe_key).isin(set(test["text"].map(dedupe_key)))]
    df["text"] = df["text"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    df = df[~df["text"].map(dedupe_key).duplicated()]
    df["label"] = (df["label3"] != "ham").astype(int)
    df["source"] = "mt_nllb"
    print(f"  MT        {path.name:<28} {len(df):>6} rows (train only)")
    return df[cols]


# ----------------------------------------------------------------- build
def build(raw_dir: Path = RAW, out_dir: Path = PROCESSED) -> tuple[pd.DataFrame, pd.DataFrame]:
    global RAW
    RAW = raw_dir
    log: list[str] = []
    print("Loading raw data ...")
    # Priority order matters for duplicates: the more specific label wins
    # (Mendeley has 'smishing' where UCI only says 'spam').
    frames = [load_mendeley(), load_multilingual(), load_uci()]
    df = pd.concat([f for f in frames if len(f)], ignore_index=True)
    if df.empty:
        sys.exit("No data found. Put the datasets in data/raw first (see docs/CHECKLIST.md Step 2).")
    log.append(f"Raw rows loaded: {len(df)}")
    log.append("  by source: " + df["source"].value_counts().to_dict().__repr__())

    df["text"] = df["text"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    before = len(df)
    df = df[(df["text"].str.len() > 0) & (df["text"] != "nan") & df["label3"].notna()]
    log.append(f"Dropped empty / unknown-label rows: {before - len(df)}")

    df["key"] = df["text"].map(dedupe_key)
    # conflicting labels for the same text -> report them
    conflicts = df.groupby("key")["label3"].nunique()
    n_conf = int((conflicts > 1).sum())
    before = len(df)
    df = df.drop_duplicates("key", keep="first")
    log.append(f"Duplicates removed: {before - len(df)} (texts with conflicting labels: {n_conf}, kept first by source priority)")

    if "language" not in df.columns:
        df["language"] = None
    df["language"] = df["language"].where(df["language"].isin(VALID_LANGS), df["text"].map(detect_language))
    df["label"] = (df["label3"] != "ham").astype(int)
    if "group" not in df.columns:
        df["group"] = None
    cols = ["text", "label", "label3", "language", "source"]

    # (a) English datasets: stratify on label3 + language
    base = df[df["source"] != "multilingual"].reset_index(drop=True)
    strat = base["label3"] + "|" + base["language"]
    counts = strat.value_counts()
    strat = strat.where(strat.map(counts) >= 2, base["label3"])
    tr_a, te_a = train_test_split(base, test_size=TEST_SIZE, random_state=SEED, stratify=strat)

    # (b) multilingual templates: split by TEMPLATE so all variants of one
    #     pattern stay on the same side (no near-copy leakage into test)
    ml = df[df["source"] == "multilingual"].reset_index(drop=True)
    test_groups: set = set()
    for _, part in ml.groupby(["language", "label3"]):
        groups = sorted(part["group"].astype(str).unique())
        random.Random(SEED).shuffle(groups)
        k = max(1, round(len(groups) * TEST_SIZE)) if len(groups) > 1 else 0
        test_groups.update(groups[:k])
    in_test = ml["group"].astype(str).isin(test_groups)
    tr_b, te_b = ml[~in_test], ml[in_test]

    train = pd.concat([tr_a, tr_b])[cols].sample(frac=1, random_state=SEED).reset_index(drop=True)
    test = pd.concat([te_a, te_b])[cols].sample(frac=1, random_state=SEED).reset_index(drop=True)
    df = df[cols]
    log.append(f"Multilingual split by template: {ml['group'].nunique()} templates -> {len(test_groups)} held out for test")

    mt = load_mt(test)
    if len(mt):
        train = pd.concat([train, mt], ignore_index=True).sample(frac=1, random_state=SEED).reset_index(drop=True)
        log.append(f"Machine-translated rows added to TRAIN only: {len(mt)}  " + mt["language"].value_counts().to_dict().__repr__())

    out_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(out_dir / "train.csv", index=False, encoding="utf-8")
    test.to_csv(out_dir / "test.csv", index=False, encoding="utf-8")

    log.append(f"Final dataset: {len(df)}  ->  train {len(train)} / test {len(test)}")
    for name, part in (("ALL", df), ("TRAIN", train), ("TEST", test)):
        c3 = part["label3"].value_counts().reindex(LABEL3_ORDER, fill_value=0).to_dict()
        pos = part["label"].mean() * 100
        log.append(f"  {name:<5} label3={c3}  positive(spam+smishing)={pos:.1f}%")
    log.append("  languages (all): " + df["language"].value_counts().to_dict().__repr__())
    report = "\n".join(log)
    (out_dir / "data_report.txt").write_text(report + "\n", encoding="utf-8")
    print("\n" + report + f"\n\nSaved to {out_dir}")
    return train, test


# ----------------------------------------------------------------- real SMS check
def check_real(path: Path = REAL) -> None:
    if not path.exists():
        sys.exit(f"{path} not found. Copy data/real_sms/real_test_TEMPLATE.csv to real_test.csv and fill it.")
    df = pd.read_csv(path, encoding="utf-8")
    need = {"id", "text", "label", "language"}
    missing = need - set(df.columns)
    if missing:
        sys.exit(f"Missing columns: {missing}")
    problems = []
    if df["id"].astype(str).str.startswith("EX-").any():
        problems.append("example rows (id EX-...) still present - delete them")
    bad_lab = df[df["label"].map(normalise_label).isna()]
    if len(bad_lab):
        problems.append(f"{len(bad_lab)} rows with unknown label (use ham/spam/smishing): ids {bad_lab['id'].tolist()[:10]}")
    bad_lang = df[~df["language"].isin(VALID_LANGS)]
    if len(bad_lang):
        problems.append(f"{len(bad_lang)} rows with unknown language (use {sorted(VALID_LANGS)})")
    phone = df["text"].astype(str).str.contains(r"(?:\+94|\b0)7\d[\s-]?\d{3}[\s-]?\d{4}\b", regex=True)
    if phone.any():
        problems.append(f"{int(phone.sum())} rows still contain a Sri Lankan mobile number - replace with <PHONE>: ids {df.loc[phone, 'id'].tolist()[:10]}")
    if train_overlap := _overlap_with_train(df):
        problems.append(f"{train_overlap} real SMS also appear in the training data - remove them")
    print(f"real_test.csv: {len(df)} rows")
    print(pd.crosstab(df["language"], df["label"].map(normalise_label), margins=True))
    print("\nOK - ready for testing." if not problems else "\nFix these:\n - " + "\n - ".join(problems))


def _overlap_with_train(real: pd.DataFrame) -> int:
    tr = PROCESSED / "train.csv"
    if not tr.exists():
        return 0
    keys = set(pd.read_csv(tr, encoding="utf-8")["text"].map(dedupe_key))
    return int(real["text"].map(dedupe_key).isin(keys).sum())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check-real", action="store_true", help="validate data/real_sms/real_test.csv")
    args = ap.parse_args()
    check_real() if args.check_real else build()

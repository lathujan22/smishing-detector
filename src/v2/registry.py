"""V2 data registry, validation and ingestion.

    python -m src.v2.registry build                      # registry of all existing data + near-dup families + audit
    python -m src.v2.registry ingest FILE --role dev      # new CSV / SMS Backup & Restore XML -> development pool
    python -m src.v2.registry ingest FILE --role locked_test   # -> NEW independent real-world test set (locked)

Every message gets: msg_id, split/role, source, language, label3 + class name, label_provenance, is_synthetic,
collection_method, licence, PII-masked text, model text, separate URL / phone risk features, family_id
(exact + near-duplicate family, cosine >= 0.8 on character n-grams, merged with template / translation groups),
quarantine flag + reason.

Rules: labels are only taken from the input (never inferred); promotional messages are never relabelled as scam;
unlabelled / "not sure" / conflicting messages go to quarantine for manual review; the 163 original real SMS stay
an audited benchmark (role = real_benchmark) and are never used for training, thresholds or tuning.
Outputs (private, data/v2/): registry.csv, quarantine.csv, incoming_<role>.csv
Outputs (aggregate, results/v2/data/): data_audit.md, distribution tables, contamination report.
"""
from __future__ import annotations

import argparse
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from src.features import extract_features
from src.import_form import MASKS
from src.preprocess import detect_language, normalise, preprocess
from src.v2.common import DATA, LANG_FAMILY, PROVENANCE, RES, ROOT, SEED

NEAR_DUP = 0.80
LABEL_IN = {"ham": "ham", "normal": "ham", "legitimate": "ham", "0": "ham",
            "spam": "spam", "promotional": "spam", "promotion": "spam", "promo": "spam", "advertisement": "spam", "1": "spam",
            "smishing": "smishing", "scam": "smishing", "phishing": "smishing", "fraud": "smishing", "2": "smishing"}
RISK_COLS = ["has_url", "url_shortener", "risky_domain", "has_phone", "has_money", "urgency", "impersonation",
             "action_request", "link_risk", "n_urgency_hits", "telco_flag"]


def pii_mask(text: str) -> str:
    t = text
    for rx, repl in MASKS:                       # e-mail, SL phones, NIC, card, long account numbers
        t = rx.sub(repl, t)
    return t


def families(texts: pd.Series, extra_groups: pd.Series | None = None) -> np.ndarray:
    """Connected components of: exact duplicates, char-n-gram cosine >= NEAR_DUP, and shared extra group ids."""
    X = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True).fit_transform(texts.astype(str))
    nn = NearestNeighbors(radius=1 - NEAR_DUP, metric="cosine", algorithm="brute").fit(X)
    G = nn.radius_neighbors_graph(X, mode="connectivity").tocoo()
    rows, cols = [G.row], [G.col]
    if extra_groups is not None:
        codes = pd.factorize(extra_groups.astype(str))[0]
        first = pd.Series(np.arange(len(codes))).groupby(codes).transform("first").to_numpy()
        rows.append(np.arange(len(codes))); cols.append(first)
    r, c = np.concatenate(rows), np.concatenate(cols)
    A = coo_matrix((np.ones(len(r)), (r, c)), shape=(len(texts), len(texts)))
    return connected_components(A, directed=False)[1]


def risk_features(text: str) -> dict:
    f = extract_features(text, preprocess(text))
    return {k: f[k] for k in RISK_COLS} | {"language_detected": f["language_detected"], "text_model": f["text_model"]}


# ------------------------------------------------------------------ build registry from existing data
def build() -> None:
    proc = ROOT / "data" / "processed"
    tr = pd.read_csv(proc / "train_feat.csv").assign(split_v1="train")
    te = pd.read_csv(proc / "test_feat.csv").assign(split_v1="test")
    rl = pd.read_csv(proc / "real_test_feat.csv").assign(split_v1="real", source="real_form", group=lambda d: "real:" + d["id"].astype(str))
    rl["label3"] = rl["label"]
    cols = ["text", "label3", "language", "source", "group", "split_v1", "text_model"] + [c for c in RISK_COLS if c in tr.columns]
    reg = pd.concat([tr[cols], te[cols], rl[cols + (["id", "label_user"] if "label_user" in rl else ["id"])]], ignore_index=True)
    reg["msg_id"] = [f"M{i:06d}" for i in range(len(reg))]
    reg["role"] = np.where(reg.split_v1 == "real", "real_benchmark", "pool")
    for k in ["label_provenance", "is_synthetic", "collection_method", "licence"]:
        reg[k] = reg.source.map(lambda s: PROVENANCE[s][k])
    reg["class3"] = reg.label3.map({"ham": "normal", "spam": "promotional", "smishing": "scam"})
    reg["lang_family"] = reg.language.map(lambda l: LANG_FAMILY.get(str(l), "mixed"))
    reg["text_pii_masked"] = reg.text.astype(str).map(lambda t: pii_mask(normalise(t)[0]))
    reg["family_id"] = families(reg.text_model, reg.group)

    # quarantine: identical model text with conflicting labels (ambiguous), within the trainable pool
    pool = reg.role == "pool"
    conflict = reg[pool].groupby("text_model").label3.transform("nunique") > 1
    reg["quarantine"] = False; reg["quarantine_reason"] = ""
    reg.loc[conflict[conflict].index, ["quarantine", "quarantine_reason"]] = [True, "identical text, conflicting labels"]

    # contamination audit of the ORIGINAL v1 split
    fam_split = reg[pool].groupby("family_id").split_v1.agg(lambda s: set(s))
    shared = fam_split[fam_split.map(len) > 1].index
    v1_test_contaminated = int(((reg.split_v1 == "test") & reg.family_id.isin(shared)).sum())
    real_fams = set(reg.loc[~pool, "family_id"]); real_contam = int((pool & reg.family_id.isin(real_fams)).sum())

    DATA.mkdir(parents=True, exist_ok=True); (RES / "data").mkdir(parents=True, exist_ok=True)
    keep = ["msg_id", "role", "split_v1", "source", "label3", "class3", "language", "lang_family", "label_provenance",
            "is_synthetic", "collection_method", "licence", "group", "family_id", "quarantine", "quarantine_reason",
            "text_pii_masked", "text_model"] + [c for c in RISK_COLS if c in reg.columns] + ["id", "label_user"]
    reg[[c for c in keep if c in reg.columns]].to_csv(DATA / "registry.csv", index=False, encoding="utf-8")
    reg[reg.quarantine].to_csv(DATA / "quarantine.csv", index=False, encoding="utf-8")

    # aggregate statistics (safe to publish: no message text)
    dist = pd.crosstab([reg.role, reg.source], reg.class3, margins=True)
    lang = pd.crosstab([reg.role, reg.lang_family], reg.class3, margins=True)
    dist.to_csv(RES / "data" / "class_by_source.csv"); lang.to_csv(RES / "data" / "class_by_language.csv")
    fam_size = reg[pool].groupby("family_id").size()
    md = ["# V2 data audit", "",
          f"- Registry rows: {len(reg)} (pool {int(pool.sum())}, real benchmark {int((~pool).sum())})",
          f"- Families (exact + near-duplicate cosine >= {NEAR_DUP} + template/translation groups): pool {fam_size.size}, "
          f"largest family {fam_size.max()} messages, families with >1 message: {(fam_size > 1).sum()}",
          f"- **V1 split contamination:** {v1_test_contaminated} of {int((reg.split_v1 == 'test').sum())} v1 test messages share a "
          f"family with v1 training messages -> v1 held-out test results are optimistic.",
          f"- Real benchmark vs pool: {real_contam} pool messages share a family with a real benchmark message.",
          f"- Quarantined (identical text, conflicting labels): {int(reg.quarantine.sum())}",
          f"- Synthetic or machine-translated rows in pool: {int(reg[pool].is_synthetic.sum())} "
          f"({reg[pool].is_synthetic.mean():.1%}); human-labelled real-text rows: {int((pool & ~reg.is_synthetic).sum())}", "",
          "## Class by source", "", dist.to_markdown(), "", "## Class by language family", "", lang.to_markdown(), "",
          "## Provenance", ""] + [f"- **{k}**: {v['label_provenance']}; {v['collection_method']}; licence: {v['licence']}"
                                   for k, v in PROVENANCE.items()]
    (RES / "data" / "data_audit.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md[:9]))


# ------------------------------------------------------------------ ingest new files
def read_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".xml":
        rows = []
        for el in ET.parse(path).getroot().iter("sms"):
            if el.get("type") not in (None, "1"):          # received messages only
                continue
            rows.append({"text": el.get("body", ""), "sender": el.get("address", ""), "date_ms": el.get("date", "")})
        return pd.DataFrame(rows).assign(label="", language="", source_detail="SMS Backup & Restore XML")
    for enc in ["utf-8-sig", "utf-16", "cp1252"]:
        try:
            df = pd.read_csv(path, encoding=enc, dtype=str).fillna("")
            break
        except (UnicodeError, UnicodeDecodeError):
            continue
    else:
        raise SystemExit(f"cannot decode {path}")
    low = {c.lower().strip(): c for c in df.columns}
    if "text" not in low and "message" not in low and "body" not in low:
        raise SystemExit("CSV needs a 'text' (or 'message' / 'body') column")
    df = df.rename(columns={low.get("text") or low.get("message") or low.get("body"): "text"})
    for c in ["label", "language", "sender"]:
        df[c] = df[low[c]] if c in low else ""
    return df.assign(source_detail=f"CSV ({path.name})")


def ingest(path: Path, role: str, source: str, licence: str) -> None:
    reg = pd.read_csv(DATA / "registry.csv")
    df = read_any(path)
    n0 = len(df)
    df["encoding_issue"] = df.text.str.contains("�|Ã.|â€", regex=True)
    norm = df.text.map(normalise)
    df["text_norm"], df["telco_flag_raw"] = norm.str[0], norm.str[1]
    df = df[df.text_norm.str.len() >= 5].copy()
    feats = pd.DataFrame([risk_features(t) for t in df.text_norm], index=df.index)
    df = pd.concat([df, feats], axis=1)
    df["text_pii_masked"] = df.text_norm.map(pii_mask)
    df["language"] = [l if l in LANG_FAMILY else d for l, d in zip(df.language.str.lower().str.strip(), df.language_detected)]
    df["label3"] = df.label.str.lower().str.strip().map(LABEL_IN)            # never inferred
    # duplicates within the file and against the registry
    df = df[~df.text_model.duplicated()]
    exact = df.text_model.isin(set(reg.text_model))
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True).fit(pd.concat([reg.text_model, df.text_model]).astype(str))
    nn = NearestNeighbors(n_neighbors=1, metric="cosine", algorithm="brute").fit(vec.transform(reg.text_model.astype(str)))
    d, i = nn.kneighbors(vec.transform(df.text_model.astype(str)))
    df["nearest_cosine"] = (1 - d[:, 0]).round(3)
    df["nearest_role"] = reg.role.to_numpy()[i[:, 0]]
    q = []
    for _, r in df.iterrows():
        why = []
        if pd.isna(r.label3): why.append("no / unrecognised label (needs manual labelling)")
        if r.encoding_issue: why.append("possible encoding problem")
        if r.nearest_cosine >= NEAR_DUP: why.append(f"near-duplicate of existing {r.nearest_role} message (cos {r.nearest_cosine})")
        q.append("; ".join(why))
    df["quarantine_reason"] = q
    df["quarantine"] = df.quarantine_reason != ""
    df = df[~exact]
    start = len(reg)
    df["msg_id"] = [f"N{start + k:06d}" for k in range(len(df))]
    df["role"], df["source"], df["licence"] = role, source, licence
    df["label_provenance"] = "human label supplied with the file" ; df["is_synthetic"] = False
    df["collection_method"] = df.source_detail
    df["class3"] = df.label3.map({"ham": "normal", "spam": "promotional", "smishing": "scam"})
    df["lang_family"] = df.language.map(lambda l: LANG_FAMILY.get(str(l), "mixed"))
    out = DATA / f"incoming_{role}.csv"
    keep = ["msg_id", "role", "source", "label3", "class3", "language", "lang_family", "label_provenance", "is_synthetic",
            "collection_method", "licence", "quarantine", "quarantine_reason", "nearest_cosine", "nearest_role",
            "text_pii_masked", "text_model"] + RISK_COLS
    prev = pd.read_csv(out) if out.exists() else pd.DataFrame(columns=keep)
    pd.concat([prev, df[keep]], ignore_index=True).to_csv(out, index=False, encoding="utf-8")
    print(f"{path.name}: {n0} rows -> {len(df)} new ({int(exact.sum())} exact duplicates of registry dropped); "
          f"quarantined {int(df.quarantine.sum())}; accepted {int((~df.quarantine).sum())} -> {out.relative_to(ROOT)}")
    print(df[~df.quarantine].groupby(["lang_family", "class3"]).size().to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["build", "ingest"])
    ap.add_argument("file", nargs="?", type=Path)
    ap.add_argument("--role", choices=["dev", "locked_test"], default="dev")
    ap.add_argument("--source", default="own phone export")
    ap.add_argument("--licence", default="research consent (own / family phones)")
    a = ap.parse_args()
    build() if a.cmd == "build" else ingest(a.file, a.role, a.source, a.licence)

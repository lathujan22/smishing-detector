"""Step 5b - Hand-crafted smishing features (the 8 flags fed to the hybrid model)
and the 3 sub-scores used by the severity engine.

    from src.features import extract_features
    extract_features("URGENT! Your BOC a/c is blocked. Verify: boc-verify.top")
    -> {'has_url': 1, 'url_shortener': 0, 'risky_domain': 1, 'has_phone': 0, 'has_money': 0,
        'urgency': 0.67, 'impersonation': 1.0, 'action_request': 1,
        'link_risk': 0.9, ...}

Build feature files for the whole dataset (+ a sanity table for the report):
    python -m src.features --build

The 8 model flags (all 0..1):
    has_url         any link
    url_shortener   bit.ly, rb.gy, clk.li ...
    risky_domain    risky TLD (.top .xyz .cc ...), IP address, or brand look-alike host
    has_phone       phone number to call / reply to
    has_money       money amount (Rs., LKR, රු., ரூ.)
    urgency         urgency / threat words, saturating (3 hits = 1.0)
    impersonation   brand/authority named, stronger with a look-alike link or a request
    action_request  asks to click / pay / verify / send a code (cancelled by "do not share")

Severity sub-scores (0..1): urgency (U), impersonation (I), link_risk (L).
"""
from __future__ import annotations

import argparse
import re
from functools import lru_cache
from pathlib import Path

from src.preprocess import preprocess

ROOT = Path(__file__).resolve().parents[1]
LEX = ROOT / "lexicons"
FLAG_NAMES = ["has_url", "url_shortener", "risky_domain", "has_phone", "has_money",
              "urgency", "impersonation", "action_request"]
IP_HOST = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$")
DEADLINE = re.compile(r"\bwithin\s+\d+\s*(?:hours?|hrs?|minutes?|mins?|days?)\b|\b\d+\s*(?:hours?|hrs?)\s*(?:only|left)\b"
                      r"|පැය\s*\d+|\d+\s*மணி", re.I)


# ----------------------------------------------------------------- lexicons
@lru_cache(maxsize=None)
def lexicon(name: str) -> tuple[str, ...]:
    lines = (LEX / f"{name}.txt").read_text(encoding="utf-8").splitlines()
    return tuple(l.strip().lower() for l in lines if l.strip() and not l.strip().startswith("#"))


@lru_cache(maxsize=None)
def matcher(name: str) -> re.Pattern:
    """Latin terms: whole-word (\"*\" = prefix). Sinhala/Tamil terms: substring."""
    parts = []
    for term in lexicon(name):
        if re.search(r"[a-z0-9]", term) and not re.search("[஀-௿඀-෿]", term):
            core = re.escape(term.rstrip("*")).replace(r"\ ", r"\s+")
            parts.append(rf"(?<![a-z0-9]){core}{'[a-z]*' if term.endswith('*') else ''}(?![a-z0-9])")
        else:
            parts.append(re.escape(term))
    return re.compile("|".join(parts), re.I)


def hits(name: str, text: str) -> list[str]:
    return [m.group(0).lower() for m in matcher(name).finditer(text)]


def is_official(host: str) -> bool:
    return any(host == d or host.endswith("." + d) for d in lexicon("official_domains"))


def is_shortener(host: str) -> bool:
    return host in lexicon("shorteners")


def is_risky(host: str) -> bool:
    if IP_HOST.match(host):
        return True
    if is_official(host):
        return False
    labels = host.split(".")
    if labels[-1] in lexicon("risky_tlds"):
        return True
    if len(labels) >= 3 and labels[-2] in {"us", "com", "net"} and labels[-1] in lexicon("risky_tlds"):
        return True
    sld = labels[-2] if len(labels) >= 2 else labels[0]
    if len(sld) <= 6 and re.search(r"\d", sld) and re.search(r"[a-z]", sld) and not is_shortener(host):
        return True                                          # random-looking host: f1n8.com, mfc0.com
    name = ".".join(labels[:-1])
    brandy = any(b in name for b in lexicon("brand_keywords"))
    lk_mimic = bool(re.search(r"[-.]lk\b|lk-|-gov|gov-", name)) and not host.endswith(".lk")
    return (brandy and not is_shortener(host)) or lk_mimic


# ----------------------------------------------------------------- features
def extract_features(text: str, pre: dict | None = None) -> dict:
    p = pre or preprocess(text)
    t = p["text_clean"]
    hosts = p["hosts"]
    official = [h for h in hosts if is_official(h)]
    shortener = any(is_shortener(h) for h in hosts)
    risky = any(is_risky(h) for h in hosts)

    urg_hits = hits("urgency", t) + DEADLINE.findall(t)
    urgency = min(1.0, len(set(urg_hits)) / 3)

    brand_hits = set(hits("impersonation", t))
    cred = bool(hits("credential", t))
    req = bool(hits("request", t))
    safe = bool(hits("safety_phrases", t))
    action_request = int(req and (cred or bool(p["urls"]) or bool(p["money"])) and not safe)

    impersonation = 0.0
    if brand_hits:
        impersonation = 0.4
        if risky or (p["urls"] and not official):
            impersonation += 0.3
        if action_request or (cred and not safe):
            impersonation += 0.3
        if official and not risky and not shortener:
            impersonation = min(impersonation, 0.3)       # genuine brand link
    impersonation = round(min(1.0, impersonation), 2)

    link_risk = 0.0
    if hosts:
        link_risk = 0.3
        link_risk += 0.3 if shortener else 0
        link_risk += 0.4 if risky else 0
        if brand_hits and not official:
            link_risk += 0.2
        if official and len(official) == len(hosts):
            link_risk = 0.1
    link_risk = round(min(1.0, link_risk), 2)

    return {
        "has_url": int(bool(hosts)),
        "url_shortener": int(shortener),
        "risky_domain": int(risky),
        "has_phone": int(bool(p["phones"]) or "<PHONE>" in p["text_model"]),
        "has_money": int(bool(p["money"])),
        "urgency": round(urgency, 2),
        "impersonation": impersonation,
        "action_request": action_request,
        "link_risk": link_risk,
        "n_urgency_hits": len(set(urg_hits)),
        "brands": "|".join(sorted(brand_hits)),
        "language_detected": p["language"],
        "telco_flag": p["telco_flag"],
        "text_model": p["text_model"],
    }


def flag_vector(text: str) -> list[float]:
    f = extract_features(text)
    return [float(f[k]) for k in FLAG_NAMES]


# ----------------------------------------------------------------- batch build
def build() -> None:
    import pandas as pd

    proc = ROOT / "data" / "processed"
    jobs = [(proc / "train.csv", proc / "train_feat.csv"), (proc / "test.csv", proc / "test_feat.csv"),
            (ROOT / "data" / "real_sms" / "real_test.csv", proc / "real_test_feat.csv")]
    summary = []
    for src, dst in jobs:
        if not src.exists():
            print(f"  skip {src.name} (not found)")
            continue
        df = pd.read_csv(src, encoding="utf-8")
        if "label3" not in df.columns:
            df["label3"] = df["label"]
        feats = pd.DataFrame([extract_features(t) for t in df["text"].astype(str)])
        out = pd.concat([df.reset_index(drop=True), feats.drop(columns=[c for c in feats if c in df])], axis=1)
        out.to_csv(dst, index=False, encoding="utf-8")
        print(f"  {src.name:<16} -> {dst.name:<20} {len(out):>6} rows")
        g = out.groupby("label3")[FLAG_NAMES + ["link_risk"]].mean().round(2)
        g.insert(0, "n", out.groupby("label3").size())
        summary.append(f"\n## {src.name}\n" + g.to_string())
    report = "Mean feature value by class (sanity check: smishing should be highest)\n" + "\n".join(summary)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "feature_summary.txt").write_text(report + "\n", encoding="utf-8")
    print("\n" + report)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", action="store_true", help="write *_feat.csv for train/test/real_test")
    ap.add_argument("text", nargs="?", help="score one message")
    a = ap.parse_args()
    if a.build:
        build()
    elif a.text:
        for k, v in extract_features(a.text).items():
            print(f"{k:>18}: {v}")
    else:
        ap.print_help()

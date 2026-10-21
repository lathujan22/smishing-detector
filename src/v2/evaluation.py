"""V2 evaluation: metrics with bootstrap confidence intervals, per-language, thresholds from development data."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)

from src.v2.common import SEED

MIN_N = 20          # below this many messages (or < 5 positives) per-group metrics are flagged unreliable


def binary(y, pred, s=None) -> dict:
    y, pred = np.asarray(y).astype(int), np.asarray(pred).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    two = s is not None and len(set(y)) == 2
    return {"n": len(y), "n_pos": int(y.sum()), "acc": accuracy_score(y, pred),
            "precision": precision_score(y, pred, zero_division=0), "recall": recall_score(y, pred, zero_division=0),
            "f1": f1_score(y, pred, zero_division=0),
            "macro_f1": f1_score(y, pred, average="macro", labels=[0, 1], zero_division=0),
            "fpr": fp / (fp + tn) if fp + tn else np.nan,
            "roc_auc": roc_auc_score(y, s) if two else np.nan, "auprc": average_precision_score(y, s) if two else np.nan,
            "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)}


def bootstrap_ci(y, pred, s, keys=("precision", "recall", "f1", "fpr", "auprc", "roc_auc"), n=1000, seed=SEED) -> dict:
    """Percentile 95% CI by resampling messages with replacement."""
    rng = np.random.RandomState(seed); y, pred = np.asarray(y), np.asarray(pred); s = None if s is None else np.asarray(s)
    vals = {k: [] for k in keys}
    for _ in range(n):
        i = rng.randint(0, len(y), len(y))
        m = binary(y[i], pred[i], None if s is None else s[i])
        for k in keys:
            vals[k].append(m[k])
    return {f"{k}_ci": f"[{np.nanpercentile(v, 2.5) * 100:.1f}, {np.nanpercentile(v, 97.5) * 100:.1f}]" for k, v in vals.items()}


def f1_threshold(y, s) -> float:
    cand = np.unique(np.quantile(s, np.linspace(0.3, 0.995, 300)))
    return float(cand[int(np.argmax([f1_score(y, (s >= t).astype(int), zero_division=0) for t in cand]))])


def per_language(lang_family, y, pred, s) -> pd.DataFrame:
    rows = []
    for g in ["EN", "SI", "TA", "mixed"]:
        m = np.asarray(lang_family) == g
        if m.sum():
            r = binary(np.asarray(y)[m], np.asarray(pred)[m], None if s is None else np.asarray(s)[m])
            r["reliable"] = bool(r["n"] >= MIN_N and r["n_pos"] >= 5)
            rows.append({"lang": g, **r})
    return pd.DataFrame(rows)


def pct(d: dict) -> dict:
    return {k: (round(v * 100, 2) if isinstance(v, (float, np.floating)) and not np.isnan(v) else v) for k, v in d.items()}

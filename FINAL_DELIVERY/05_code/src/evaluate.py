"""Shared evaluation helpers - used by baselines.py now and by the deep models later.

metrics(y_true, y_pred, y_score) -> Acc, Precision, Recall, F1, ROC-AUC, FPR
by_language(df, ...)             -> the same metrics per language group
mcnemar(y, pred_a, pred_b)       -> exact McNemar test between two models
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score)
from scipy.stats import binomtest

LANG_GROUP = {"en": "EN", "si": "SI", "si-en": "SI", "ta": "TA", "ta-en": "TA", "si-ta": "mixed", "mixed": "mixed"}


def to_binary(labels) -> np.ndarray:
    """ham -> 0, spam/smishing -> 1 (also accepts 0/1)."""
    s = pd.Series(labels).astype(str).str.lower()
    return (~s.isin(["ham", "0"])).astype(int).to_numpy()


def metrics(y_true, y_pred, y_score=None) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out = {
        "n": len(y_true),
        "acc": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score) if y_score is not None and len(set(y_true)) == 2 else np.nan,
        "fpr": fp / (fp + tn) if (fp + tn) else np.nan,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }
    return {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else int(v)) for k, v in out.items()}


def by_language(lang, y_true, y_pred, y_score=None) -> pd.DataFrame:
    g = pd.Series(lang).map(lambda l: LANG_GROUP.get(str(l), "mixed")).to_numpy()
    rows = []
    for grp in ["EN", "SI", "TA", "mixed"]:
        m = g == grp
        if m.sum():
            rows.append({"lang": grp, **metrics(np.asarray(y_true)[m], np.asarray(y_pred)[m],
                                                None if y_score is None else np.asarray(y_score)[m])})
    return pd.DataFrame(rows)


def mcnemar(y_true, pred_a, pred_b) -> dict:
    """Exact McNemar: do models A and B make significantly different errors?"""
    y, a, b = map(np.asarray, (y_true, pred_a, pred_b))
    a_ok, b_ok = a == y, b == y
    n01, n10 = int((a_ok & ~b_ok).sum()), int((~a_ok & b_ok).sum())
    p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    return {"a_right_b_wrong": n01, "a_wrong_b_right": n10, "p_value": round(float(p), 4)}

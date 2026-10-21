"""Audit of the real-SMS evaluation (leakage-safe repeat + class-definition check).
    python -m src.audit_real -> results/audit_real.csv, results/audit_real_scores.csv, results/AUDIT_REAL.md

Two label definitions:
  A "spam+scam"  positive = smishing OR promotional spam   (definition used so far)
  B "scam-only"  positive = smishing only; promotional spam counts as NEGATIVE (not a scam)
Protocol: 3 random repeats x 5 folds over the 163 real SMS; near-duplicates grouped at a STRICT cosine >= 0.3.
Per fold: noise augmentation, TF-IDF fitting, SMOTE and classifier training are done on the fold's
training part only (original training set + real SMS of the other folds, x5). Real SMS of the held-out
fold are never seen. Reported: mean +- std over repeats, and per-class flag rates
(normal flagged = false alarms, promo flagged, scam caught).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline

from src.augment import noisy_copies, smote
from src.baselines import SEED, tfidf
from src.evaluate import metrics
from src.final_baselines import CONF, tfidf_mindf3

ROOT = Path(__file__).resolve().parents[1]; PROC, RES = ROOT / "data/processed", ROOT / "results"
MODES = {"A spam+scam": lambda l: (l != "ham").astype(int).to_numpy(),
         "B scam-only": lambda l: (l == "smishing").astype(int).to_numpy()}
MODELS, REPEATS, REAL_WEIGHT, DUP_T = ["SVM", "RandomForest"], [42, 7, 2026], 5, 0.3


def fit(name, texts, y):
    n, make = CONF[name]
    Xa, ya, _, _ = noisy_copies(list(texts), y, seed=SEED)
    vec = tfidf_mindf3(n) if name == "RandomForest" else tfidf(n)
    M, yy = smote(vec.fit_transform(Xa), ya, k=5, seed=SEED)
    clf = make(); clf.fit(M, yy)
    return Pipeline([("tfidf", vec), ("clf", clf)])


def score(p, X):
    return p.decision_function(X) if hasattr(p.named_steps["clf"], "decision_function") else p.predict_proba(X)[:, 1]


def flags(lbl, pred):
    lbl = np.asarray(lbl)
    return {f"{c}_flagged": round(pred[lbl == c].mean() * 100, 2) for c in ["ham", "spam", "smishing"]}


if __name__ == "__main__":
    tr = pd.read_csv(PROC / "train_feat.csv"); te = pd.read_csv(PROC / "test_feat.csv")
    rl = pd.read_csv(PROC / "real_test_feat.csv")
    Xtr, Xte, Xrl = tr.text_model.astype(str), te.text_model.astype(str), rl.text_model.astype(str)
    M = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit_transform(Xrl)
    groups = connected_components(cosine_similarity(M) >= DUP_T, directed=False)[1]
    print(f"real {len(rl)} -> {len(set(groups))} groups at cosine >= {DUP_T}", flush=True)
    rows, scores = [], rl[["id", "label", "language"]].copy()
    for mode, lab in MODES.items():
        ytr, yte, yrl = lab(tr.label3), lab(te.label3), lab(rl.label)
        for name in MODELS:
            full = fit(name, Xtr, ytr)                                   # trained on the training set only
            pte = full.predict(Xte); m = metrics(yte, pte, score(full, Xte))
            rows.append({"mode": mode, "model": name, "set": "test", "repeat": "-",
                         **{k: round(m[k] * 100, 2) for k in ["acc", "precision", "recall", "f1", "fpr", "roc_auc"]},
                         **flags(te.label3, pte)})
            for rep in REPEATS:
                oof, oofs = np.zeros(len(rl), int), np.zeros(len(rl))
                for a, b in StratifiedGroupKFold(5, shuffle=True, random_state=rep).split(Xrl, rl.label, groups):
                    X = pd.concat([Xtr] + [Xrl.iloc[a]] * REAL_WEIGHT); y = np.concatenate([ytr] + [yrl[a]] * REAL_WEIGHT)
                    p = fit(name, X, y); oof[b] = p.predict(Xrl.iloc[b]); oofs[b] = score(p, Xrl.iloc[b])
                m = metrics(yrl, oof)
                rows.append({"mode": mode, "model": name, "set": "real (CV)", "repeat": rep,
                             **{k: round(m[k] * 100, 2) for k in ["acc", "precision", "recall", "f1", "fpr"]},
                             "roc_auc": round(roc_auc_score(yrl, oofs) * 100, 2), **flags(rl.label, oof)})
                scores[f"{mode[0]}_{name}_{rep}"] = oofs
                print(mode, name, rep, rows[-1]["f1"], rows[-1]["fpr"], flush=True)
    r = pd.DataFrame(rows); r.to_csv(RES / "audit_real.csv", index=False)
    scores.drop(columns=["label", "language"]).to_csv(RES / "audit_real_scores.csv", index=False)
    cols = ["acc", "precision", "recall", "f1", "fpr", "roc_auc", "ham_flagged", "spam_flagged", "smishing_flagged"]
    real = r[r.set == "real (CV)"].groupby(["mode", "model"])[cols]
    mean, std = real.mean().round(2), real.std().round(2)
    md = ["# Real-SMS audit (%, mean +- std over 3 repeats x 5 folds; near-duplicates grouped at cosine >= 0.3)", "",
          "| Definition | Model | Acc | Precision | Recall | F1 | FPR | AUC | Normal flagged | Promo flagged | Scam caught |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for (mo, mdl) in mean.index:
        md.append(f"| {mo} | {mdl} | " + " | ".join(f"{mean.loc[(mo, mdl), c]:.1f} ± {std.loc[(mo, mdl), c]:.1f}" for c in cols) + " |")
    md += ["", "## Held-out test set (n=1,589), models trained on the training set only", "",
           "| Definition | Model | Acc | Precision | Recall | F1 | FPR | AUC | Normal flagged | Promo flagged | Scam caught |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    md += [f"| {x['mode']} | {x['model']} | " + " | ".join(f"{x[c]:.1f}" for c in cols) + " |" for _, x in r[r.set == "test"].iterrows()]
    (RES / "AUDIT_REAL.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))

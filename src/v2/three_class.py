"""V2 three-class experiment: Normal (0) / Promotional (1) / Scam (2) + Experiment C (binary vs three-class SVM).
    python -m src.v2.three_class      (run after src.v2.svm_experiments)
Models: multiclass LinearSVC, multinomial Logistic Regression, Random Forest, Gradient Boosting (sklearn
GradientBoostingClassifier - a separate algorithm, NOT XGBoost). Same features, augmentation and multi-class SMOTE
as the SVM baseline, fitted inside training folds only; grouped CV on data/v2/train.csv (groups = family_id).
Outputs results/v2/three_class/: metrics.csv, by_language.csv, exp_c.csv, pred_*.csv, three_class_manifest.json
"""
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import label_binarize

from src.v2.common import CLASS3, CLASS_NAMES, DATA, MOD, RES, SEED, write_manifest
from src.v2.evaluation import MIN_N, binary, bootstrap_ci, pct
from src.v2.models import BASE, TextModel

OUT = RES / "three_class"; OUT.mkdir(parents=True, exist_ok=True)
MODELS = {"LinearSVC": {"clf": "svm"}, "LogisticRegression": {"clf": "logreg"},
          "RandomForest": {"clf": "rf"}, "GradientBoosting (sklearn, not XGBoost)": {"clf": "gb"}}


def multi(y, pred, s=None):
    y, pred = np.asarray(y), np.asarray(pred)
    out = {"n": len(y), "acc": accuracy_score(y, pred),
           "macro_precision": precision_score(y, pred, average="macro", labels=[0, 1, 2], zero_division=0),
           "macro_recall": recall_score(y, pred, average="macro", labels=[0, 1, 2], zero_division=0),
           "macro_f1": f1_score(y, pred, average="macro", labels=[0, 1, 2], zero_division=0)}
    for i, c in enumerate(CLASS_NAMES):
        out[f"{c}_precision"] = precision_score(y == i, pred == i, zero_division=0)
        out[f"{c}_recall"] = recall_score(y == i, pred == i, zero_division=0)
    if s is not None and len(set(y)) == 3:
        out["macro_roc_auc"] = roc_auc_score(y, s, multi_class="ovr", labels=[0, 1, 2])
        out["macro_auprc"] = average_precision_score(label_binarize(y, classes=[0, 1, 2]), s, average="macro")
    out["nonscam_to_scam_fpr"] = (pred[y != 2] == 2).mean()
    out["normal_to_scam"] = (pred[y == 0] == 2).mean() if (y == 0).any() else np.nan
    out["promo_to_scam"] = (pred[y == 1] == 2).mean() if (y == 1).any() else np.nan
    out["cm"] = json.dumps(confusion_matrix(y, pred, labels=[0, 1, 2]).tolist())
    return out


if __name__ == "__main__":
    t0 = time.time()
    tr = pd.read_csv(DATA / "train.csv"); te = pd.read_csv(DATA / "test.csv")
    reg = pd.read_csv(DATA / "registry.csv"); rl = reg[reg.role == "real_benchmark"].reset_index(drop=True)
    y, yte, yrl = [d.label3.map(CLASS3).to_numpy() for d in (tr, te, rl)]
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(tr, tr.class3 + "|" + tr.lang_family, tr.family_id))
    write_manifest(OUT, "three_class", {"base_cfg": {**BASE, "task": "three"}, "models": MODELS,
                                        "decision": "argmax over classes", "cv": "StratifiedGroupKFold(5), groups=family_id"},
                   {"train": DATA / "train.csv", "test": DATA / "test.csv", "registry": DATA / "registry.csv"})
    rows, langs = [], []
    preds = {"cv": tr[["msg_id", "lang_family", "class3"]].copy(), "test": te[["msg_id", "lang_family", "class3"]].copy(),
             "real": rl[["msg_id", "lang_family", "class3"]].copy()}
    for name, ch in MODELS.items():
        cfg = {**BASE, "task": "three", **ch}; t1 = time.time()
        oof_s, oof_p = np.zeros((len(tr), 3)), np.zeros(len(tr), int)
        for a, b in folds:
            m = TextModel(cfg).fit(tr.iloc[a], y[a]); oof_s[b] = m.score(tr.iloc[b]); oof_p[b] = m.predict(tr.iloc[b])
        m = TextModel(cfg).fit(tr, y)
        joblib.dump({"model": m, "cfg": cfg}, MOD / f"three_{name.split()[0].lower()}.joblib")
        sets = {"cv (validation)": (tr, y, oof_s, oof_p), "train (in-sample)": (tr, y, m.score(tr), m.predict(tr)),
                "held-out test": (te, yte, m.score(te), m.predict(te)), "real benchmark": (rl, yrl, m.score(rl), m.predict(rl))}
        for sname, (d, yy, s, p) in sets.items():
            r = {"model": name, "set": sname, **pct(multi(yy, p, s))}
            if sname in ("held-out test", "real benchmark"):     # CI for scam-vs-rest view
                r.update({f"scam_{k}": v for k, v in bootstrap_ci((yy == 2).astype(int), (p == 2).astype(int), s[:, 2],
                                                                  keys=("recall", "fpr", "precision")).items()})
            rows.append(r)
            if not sname.startswith("train"):
                for g in ["EN", "SI", "TA", "mixed"]:
                    k = d.lang_family.to_numpy() == g
                    if k.sum():
                        mm = multi(yy[k], p[k]); n_scam = int((yy[k] == 2).sum())
                        langs.append({"model": name, "set": sname, "lang": g, "n": int(k.sum()), "n_scam": n_scam,
                                      "reliable": bool(k.sum() >= MIN_N and n_scam >= 5),
                                      **pct({x: mm[x] for x in ["macro_f1", "scam_recall", "nonscam_to_scam_fpr", "normal_to_scam", "promo_to_scam"]})})
        for key, (s, p) in {"cv": (oof_s, oof_p), "test": sets["held-out test"][2:], "real": sets["real benchmark"][2:]}.items():
            preds[key][f"{name}|pred"] = p
            for i, c in enumerate(CLASS_NAMES):
                preds[key][f"{name}|p_{c}"] = s[:, i]
        print(f"{name:<42} CV macro-F1 {rows[-4]['macro_f1']:.2f} | test {rows[-2]['macro_f1']:.2f} | real {rows[-1]['macro_f1']:.2f} ({time.time()-t1:.0f}s)", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "metrics.csv", index=False); pd.DataFrame(langs).to_csv(OUT / "by_language.csv", index=False)
    for k, d in preds.items():
        d.to_csv(OUT / f"pred_{k}.csv", index=False)

    # ---------------- Experiment C: binary SVM vs three-class SVM, scam-vs-rest view (same base recipe)
    exc = []
    bt = pd.read_csv(RES / "svm" / "pred_test.csv"); br = pd.read_csv(RES / "svm" / "pred_real.csv")
    for sname, bdf, tdf, yy in [("held-out test", bt, preds["test"], yte), ("real benchmark", br, preds["real"], yrl)]:
        ys = (yy == 2).astype(int)
        for label, p in [("binary SVM (original recipe)", bdf["A: original SVM (baseline)|pred"].to_numpy()),
                         ("binary SVM (improved)", bdf["A: improved SVM (CV-selected)|pred"].to_numpy()),
                         ("three-class LinearSVC", (tdf["LinearSVC|pred"].to_numpy() == 2).astype(int))]:
            r = binary(ys, p)
            exc.append({"set": sname, "model": label, "scam_recall": round(r["recall"] * 100, 2),
                        "nonscam_to_scam_fpr": round(r["fpr"] * 100, 2), "precision": round(r["precision"] * 100, 2),
                        "f1": round(r["f1"] * 100, 2),
                        "normal_to_scam": round(p[yy == 0].mean() * 100, 2), "promo_to_scam": round(p[yy == 1].mean() * 100, 2),
                        **bootstrap_ci(ys, p, None, keys=("recall", "fpr"))})
    pd.DataFrame(exc).to_csv(OUT / "exp_c.csv", index=False)
    print(pd.DataFrame(exc).to_string(index=False)); print(f"total {time.time()-t0:.0f}s")

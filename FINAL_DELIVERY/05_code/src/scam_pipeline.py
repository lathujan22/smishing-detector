"""PRIMARY EXPERIMENT - scam-only smishing detection (+ separate three-class experiment).

    python -m src.scam_pipeline scam_only      # positive = smishing; normal AND promotional = negative
    python -m src.scam_pipeline three_class    # normal / promotional / scam

Data roles (never mixed)
    train   data/processed/train_feat.csv      training + grouped 5-fold CV (validation)
    test    data/processed/test_feat.csv       held-out test: public datasets + generated / translated templates
    real    data/processed/real_test_feat.csv  INDEPENDENT real Sri Lankan SMS - final evaluation ONLY
            (never used for training, hyper-parameters, thresholds or model choice)

Leakage control: inside every CV fold, noise augmentation, TF-IDF fitting, SMOTE and classifier
training use the fold's training part only. CV groups keep each multilingual template and each
machine-translated message with its English source in one fold (src.baselines.cv_groups).
Binary decision threshold: chosen on CV out-of-fold scores (training data only) to maximise F1;
the default threshold is reported as well.

Outputs  results/<task>/  config.json, metrics.csv, by_language.csv, cv_folds.csv, pred_*.csv, REPORT.md
         models/<task>/   <Model>.joblib, thresholds.json
The previous spam+scam results (results/*.csv, models/final, models/final_v2) are kept unchanged as the
baseline experiment.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import label_binarize
from sklearn.svm import LinearSVC

from src.augment import noisy_copies, smote, smote_multi
from src.baselines import cv_groups, tfidf
from src.evaluate import LANG_GROUP
from src.final_baselines import tfidf_mindf3

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
SEED = 42
CLASSES3 = ["normal", "promotional", "scam"]
TASKS = {
    "scam_only": lambda l: (pd.Series(l) == "smishing").astype(int).to_numpy(),
    "three_class": lambda l: pd.Series(l).map({"ham": 0, "spam": 1, "smishing": 2}).astype(int).to_numpy(),
}
NOISE = {"p_char": 0.15, "p_drop": 0.08, "p_swap": 0.05}


def boosting():
    if HAS_XGB:
        return XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.5,
                             tree_method="hist", random_state=SEED, n_jobs=-1)
    return GradientBoostingClassifier(n_estimators=400, learning_rate=0.1, max_depth=4, max_features="sqrt",
                                      subsample=0.8, random_state=SEED)


MODELS = {   # name: (vectoriser factory, classifier factory, description)
    "NaiveBayes": (lambda: tfidf(30000), lambda: ComplementNB(alpha=0.01), "ComplementNB(alpha=0.01), TF-IDF word+char 30k"),
    "SVM": (lambda: tfidf(10000), lambda: LinearSVC(C=0.1, random_state=SEED), "LinearSVC(C=0.1), TF-IDF word+char 10k"),
    "RandomForest": (lambda: tfidf_mindf3(8000),
                     lambda: RandomForestClassifier(200, max_features="sqrt", n_jobs=-1, random_state=SEED),
                     "RandomForest(200 trees, max_features=sqrt, full depth), TF-IDF min_df=3 8k"),
    "XGBoost": (lambda: tfidf(8000), boosting,
                ("XGBClassifier(400, depth 6, lr 0.1, subsample 0.8, colsample 0.5)" if HAS_XGB else
                 "FALLBACK: sklearn GradientBoostingClassifier(400, depth 4, lr 0.1, subsample 0.8, max_features=sqrt) "
                 "- xgboost not installable in this environment; the same script uses XGBoost where installed")
                + ", TF-IDF word+char 8k"),
}


def fit(name, texts, y, task):
    vec_f, clf_f, _ = MODELS[name]
    Xa, ya, _, _ = noisy_copies(list(texts), y, seed=SEED, **NOISE)          # training data only
    vec = vec_f()
    M = vec.fit_transform(Xa)                                                # fitted on training data only
    M, yy = (smote(M, ya, k=5, seed=SEED) if task == "scam_only" else smote_multi(M, ya, k=5, seed=SEED))
    clf = clf_f(); clf.fit(M, yy)
    return Pipeline([("tfidf", vec), ("clf", clf)])


def scores(pipe, X, task):
    clf = pipe.named_steps["clf"]
    if hasattr(clf, "decision_function") and not isinstance(clf, GradientBoostingClassifier):
        s = pipe.decision_function(X)
        if task == "three_class":
            e = np.exp(s - s.max(1, keepdims=True)); s = e / e.sum(1, keepdims=True)   # softmax for AUC
        return s
    p = pipe.predict_proba(X)
    return p[:, 1] if task == "scam_only" else p


# ---------------------------------------------------------------- metrics
def bin_metrics(y, pred, s=None):
    y, pred = np.asarray(y), np.asarray(pred)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    both = s is not None and len(set(y)) == 2
    return {"n": len(y), "n_pos": int(y.sum()), "acc": accuracy_score(y, pred),
            "precision": precision_score(y, pred, zero_division=0), "recall": recall_score(y, pred, zero_division=0),
            "f1": f1_score(y, pred, zero_division=0), "fpr": fp / (fp + tn) if fp + tn else np.nan,
            "roc_auc": roc_auc_score(y, s) if both else np.nan, "auprc": average_precision_score(y, s) if both else np.nan,
            "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)}


def multi_metrics(y, pred, s=None):
    y, pred = np.asarray(y), np.asarray(pred)
    out = {"n": len(y), "acc": accuracy_score(y, pred),
           "macro_precision": precision_score(y, pred, average="macro", labels=[0, 1, 2], zero_division=0),
           "macro_recall": recall_score(y, pred, average="macro", labels=[0, 1, 2], zero_division=0),
           "macro_f1": f1_score(y, pred, average="macro", labels=[0, 1, 2], zero_division=0)}
    for i, c in enumerate(CLASSES3):
        out[f"{c}_precision"] = precision_score(y == i, pred == i, zero_division=0)
        out[f"{c}_recall"] = recall_score(y == i, pred == i, zero_division=0)
        out[f"{c}_f1"] = f1_score(y == i, pred == i, zero_division=0)
    if s is not None and len(set(y)) == 3:
        out["macro_roc_auc"] = roc_auc_score(y, s, multi_class="ovr", average="macro", labels=[0, 1, 2])
        out["macro_auprc"] = average_precision_score(label_binarize(y, classes=[0, 1, 2]), s, average="macro")
    nonscam = y != 2                                                          # scam-vs-rest view of the 3-class model
    out["scam_fpr"] = (pred[nonscam] == 2).mean() if nonscam.any() else np.nan
    out["normal_flagged_as_scam"] = (pred[y == 0] == 2).mean() if (y == 0).any() else np.nan
    out["promo_flagged_as_scam"] = (pred[y == 1] == 2).mean() if (y == 1).any() else np.nan
    return out


def by_language(lang, y, pred, s, task):
    g = pd.Series(lang).map(lambda l: LANG_GROUP.get(str(l), "mixed")).to_numpy()
    rows = []
    for grp in ["EN", "SI", "TA", "mixed"]:
        m = g == grp
        if m.sum():
            f = bin_metrics if task == "scam_only" else multi_metrics
            rows.append({"lang": grp, **f(np.asarray(y)[m], np.asarray(pred)[m], None if s is None else np.asarray(s)[m])})
    return rows


def best_threshold(y, s):
    """Threshold on CV out-of-fold scores (training data) that maximises F1."""
    cand = np.unique(np.quantile(s, np.linspace(0.5, 0.995, 400)))
    f1s = [f1_score(y, (s >= t).astype(int), zero_division=0) for t in cand]
    return float(cand[int(np.argmax(f1s))])


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def rnd(d):
    return {k: (round(float(v) * 100, 2) if isinstance(v, (float, np.floating)) and k not in ("threshold",) else v)
            for k, v in d.items()}


# ---------------------------------------------------------------- main
def main(task: str) -> None:
    t0 = time.time()
    RES = ROOT / "results" / task; MOD = ROOT / "models" / task
    RES.mkdir(parents=True, exist_ok=True); MOD.mkdir(parents=True, exist_ok=True)
    tr = pd.read_csv(PROC / "train_feat.csv"); te = pd.read_csv(PROC / "test_feat.csv")
    rl = pd.read_csv(PROC / "real_test_feat.csv")
    lab = TASKS[task]
    Xtr, Xte, Xrl = (d.text_model.astype(str).reset_index(drop=True) for d in (tr, te, rl))
    ytr, yte, yrl = lab(tr.label3), lab(te.label3), lab(rl.label)
    groups = cv_groups(tr)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = list(cv.split(Xtr, tr.label3, groups))
    M = bin_metrics if task == "scam_only" else multi_metrics

    config = {
        "task": task, "seed": SEED, "created": time.strftime("%Y-%m-%d %H:%M"),
        "labels": ("positive = smishing (scam); negative = ham (normal) + spam (promotional)" if task == "scam_only"
                   else "0 = normal (ham), 1 = promotional (spam), 2 = scam (smishing)"),
        "data": {k: {"path": str(p.relative_to(ROOT)), "rows": int(n), "md5": md5(p)} for k, p, n in
                 [("train", PROC / "train_feat.csv", len(tr)), ("test", PROC / "test_feat.csv", len(te)),
                  ("real", PROC / "real_test_feat.csv", len(rl))]},
        "class_counts": {"train": tr.label3.value_counts().to_dict(), "test": te.label3.value_counts().to_dict(),
                         "real": rl.label.value_counts().to_dict()},
        "cv": {"method": "StratifiedGroupKFold", "n_splits": 5, "shuffle": True, "random_state": SEED,
               "stratify_on": "label3", "groups": "multilingual template id / machine-translation source",
               "n_groups": int(len(set(groups)))},
        "augmentation": {"method": "noisy copy (typos, look-alike letters, word drop/swap)", "copies": 1, **NOISE,
                         "seed": SEED, "scope": "training part of each fold / full training set only"},
        "resampling": {"method": "SMOTE on TF-IDF (cosine kNN, k=5)",
                       "target": "minority to 1:1" if task == "scam_only" else "every class to the majority size",
                       "scope": "after TF-IDF fit, training part only"},
        "models": {k: v[2] for k, v in MODELS.items()},
        "hyperparameters_note": ("Fixed settings carried over from the earlier grouped-CV tuning on the training set "
                                 "(NB alpha, RF depth/features by CV; SVM C=0.1 chosen to reduce the train-test gap). "
                                 "No test or real SMS was used to choose them."),
        "threshold": ("binary: CV out-of-fold F1-optimal threshold (training data only); default also reported"
                      if task == "scam_only" else "argmax over classes"),
        "real_sms_policy": "independent final evaluation only - not used for training, tuning, thresholds or model choice",
        "environment": {"python": sys.version.split()[0], "platform": platform.platform(), "sklearn": sklearn.__version__,
                        "numpy": np.__version__, "scipy": scipy.__version__, "pandas": pd.__version__,
                        "xgboost_available": HAS_XGB},
    }
    (RES / "config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")
    print(f"[{task}] train {len(tr)} | test {len(te)} | real {len(rl)} | {len(set(groups))} CV groups | xgboost={HAS_XGB}", flush=True)

    rows, lang_rows, fold_rows, thresholds = [], [], [], {}
    preds = {s: d[["text", "language"] + (["label3"] if s != "real" else ["id", "label"])].copy()
             for s, d in [("cv", tr), ("test", te), ("real", rl)]}
    for name in MODELS:
        t1 = time.time()
        shape = (len(tr),) if task == "scam_only" else (len(tr), 3)
        oof_s, oof_p = np.zeros(shape), np.zeros(len(tr), int)
        for k, (a, b) in enumerate(folds):
            p = fit(name, Xtr.iloc[a], ytr[a], task)
            oof_s[b], oof_p[b] = scores(p, Xtr.iloc[b], task), p.predict(Xtr.iloc[b])
            fold_rows.append({"model": name, "fold": k + 1, "n_val": len(b), **rnd(M(ytr[b], oof_p[b], oof_s[b]))})
        full = fit(name, Xtr, ytr, task)
        joblib.dump(full, MOD / f"{name}.joblib")
        sets = {"train (in-sample)": (Xtr, ytr, tr.language), "test": (Xte, yte, te.language), "real": (Xrl, yrl, rl.language)}
        out_s = {s: scores(full, X, task) for s, (X, _, _) in sets.items()}
        out_p = {s: full.predict(X) for s, (X, _, _) in sets.items()}
        variants = {"default": None}
        if task == "scam_only":
            thr = best_threshold(ytr, oof_s); thresholds[name] = thr; variants["cv_f1"] = thr
        for vname, thr in variants.items():
            P = (lambda s, d: d) if thr is None else (lambda s, d: (s >= thr).astype(int))
            cvp = P(oof_s, oof_p)
            rows.append({"model": name, "set": "cv (validation)", "threshold": vname, **rnd(M(ytr, cvp, oof_s))})
            for s, (X, y, lg) in sets.items():
                pr = P(out_s[s], out_p[s])
                rows.append({"model": name, "set": s, "threshold": vname, **rnd(M(y, pr, out_s[s]))})
                if s != "train (in-sample)":
                    lang_rows += [{"model": name, "set": s, "threshold": vname, **rnd(r)} for r in by_language(lg, y, pr, out_s[s], task)]
                    key = "test" if s == "test" else "real"
                    preds[key][f"{name}_{vname}_pred"] = pr
            lang_rows += [{"model": name, "set": "cv (validation)", "threshold": vname, **rnd(r)}
                          for r in by_language(tr.language, ytr, cvp, oof_s, task)]
            preds["cv"][f"{name}_{vname}_pred"] = cvp
        for key, s in [("cv", oof_s), ("test", out_s["test"]), ("real", out_s["real"])]:
            if task == "scam_only":
                preds[key][f"{name}_score"] = s
            else:
                for i, c in enumerate(CLASSES3):
                    preds[key][f"{name}_score_{c}"] = s[:, i]
        print(f"  {name:<12} done in {time.time() - t1:.0f}s", flush=True)

    pd.DataFrame(rows).to_csv(RES / "metrics.csv", index=False)
    pd.DataFrame(lang_rows).to_csv(RES / "by_language.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(RES / "cv_folds.csv", index=False)
    for k, d in preds.items():
        d.to_csv(RES / f"pred_{k}.csv", index=False, encoding="utf-8")
    if thresholds:
        (MOD / "thresholds.json").write_text(json.dumps(thresholds, indent=2), encoding="utf-8")
    config["runtime_seconds"] = round(time.time() - t0)
    (RES / "config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")
    print(f"[{task}] finished in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "scam_only")

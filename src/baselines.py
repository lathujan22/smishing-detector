"""Step 6 - Baseline models: TF-IDF + Naive Bayes / linear SVM / Random Forest.

    python -m src.baselines            # 10-fold CV on train, final test, real-SMS test
    python -m src.baselines --fast     # 3-fold CV (quick check)

Input   data/processed/{train,test,real_test}_feat.csv  (from `python -m src.features --build`)
Output  results/baselines.csv                 CV mean±std + test + real-SMS scores per model
        results/baselines_by_language.csv     test and real-SMS scores per language
        results/pred_baselines_{test,real}.csv  per-message predictions (for McNemar later)
        models/baselines/*.joblib             fitted pipelines

Text = `text_model` (cleaned + masked). Features = word 1-2 grams + character 2-5 grams:
character n-grams work across English, Sinhala, Tamil and romanised spelling variants
without any language-specific tokeniser. Class imbalance -> class_weight='balanced'.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from src.evaluate import by_language, metrics, to_binary

ROOT = Path(__file__).resolve().parents[1]
PROC, RES, MOD = ROOT / "data" / "processed", ROOT / "results", ROOT / "models" / "baselines"
SEED = 42


def tfidf(max_features: int) -> FeatureUnion:
    return FeatureUnion([
        ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                                 token_pattern=r"<\w+>|[^\s<>.,!?:;()\"']+", max_features=max_features)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True,
                                 max_features=max_features)),
    ])


def models() -> dict[str, Pipeline]:
    return {
        "NaiveBayes": Pipeline([("tfidf", tfidf(30000)), ("clf", ComplementNB(alpha=0.3))]),
        "SVM": Pipeline([("tfidf", tfidf(50000)), ("clf", LinearSVC(C=0.5, class_weight="balanced", random_state=SEED))]),
        "RandomForest": Pipeline([("tfidf", tfidf(8000)),
                                  ("clf", RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                                                 min_samples_leaf=1, n_jobs=-1, random_state=SEED))]),
    }


def cv_groups(tr: pd.DataFrame) -> np.ndarray:
    """CV groups: all variants of one multilingual template share a group, so a
    pattern never sits in both the training and validation fold (same rule as data_prep)."""
    if "group" in tr.columns and tr["group"].notna().all():
        return tr["group"].astype(str).to_numpy()          # written by data_prep (templates + MT sources)
    ml = ROOT / "data" / "raw" / "multilingual_train.csv"
    key = lambda t: " ".join(str(t).split()).lower()
    tid = {}
    if ml.exists():
        m = pd.read_csv(ml, encoding="utf-8")
        if "template_id" in m.columns:
            tid = dict(zip(m["text"].map(key), m["template_id"]))
    return np.array([tid.get(key(t), f"row{i}") for i, t in enumerate(tr["text"])])


def score(pipe: Pipeline, X) -> np.ndarray:
    clf = pipe.named_steps["clf"]
    return pipe.decision_function(X) if hasattr(clf, "decision_function") else pipe.predict_proba(X)[:, 1]


def main(folds: int) -> None:
    tr = pd.read_csv(PROC / "train_feat.csv", encoding="utf-8")
    te = pd.read_csv(PROC / "test_feat.csv", encoding="utf-8")
    real_path = PROC / "real_test_feat.csv"
    rl = pd.read_csv(real_path, encoding="utf-8") if real_path.exists() else None
    Xtr, ytr = tr["text_model"].astype(str), to_binary(tr["label3"])
    Xte, yte = te["text_model"].astype(str), to_binary(te["label3"])
    print(f"train {len(tr)} (pos {ytr.mean():.1%}) | test {len(te)} | real SMS {0 if rl is None else len(rl)} | {folds}-fold CV\n")

    RES.mkdir(exist_ok=True); MOD.mkdir(parents=True, exist_ok=True)
    rows, lang_rows = [], []
    pred_te = te[["text", "label3", "language"]].copy()
    pred_rl = rl[["id", "text", "label", "language"]].copy() if rl is not None else None
    cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=SEED)
    groups = cv_groups(tr)
    print(f"CV groups: {len(set(groups))} (multilingual templates kept whole)")

    for name, pipe in models().items():
        t0 = time.time()
        cvr = cross_validate(pipe, Xtr, ytr, groups=groups, cv=cv, n_jobs=1,
                             scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])
        pipe.fit(Xtr, ytr)
        joblib.dump(pipe, MOD / f"{name}.joblib")
        p, s = pipe.predict(Xte), score(pipe, Xte)
        m_te = metrics(yte, p, s)
        pred_te[f"{name}_pred"], pred_te[f"{name}_score"] = p, s
        row = {"model": name,
               **{f"cv_{k}": round(cvr[f"test_{k}"].mean(), 4) for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]},
               "cv_f1_std": round(cvr["test_f1"].std(), 4),
               **{f"test_{k}": m_te[k] for k in ["acc", "precision", "recall", "f1", "roc_auc", "fpr"]}}
        lang_rows.append(by_language(te["language"], yte, p, s).assign(model=name, set="test"))
        if rl is not None:
            yrl = to_binary(rl["label"])
            pr, sr = pipe.predict(rl["text_model"].astype(str)), score(pipe, rl["text_model"].astype(str))
            m_rl = metrics(yrl, pr, sr)
            row.update({f"real_{k}": m_rl[k] for k in ["acc", "precision", "recall", "f1", "fpr"]})
            row["real_smishing_recall"] = round(float(pr[rl["label"].eq("smishing").to_numpy()].mean()), 4)
            pred_rl[f"{name}_pred"], pred_rl[f"{name}_score"] = pr, sr
            lang_rows.append(by_language(rl["language"], yrl, pr, sr).assign(model=name, set="real"))
        row["train_seconds"] = round(time.time() - t0, 1)
        rows.append(row)
        print(f"{name:<13} CV F1 {row['cv_f1']:.3f}±{row['cv_f1_std']:.3f} | test F1 {m_te['f1']:.3f} "
              f"FPR {m_te['fpr']:.3%} | real F1 {row.get('real_f1', float('nan')):.3f}  ({row['train_seconds']}s)")

    res = pd.DataFrame(rows)
    res.to_csv(RES / "baselines.csv", index=False)
    langs = pd.concat(lang_rows)[["model", "set", "lang", "n", "acc", "precision", "recall", "f1", "fpr", "tp", "fp", "tn", "fn"]]
    langs.to_csv(RES / "baselines_by_language.csv", index=False)
    pred_te.to_csv(RES / "pred_baselines_test.csv", index=False, encoding="utf-8")
    if pred_rl is not None:
        pred_rl.to_csv(RES / "pred_baselines_real.csv", index=False, encoding="utf-8")

    pd.set_option("display.width", 200)
    print("\n== Summary ==\n" + res.drop(columns=["train_seconds"]).to_string(index=False))
    print("\n== Test set by language ==\n" + langs[langs.set == "test"].drop(columns=["set"]).to_string(index=False))
    if pred_rl is not None:
        print("\n== Real SMS by language ==\n" + langs[langs.set == "real"].drop(columns=["set"]).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fast", action="store_true", help="3-fold CV instead of 10")
    main(3 if ap.parse_args().fast else 10)

"""Reduce the train-test gap: regularised baselines vs current.   python -m src.regularise -> results/regularise.csv"""
import time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from src.baselines import tfidf, models, cv_groups, SEED
from src.balance_check import reweight
from src.evaluate import to_binary

ROOT = Path(__file__).resolve().parents[1]
tr = pd.read_csv(ROOT / "data/processed/train_feat.csv"); te = pd.read_csv(ROOT / "data/processed/test_feat.csv")
G = {"en": "EN", "si": "SI", "si-en": "SI", "ta": "TA", "ta-en": "TA"}; tr["g"] = tr.language.map(G).fillna("mixed")
X, y, g = tr.text_model.astype(str), to_binary(tr.label3), cv_groups(tr); Xt, yt = te.text_model.astype(str), to_binary(te.label3)
w = reweight(tr, pd.Series(y))
cv = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(X, y, g))
P = lambda n, clf: Pipeline([("tfidf", tfidf(n)), ("clf", clf)])
C = {
 "SVM current (C=0.5, 50k feat)":            models()["SVM"],
 "SVM C=0.1, 10k feat":                      P(10000, LinearSVC(C=0.1, class_weight="balanced", random_state=SEED)),
 "SVM C=0.05, 5k feat":                      P(5000, LinearSVC(C=0.05, class_weight="balanced", random_state=SEED)),
 "LogReg C=1, 10k feat":                     P(10000, LogisticRegression(C=1, class_weight="balanced", max_iter=2000)),
 "LogReg C=0.5, 5k feat":                    P(5000, LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000)),
 "RF current (full trees)":                  models()["RandomForest"],
 "RF depth=30, leaf=2, 50% bootstrap":       P(8000, RandomForestClassifier(300, max_depth=30, min_samples_leaf=2, max_samples=0.5, class_weight="balanced_subsample", n_jobs=-1, random_state=SEED)),
 "RF depth=20, leaf=3, 50% bootstrap, 3k":   P(3000, RandomForestClassifier(300, max_depth=20, min_samples_leaf=3, max_samples=0.5, class_weight="balanced_subsample", n_jobs=-1, random_state=SEED)),
}
rows = []
for name, pipe in C.items():
    t = time.time()
    cvf = cross_val_score(pipe, X, y, groups=g, cv=cv, scoring="f1", params={"clf__sample_weight": w}).mean()
    pipe.fit(X, y, clf__sample_weight=w)
    trf, tef = f1_score(y, pipe.predict(X)), f1_score(yt, pipe.predict(Xt))
    rows.append({"model": name, "train_f1": trf, "cv_f1": cvf, "test_f1": tef, "gap_train_test": trf - tef})
    print(f"{name:<40} train {100*trf:6.2f}  cv {100*cvf:6.2f}  test {100*tef:6.2f}  gap {100*(trf-tef):5.2f}  ({time.time()-t:.0f}s)", flush=True)
d = pd.DataFrame(rows); d.iloc[:, 1:] = (d.iloc[:, 1:] * 100).round(2); d.to_csv(ROOT / "results/regularise.csv", index=False)

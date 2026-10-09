"""Overfitting check: train vs 5-fold grouped CV vs test for regularised SVM / Random Forest settings.
    python -m src.overfit_check  -> results/overfit_check.csv, results/figures/fig_learning_curve.png"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score, learning_curve
from src.baselines import models, cv_groups, SEED
from src.evaluate import to_binary

ROOT = Path(__file__).resolve().parents[1]
tr = pd.read_csv(ROOT / "data/processed/train_feat.csv"); te = pd.read_csv(ROOT / "data/processed/test_feat.csv")
X, y, g = tr.text_model.astype(str), to_binary(tr.label3), cv_groups(tr)
Xt, yt = te.text_model.astype(str), to_binary(te.label3)
cv = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(X, y, g))
m = models()
CONFIGS = [("SVM", "C=0.05", {"clf__C": 0.05}), ("SVM", "C=0.1", {"clf__C": 0.1}), ("SVM", "C=0.5 (used)", {"clf__C": 0.5}), ("SVM", "C=2", {"clf__C": 2}),
           ("RandomForest", "default (used)", {}),
           ("RandomForest", "min_leaf=3", {"clf__min_samples_leaf": 3}),
           ("RandomForest", "min_leaf=5, depth=40", {"clf__min_samples_leaf": 5, "clf__max_depth": 40}),
           ("RandomForest", "min_leaf=10, depth=25", {"clf__min_samples_leaf": 10, "clf__max_depth": 25})]
rows = []
for name, label, params in CONFIGS:
    t = time.time(); p = clone(m[name]).set_params(**params)
    cvf = cross_val_score(p, X, y, groups=g, cv=cv, scoring="f1").mean()
    p.fit(X, y)
    r = {"model": name, "setting": label, "train_f1": f1_score(y, p.predict(X)), "cv_f1": cvf, "test_f1": f1_score(yt, p.predict(Xt))}
    r["gap_train_cv"] = r["train_f1"] - r["cv_f1"]; rows.append(r)
    print(f"{name:<12} {label:<22} train {100*r['train_f1']:.2f}  cv {100*cvf:.2f}  test {100*r['test_f1']:.2f}  gap {100*r['gap_train_cv']:.2f}  ({time.time()-t:.0f}s)", flush=True)
d = pd.DataFrame(rows); d.iloc[:, 2:] = (d.iloc[:, 2:] * 100).round(2); d.to_csv(ROOT / "results/overfit_check.csv", index=False)

# learning curves (train vs CV F1 as training size grows) - the classic overfitting diagnostic
fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
for ax, (name, title) in zip(axes, [("SVM", "Linear SVM"), ("RandomForest", "Random Forest")]):
    sizes, trs, vas = learning_curve(m[name], X, y, groups=g, cv=cv, train_sizes=[0.1, 0.25, 0.5, 0.75, 1.0], scoring="f1", n_jobs=1)
    for s, c, lab in [(trs, "#2f5f98", "Training F1"), (vas, "#c0392b", "Validation F1 (5-fold CV)")]:
        mu, sd = 100 * s.mean(1), 100 * s.std(1)
        ax.plot(sizes, mu, "o-", color=c, label=lab); ax.fill_between(sizes, mu - sd, mu + sd, color=c, alpha=.15)
    ax.set_title(f"{title} - learning curve", weight="bold"); ax.set_xlabel("Training messages"); ax.set_ylabel("F1 (%)")
    ax.grid(alpha=.25); ax.legend(frameon=False, loc="lower right"); ax.spines[["top", "right"]].set_visible(False)
    print(name, "learning curve val:", np.round(100 * vas.mean(1), 2), flush=True)
fig.tight_layout(); fig.savefig(ROOT / "results/figures/fig_learning_curve.png", dpi=300, facecolor="white")

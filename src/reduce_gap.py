"""Find regularised settings that lower train F1 (less memorisation) while keeping test F1.
    python -m src.reduce_gap SVM | RandomForest  -> results/reduce_gap_<model>.csv
All runs: language re-weighting off, noise augmentation + SMOTE on (training only)."""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC
from src.augment import noisy_copies, smote
from src.baselines import cv_groups, SEED
from src.evaluate import to_binary

ROOT = Path(__file__).resolve().parents[1]
tr = pd.read_csv(ROOT / "data/processed/train_feat.csv"); te = pd.read_csv(ROOT / "data/processed/test_feat.csv")
rl = pd.read_csv(ROOT / "data/processed/real_test_feat.csv")
X, y, grp = tr.text_model.astype(str).tolist(), to_binary(tr.label3), cv_groups(tr)
Xt, yt = te.text_model.astype(str).tolist(), to_binary(te.label3)
Xr, yr = rl.text_model.astype(str).tolist(), to_binary(rl.label)

def vec(n, min_df=3):
    return FeatureUnion([("word", TfidfVectorizer(ngram_range=(1, 2), min_df=min_df, sublinear_tf=True, max_features=n,
                                                  token_pattern=r"<\w+>|[^\s<>.,!?:;()\"']+")),
                         ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=min_df, sublinear_tf=True, max_features=n))])

def fit(clf_fn, n, Xa, ya):
    Xa, ya, _, _ = noisy_copies(Xa, ya, seed=SEED)
    v = vec(n); M, yy = smote(v.fit_transform(Xa), ya, seed=SEED)
    clf = clf_fn(); clf.fit(M, yy); return lambda T: clf.predict(v.transform(T))

GRID = {
 "SVM": [(f"C={c}, {n//1000}k feat", (lambda c=c: LinearSVC(C=c, random_state=SEED)), n)
         for c, n in [(0.1, 10000), (0.03, 10000), (0.03, 3000), (0.02, 2000), (0.01, 3000), (0.01, 1000), (0.005, 1000)]],
 "RandomForest": [(f"depth={d}, leaf={l}, {n//1000}k feat", (lambda d=d, l=l: RandomForestClassifier(200, max_depth=d, min_samples_leaf=l,
                    max_features="sqrt", n_jobs=-1, random_state=SEED)), n)
         for d, l, n in [(None, 1, 8000), (40, 2, 3000), (25, 3, 3000), (20, 5, 2000), (15, 5, 1000), (12, 8, 1000)]],
}
model = sys.argv[1]
cv = list(StratifiedGroupKFold(3, shuffle=True, random_state=SEED).split(X, y, grp)); Xa = np.array(X, dtype=object)
rows = []
for name, fn, n in GRID[model]:
    t = time.time()
    cvf = np.mean([f1_score(y[va], fit(fn, n, Xa[tri].tolist(), y[tri])(Xa[va].tolist())) for tri, va in cv])
    p = fit(fn, n, X, y)
    r = {"model": model, "setting": name, "train_f1": f1_score(y, p(X)), "cv_f1": cvf, "test_f1": f1_score(yt, p(Xt)), "real_f1": f1_score(yr, p(Xr))}
    r["gap_train_test"] = r["train_f1"] - r["test_f1"]; rows.append(r)
    print(f"{name:<28} train {100*r['train_f1']:6.2f}  cv {100*cvf:6.2f}  test {100*r['test_f1']:6.2f}  real {100*r['real_f1']:6.2f}  gap {100*r['gap_train_test']:5.2f}  ({time.time()-t:.0f}s)", flush=True)
d = pd.DataFrame(rows); d.iloc[:, 2:] = (d.iloc[:, 2:] * 100).round(2); d.to_csv(ROOT / f"results/reduce_gap_{model}.csv", index=False)

"""Step 6b - Retrain baselines with noise augmentation and/or SMOTE (training data only).
    python -m src.noise_smote SVM            (or RandomForest)
    -> results/noise_smote_<model>.csv ; results/NOISE_SMOTE.md (combined)
Evaluation: train F1 (original msgs), 3-fold grouped CV F1 (augmentation inside each fold),
test F1, NOISY test F1 (robustness to typos / look-alike spelling), real-SMS F1, SI / TA test F1."""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.svm import LinearSVC
from src.augment import add_noise, noisy_copies, smote
from src.balance_check import reweight
from src.baselines import tfidf, cv_groups, SEED
from src.evaluate import to_binary, by_language

ROOT = Path(__file__).resolve().parents[1]
G = {"en": "EN", "si": "SI", "si-en": "SI", "ta": "TA", "ta-en": "TA"}
tr = pd.read_csv(ROOT / "data/processed/train_feat.csv"); tr["g"] = tr.language.map(G).fillna("mixed")
te = pd.read_csv(ROOT / "data/processed/test_feat.csv"); rl = pd.read_csv(ROOT / "data/processed/real_test_feat.csv")
X, y, grp = tr.text_model.astype(str).tolist(), to_binary(tr.label3), cv_groups(tr)
W = reweight(tr, pd.Series(y))
rng = np.random.RandomState(7)
te_noisy = [add_noise(t, rng, p_char=0.25, p_drop=0.1) for t in te.text_model.astype(str)]   # fixed noisy test

MODELS = {"SVM": lambda cw: LinearSVC(C=0.1, class_weight=cw, random_state=SEED),
          "RandomForest": lambda cw: RandomForestClassifier(200, class_weight=cw, n_jobs=-1, random_state=SEED)}
FEATS = {"SVM": 10000, "RandomForest": 8000}


def fit(model, Xt, yt, wt, noise, use_smote):
    if noise:
        Xt, yt, _, wt = noisy_copies(Xt, yt, None, wt, seed=SEED)
    vec = tfidf(FEATS[model]); M = vec.fit_transform(Xt)
    if use_smote:                                   # SMOTE balances the classes -> no extra class weights
        M, yt = smote(M, yt, k=5, seed=SEED); clf = MODELS[model](None); clf.fit(M, yt)
    else:
        clf = MODELS[model](None); clf.fit(M, yt, sample_weight=wt)
    return lambda texts: clf.predict(vec.transform(texts))


def fpr(yt, p):
    tn, fp, fn, tp = confusion_matrix(yt, p, labels=[0, 1]).ravel(); return fp / (fp + tn)


model = sys.argv[1] if len(sys.argv) > 1 else "SVM"
cv = list(StratifiedGroupKFold(3, shuffle=True, random_state=SEED).split(X, y, grp))
Xa = np.array(X, dtype=object)
rows = []
for name, noise, sm in [("no augmentation", 0, 0), ("+ noise", 1, 0), ("+ SMOTE", 0, 1), ("+ noise + SMOTE", 1, 1)]:
    t0 = time.time()
    cvf = [f1_score(y[va], fit(model, Xa[tri].tolist(), y[tri], W[tri], noise, sm)(Xa[va].tolist())) for tri, va in cv]
    pred = fit(model, X, y, W, noise, sm)
    yte, yrl = to_binary(te.label3), to_binary(rl.label)
    pt, pn, pr = pred(te.text_model.astype(str).tolist()), pred(te_noisy), pred(rl.text_model.astype(str).tolist())
    L = by_language(te.language, yte, pt).set_index("lang")
    r = {"model": model, "setting": name, "train_f1": f1_score(y, pred(X)), "cv_f1": np.mean(cvf), "test_f1": f1_score(yte, pt),
         "test_fpr": fpr(yte, pt), "noisy_test_f1": f1_score(yte, pn), "real_f1": f1_score(yrl, pr),
         "SI_f1": L.loc["SI", "f1"], "TA_f1": L.loc["TA", "f1"], "SI_fpr": L.loc["SI", "fpr"], "TA_fpr": L.loc["TA", "fpr"]}
    r["gap_train_test"] = r["train_f1"] - r["test_f1"]; rows.append(r)
    print(f"{model:<12} {name:<17} train {100*r['train_f1']:.2f} cv {100*r['cv_f1']:.2f} test {100*r['test_f1']:.2f} "
          f"noisy {100*r['noisy_test_f1']:.2f} real {100*r['real_f1']:.2f} gap {100*r['gap_train_test']:.2f} ({time.time()-t0:.0f}s)", flush=True)
d = pd.DataFrame(rows); d.iloc[:, 2:] = (d.iloc[:, 2:] * 100).round(2)
d.to_csv(ROOT / f"results/noise_smote_{model}.csv", index=False)
parts = [pd.read_csv(p) for p in sorted((ROOT / "results").glob("noise_smote_*.csv"))]
(ROOT / "results/NOISE_SMOTE.md").write_text("# Noise augmentation + SMOTE (training only)  - values in %\n\n" + pd.concat(parts).to_markdown(index=False), encoding="utf-8")

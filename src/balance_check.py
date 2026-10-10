"""Does balancing help? Compare current / per-language re-weighting / down-sampling on the same test + real sets.
    python -m src.balance_check -> results/balance_check.csv, results/BALANCE_CHECK.md"""
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.base import clone
from src.baselines import models, SEED
from src.evaluate import metrics, by_language, to_binary

ROOT = Path(__file__).resolve().parents[1]
tr = pd.read_csv(ROOT / "data/processed/train_feat.csv"); te = pd.read_csv(ROOT / "data/processed/test_feat.csv")
rl = pd.read_csv(ROOT / "data/processed/real_test_feat.csv")
G = {"en": "EN", "si": "SI", "si-en": "SI", "ta": "TA", "ta-en": "TA"}
tr["g"] = tr.language.map(G).fillna("mixed"); y = to_binary(tr.label3)

def reweight(df, y):
    """weight so that within EVERY language, normal and scam carry equal total weight, and languages carry equal weight"""
    w = np.ones(len(df)); key = df.g.astype(str) + "|" + y.astype(str)
    for k, n in key.value_counts().items(): w[(key == k).to_numpy()] = len(df) / (key.nunique() * n)
    return w

def downsample(df, y, target=0.20):
    keep = np.ones(len(df), bool); rng = np.random.RandomState(SEED)
    for lang in ["SI", "TA"]:
        pos = np.where((df.g == lang).to_numpy() & (y == 1))[0]; neg = ((df.g == lang).to_numpy() & (y == 0)).sum()
        n_keep = int(target / (1 - target) * neg)
        if n_keep < len(pos): keep[rng.choice(pos, len(pos) - n_keep, replace=False)] = False
    return keep

if __name__ == "__main__":
    rows = []
    for name in ["SVM", "RandomForest"]:
        for strat in ["current", "A) re-weight", "B) down-sample"]:
            p = clone(models()[name]); X = tr.text_model.astype(str)
            if strat == "current": p.fit(X, y)
            elif strat.startswith("A"):
                if name == "SVM": p.set_params(clf__class_weight=None)
                else: p.set_params(clf__class_weight=None)
                p.fit(X, y, clf__sample_weight=reweight(tr, pd.Series(y)))
            else:
                k = downsample(tr, y); p.fit(X[k], y[k])
            for sname, df, lab in [("Test", te, "label3"), ("Real", rl, "label")]:
                yy = to_binary(df[lab]); pr = p.predict(df.text_model.astype(str)); m = metrics(yy, pr)
                L = by_language(df.language, yy, pr).set_index("lang")
                row = {"model": name, "strategy": strat, "set": sname, "acc": m["acc"], "f1": m["f1"], "fpr": m["fpr"], "recall": m["recall"]}
                for lg in ["EN", "SI", "TA"]:
                    if lg in L.index: row[f"{lg}_f1"] = L.loc[lg, "f1"]; row[f"{lg}_fpr"] = L.loc[lg, "fpr"]
                rows.append(row)
            print(name, strat, "done", flush=True)
    d = pd.DataFrame(rows); num = d.columns[3:]; d[num] = (d[num] * 100).round(2)
    d.to_csv(ROOT / "results/balance_check.csv", index=False)
    md = "# Does balancing help?  (values in %)\n\n" + d.to_markdown(index=False)
    (ROOT / "results/BALANCE_CHECK.md").write_text(md, encoding="utf-8"); print(md)

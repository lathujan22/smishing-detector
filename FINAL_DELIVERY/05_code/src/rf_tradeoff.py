"""Random Forest: (A) what a shallow forest does, (B) cross-validated training accuracy,
(C) decision threshold chosen on training CV to lower false positives.
    python -m src.rf_tradeoff -> results/rf_tradeoff_depth.csv, results/rf_threshold.csv, results/RF_TRADEOFF.md,
                                 results/figures/fig_rf_tradeoff.png
The threshold is picked ONLY from out-of-fold training predictions (test / real SMS never used for choosing)."""
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold

from src.augment import noisy_copies, smote
from src.baselines import SEED, cv_groups
from src.evaluate import metrics, to_binary
from src.final_baselines import tfidf_mindf3

ROOT = Path(__file__).resolve().parents[1]
PROC, RES = ROOT / "data/processed", ROOT / "results"
P = lambda m: {k: round(m[k] * 100, 2) for k in ["acc", "precision", "recall", "f1", "fpr"]}


def fit(X, y, depth=None, leaf=1):
    Xa, ya, _, _ = noisy_copies(list(X), y, seed=SEED)
    v = tfidf_mindf3(8000); M, yy = smote(v.fit_transform(Xa), ya, seed=SEED)
    clf = RandomForestClassifier(200, max_depth=depth, min_samples_leaf=leaf, max_features="sqrt",
                                 n_jobs=-1, random_state=SEED).fit(M, yy)
    return lambda T: clf.predict_proba(v.transform(list(T)))[:, 1]


if __name__ == "__main__":
    tr = pd.read_csv(PROC / "train_feat.csv"); te = pd.read_csv(PROC / "test_feat.csv")
    rl = pd.read_csv(PROC / "real_test_feat.csv")
    Xtr, ytr = tr.text_model.astype(str), to_binary(tr.label3)
    sets = {"test": (te.text_model.astype(str), to_binary(te.label3)), "real": (rl.text_model.astype(str), to_binary(rl.label))}

    # (A) shallow forests
    rows = []
    for depth, leaf in [(None, 1), (20, 5), (8, 10), (5, 20), (3, 50), (2, 100)]:
        f = fit(Xtr, ytr, depth, leaf)
        r = {"setting": f"depth={depth or 'full'}, leaf={leaf}"}
        for s, (X, y) in [("train", (Xtr, ytr)), *sets.items()]:
            m = P(metrics(y, (f(X) >= 0.5).astype(int)))
            r |= {f"{s}_acc": m["acc"], f"{s}_f1": m["f1"], f"{s}_fpr": m["fpr"]}
        rows.append(r); print(r, flush=True)
    dep = pd.DataFrame(rows); dep.to_csv(RES / "rf_tradeoff_depth.csv", index=False)

    # (B) out-of-fold (cross-validated) training predictions with the final RF setting
    oof = np.zeros(len(tr)); cv = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
    for k, (a, b) in enumerate(cv.split(Xtr, ytr, cv_groups(tr))):
        oof[b] = fit(Xtr.iloc[a], ytr[a])(Xtr.iloc[b]); print("fold", k + 1, flush=True)

    # (C) thresholds: 0.5 (default) and the lowest threshold with CV-FPR <= 1% and <= 0.5%
    grid = np.round(np.arange(0.30, 0.91, 0.01), 2)
    fpr_at = {t: metrics(ytr, (oof >= t).astype(int))["fpr"] for t in grid}
    pick = {"default 0.50": 0.5,
            "CV-FPR <= 1%": min(t for t in grid if fpr_at[t] <= 0.01),
            "CV-FPR <= 0.5%": min(t for t in grid if fpr_at[t] <= 0.005)}
    final = fit(Xtr, ytr)
    probs = {s: final(X) for s, (X, _) in sets.items()}
    th = []
    for name, t in pick.items():
        r = {"threshold": name, "t": t, **{f"cv_{k}": v for k, v in P(metrics(ytr, (oof >= t).astype(int))).items()}}
        for s, (_, y) in sets.items():
            r |= {f"{s}_{k}": v for k, v in P(metrics(y, (probs[s] >= t).astype(int))).items()}
        th.append(r); print(r, flush=True)
    th = pd.DataFrame(th); th.to_csv(RES / "rf_threshold.csv", index=False)
    joblib.dump({"threshold": float(pick["CV-FPR <= 1%"])}, ROOT / "models/final/RandomForest_threshold.joblib")

    md = ["# Random Forest - capacity and threshold trade-off (%)", "", "## A. Shallow forests (threshold 0.5)", "",
          "| Setting | Train acc | Test acc | Test F1 | Test FPR | Real acc | Real F1 | Real FPR |", "|---|---|---|---|---|---|---|---|"]
    md += [f"| {r.setting} | {r.train_acc} | {r.test_acc} | {r.test_f1} | {r.test_fpr} | {r.real_acc} | {r.real_f1} | {r.real_fpr} |" for r in dep.itertuples()]
    md += ["", "## B + C. Cross-validated training accuracy and threshold tuning (final RF)", "",
           "| Threshold | t | CV acc | CV F1 | CV FPR | Test acc | Test F1 | Test FPR | Real acc | Real F1 | Real FPR |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    md += [f"| {r.threshold} | {r.t} | {r.cv_acc} | {r.cv_f1} | {r.cv_fpr} | {r.test_acc} | {r.test_f1} | {r.test_fpr} | {r.real_acc} | {r.real_f1} | {r.real_fpr} |" for r in th.itertuples()]
    (RES / "RF_TRADEOFF.md").write_text("\n".join(md), encoding="utf-8")

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    x = np.arange(len(dep))
    for c, lab, col in [("train_acc", "Train accuracy", "#9ecae1"), ("test_acc", "Test accuracy", "#2171b5"), ("test_f1", "Test F1", "#08306b")]:
        ax[0].plot(x, dep[c], "o-", label=lab, color=col)
    ax[0].set_xticks(x); ax[0].set_xticklabels(dep.setting, rotation=20, ha="right", fontsize=8)
    ax[0].set_ylabel("%"); ax[0].set_title("A. Shallower forest: train drops, test drops too"); ax[0].legend(frameon=False)
    ts = np.array(grid); ax[1].plot(ts, [fpr_at[t] * 100 for t in ts], color="#cb181d", label="CV false-positive rate")
    ax[1].plot(ts, [metrics(ytr, (oof >= t).astype(int))["recall"] * 100 for t in ts], color="#08306b", label="CV recall")
    ax[1].axvline(pick["CV-FPR <= 1%"], ls="--", color="grey"); ax[1].set_xlabel("decision threshold")
    ax[1].set_ylabel("%"); ax[1].set_title("C. Higher threshold: fewer false alarms"); ax[1].legend(frameon=False)
    for a in ax: a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(RES / "figures/fig_rf_tradeoff.png", dpi=300)
    print("\n".join(md))

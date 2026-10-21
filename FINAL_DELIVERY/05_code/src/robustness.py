"""Robustness evaluation - how the final models behave under harder, more realistic conditions.
    python -m src.robustness  -> results/robustness.csv, results/ROBUSTNESS.md, results/figures/fig_robustness.png

Scenarios
  1 standard        final models on the held-out test set (reference)
  2 public_only     test rows from the public datasets only (UCI + Mendeley, no generated templates)
  3 noisy_test      test messages with heavy typos / look-alike letters / dropped words (filter evasion)
  4 real_sms        the 108 real SMS collected from people (never used for training or tuning)
  5 leave_si_out    retrain WITHOUT any Sinhala (si, si-en), test on the Sinhala test rows
  6 leave_ta_out    retrain WITHOUT any Tamil (ta, ta-en), test on the Tamil test rows
  7 cross_source    retrain WITHOUT the multilingual templates, test on the multilingual test rows
Scenarios 5-7 retrain with exactly the final settings (noise + SMOTE) - only the training data changes.
"""
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline

from src.augment import add_noise, noisy_copies, smote
from src.baselines import SEED, tfidf
from src.evaluate import metrics, to_binary
from src.final_baselines import CONF, tfidf_mindf3

ROOT = Path(__file__).resolve().parents[1]
PROC, RES, MOD = ROOT / "data/processed", ROOT / "results", ROOT / "models/final"
NAMES = {"NaiveBayes": "Naive Bayes", "SVM": "SVM", "RandomForest": "Random Forest"}
SI, TA = {"si", "si-en"}, {"ta", "ta-en"}


def train(name, df):
    n, make = CONF[name]
    X, y = df.text_model.astype(str).tolist(), to_binary(df.label3)
    Xa, ya, _, _ = noisy_copies(X, y, seed=SEED)
    vec = tfidf_mindf3(n) if name == "RandomForest" else tfidf(n)
    M, yy = smote(vec.fit_transform(Xa), ya, k=5, seed=SEED)
    clf = make(); clf.fit(M, yy)
    return Pipeline([("tfidf", vec), ("clf", clf)])


def score(pipe, texts, labels):
    m = metrics(to_binary(labels), pipe.predict(pd.Series(texts).astype(str)))
    return {k: round(m[k] * 100, 2) for k in ["acc", "precision", "recall", "f1", "fpr"]} | {"n": m["n"]}


if __name__ == "__main__":
    tr = pd.read_csv(PROC / "train_feat.csv"); te = pd.read_csv(PROC / "test_feat.csv")
    rl = pd.read_csv(PROC / "real_test_feat.csv")
    rng = np.random.RandomState(SEED)
    noisy = [add_noise(t, rng, p_char=0.30, p_drop=0.10, p_swap=0.10) for t in te.text_model.astype(str)]
    pub = te[te.source.isin(["uci", "mendeley"])]
    rows = []
    for name in CONF:
        final = joblib.load(MOD / f"{name}.joblib")
        sc = {"1 Standard test": score(final, te.text_model, te.label3),
              "2 Public datasets only": score(final, pub.text_model, pub.label3),
              "3 Noisy test (evasion)": score(final, noisy, te.label3),
              "4 Real SMS": score(final, rl.text_model, rl.label)}
        for scen, keep, test in [("5 Unseen Sinhala", ~tr.language.isin(SI), te[te.language.isin(SI)]),
                                 ("6 Unseen Tamil", ~tr.language.isin(TA), te[te.language.isin(TA)]),
                                 ("7 Unseen templates", tr.source != "multilingual", te[te.source == "multilingual"])]:
            sc[scen] = score(train(name, tr[keep]), test.text_model, test.label3)
        for scen, m in sc.items():
            rows.append({"model": name, "scenario": scen, **m})
        print(name, {k: v["f1"] for k, v in sc.items()}, flush=True)

    r = pd.DataFrame(rows); r.to_csv(RES / "robustness.csv", index=False)
    piv = r.pivot(index="scenario", columns="model", values="f1")[list(CONF)]
    nrow = r.drop_duplicates("scenario").set_index("scenario")["n"]

    md = ["# Robustness evaluation (F1 %)", "",
          "| Scenario | n | Naive Bayes | SVM | Random Forest |", "|---|---|---|---|---|"]
    md += [f"| {s} | {nrow[s]} | " + " | ".join(f"{piv.loc[s, m]:.2f}" for m in CONF) + " |" for s in piv.index]
    md += ["", "Scenarios 5-7 retrain the model without that language / source (same noise + SMOTE settings).",
           "", "Full metrics (acc, precision, recall, F1, FPR): `results/robustness.csv`."]
    (RES / "ROBUSTNESS.md").write_text("\n".join(md), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(11, 5.2))
    x, w = np.arange(len(piv)), 0.26
    cols = ["#9ecae1", "#4292c6", "#08306b"]
    for i, m in enumerate(CONF):
        b = ax.bar(x + (i - 1) * w, piv[m], w, label=NAMES[m], color=cols[i])
        ax.bar_label(b, fmt="%.1f", fontsize=7.5, padding=2)
    ax.set_xticks(x); ax.set_xticklabels([s[2:] for s in piv.index], rotation=15, ha="right")
    ax.set_ylabel("F1 (%)"); ax.set_ylim(0, 105); ax.legend(loc="lower left", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Robustness of the baseline models under harder test conditions")
    fig.tight_layout(); fig.savefig(RES / "figures/fig_robustness.png", dpi=300)
    print("\n" + "\n".join(md))

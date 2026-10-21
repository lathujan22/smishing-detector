"""Bar charts for the retraining result.  python -m src.fig_retrain_bars
-> results/figures/fig_retrain_bars.png  (real SMS: before vs after, accuracy / F1 / FPR)
-> results/figures/fig_retrain_sets_bars.png  (final_v2: train / test / real SMS accuracy and F1)"""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]; FIG = ROOT / "results/figures"
r = pd.read_csv(ROOT / "results/retrain_real.csv")
M = [("NaiveBayes", "Naive Bayes"), ("SVM", "SVM"), ("RandomForest", "Random Forest")]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
BEFORE, AFTER = "#9ecae1", "#08519c"

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, (col, title) in zip(axes, [("real_acc", "Accuracy (%)"), ("real_f1", "F1 score (%)"), ("real_fpr", "False-positive rate (%)  - lower is better")]):
    x = np.arange(len(M)); w = 0.36
    b = [r[(r.model == k) & r.version.str.startswith("before")][col].item() for k, _ in M]
    a = [r[(r.model == k) & r.version.str.startswith("after")][col].item() for k, _ in M]
    for off, vals, lab, c in [(-w / 2, b, "Before", BEFORE), (w / 2, a, "After (+ real SMS)", AFTER)]:
        bars = ax.bar(x + off, vals, w, label=lab, color=c); ax.bar_label(bars, fmt="%.1f", fontsize=9, padding=2)
    ax.set_xticks(x, [n for _, n in M]); ax.set_title(title, fontsize=12); ax.set_ylim(0, 105 if "rate" not in title else 75)
    ax.spines[["top", "right"]].set_visible(False)
fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=2, frameon=False)
fig.suptitle("Real SMS (n = 163) before and after retraining with real Sri Lankan messages (5-fold out-of-fold)", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.9]); fig.savefig(FIG / "fig_retrain_bars.png", dpi=300); plt.close(fig)

# train / test / real for the retrained models
import joblib
from src.evaluate import metrics, to_binary
pr = pd.read_csv(ROOT / "results/pred_retrain_real.csv")
tr = pd.read_csv(ROOT / "data/processed/train_feat.csv"); te = pd.read_csv(ROOT / "data/processed/test_feat.csv")
vals = {}
for k, n in M:
    mdl = joblib.load(ROOT / f"models/final_v2/{k}.joblib")
    vals[n] = [metrics(to_binary(tr.label3), mdl.predict(tr.text_model.astype(str))),
               metrics(to_binary(te.label3), mdl.predict(te.text_model.astype(str))),
               metrics(to_binary(pr.label), pr[f"{k}_after"])]
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
cols = ["#c6dbef", "#4292c6", "#08306b"]; sets = ["Train", "Test", "Real SMS (CV)"]
for ax, (c, title) in zip(axes, [("acc", "Accuracy (%)"), ("f1", "F1 score (%)")]):
    x = np.arange(len(M)); w = 0.26
    for i, s in enumerate(sets):
        bars = ax.bar(x + (i - 1) * w, [vals[n][i][c] * 100 for _, n in M], w, label=s, color=cols[i])
        ax.bar_label(bars, fmt="%.1f", fontsize=8.5, padding=2)
    ax.set_xticks(x, [n for _, n in M]); ax.set_ylim(60, 104); ax.set_title(title, fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)
fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=3, frameon=False)
fig.suptitle("Retrained models (final_v2): train vs test vs real SMS", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.9]); fig.savefig(FIG / "fig_retrain_sets_bars.png", dpi=300)

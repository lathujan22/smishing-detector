"""Figure for the retrained models: before/after table + confusion matrices on real SMS (out-of-fold)
and train/test/real table of models/final_v2.   python -m src.fig_retrain -> results/figures/fig_retrain_*.png"""
from pathlib import Path
import joblib, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from src.evaluate import metrics, to_binary

ROOT = Path(__file__).resolve().parents[1]; FIG = ROOT / "results/figures"
M = [("NaiveBayes", "Naive Bayes"), ("SVM", "SVM"), ("RandomForest", "Random Forest")]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
pr = pd.read_csv(ROOT / "results/pred_retrain_real.csv"); y = to_binary(pr.label)

def table(ax, rows, header, title, bold=None):
    ax.axis("off"); ax.set_title(title, fontsize=13, pad=4)
    t = ax.table(cellText=rows, colLabels=header, loc="center", cellLoc="center", edges="horizontal")
    t.auto_set_font_size(False); t.set_fontsize(11); t.scale(1, 1.6)
    for (r, c), cell in t.get_celld().items():
        if r == 0 or (bold and r in bold): cell.set_text_props(weight="bold")

# 1) before vs after on real SMS + confusion matrices (after)
fig = plt.figure(figsize=(14, 9)); gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.15])
rows, f1s = [], []
for k, n in M:
    for tag in ["before", "after"]:
        m = metrics(y, pr[f"{k}_{tag}"])
        rows.append([n, "Before" if tag == "before" else "After (+ real SMS)"] +
                    [f"{m[c]*100:.2f}" for c in ["acc", "precision", "recall", "f1", "fpr"]])
table(fig.add_subplot(gs[0, :]), rows, ["Model", "Version", "Acc (%)", "Prec (%)", "Rec (%)", "F1 (%)", "FPR (%)"],
      f"Table: Real SMS results before and after retraining (n = {len(pr)}, 5-fold out-of-fold)", bold=[2, 4, 6])
for i, (k, n) in enumerate(M):
    ax = fig.add_subplot(gs[1, i]); cm = confusion_matrix(y, pr[f"{k}_after"], labels=[0, 1])
    ax.imshow(cm, cmap="Blues")
    for (a, b), v in np.ndenumerate(cm):
        ax.text(b, a, f"{v:,}", ha="center", va="center", fontsize=15, weight="bold", color="white" if v > cm.max() / 2 else "#1f3a5f")
    ax.set_xticks([0, 1], ["Legitimate", "Smishing/Spam"]); ax.set_yticks([0, 1], ["Legitimate", "Smishing/Spam"], rotation=90, va="center")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True" if i == 0 else "")
    ax.set_title(f"{n}  (F1 = {metrics(y, pr[f'{k}_after'])['f1']*100:.2f}%)", weight="bold", color="#1f3a5f")
fig.text(0.5, 0.01, "Figure: Confusion matrices of the retrained models on real SMS.", ha="center", style="italic")
fig.tight_layout(rect=[0, 0.03, 1, 1]); fig.savefig(FIG / "fig_retrain_real.png", dpi=300); plt.close(fig)

# 2) final_v2 models on train / test / real (real = in-sample here, shown for completeness)
sets = [("Train", pd.read_csv(ROOT / "data/processed/train_feat.csv"), "label3"),
        ("Test", pd.read_csv(ROOT / "data/processed/test_feat.csv"), "label3")]
rows = []
for k, n in M:
    mdl = joblib.load(ROOT / f"models/final_v2/{k}.joblib")
    for s, d, lab in sets:
        m = metrics(to_binary(d[lab]), mdl.predict(d.text_model.astype(str)))
        rows.append([n, s] + [f"{m[c]*100:.2f}" for c in ["acc", "precision", "recall", "f1", "fpr"]])
    m = metrics(y, pr[f"{k}_after"]); rows.append([n, "Real SMS (CV)"] + [f"{m[c]*100:.2f}" for c in ["acc", "precision", "recall", "f1", "fpr"]])
fig, ax = plt.subplots(figsize=(13, 5.2))
table(ax, rows, ["Model", "Set", "Acc (%)", "Prec (%)", "Rec (%)", "F1 (%)", "FPR (%)"], "Table: Retrained models (final_v2) on train, test and real SMS")
fig.tight_layout(); fig.savefig(FIG / "fig_retrain_sets.png", dpi=300)
print("\n".join(" | ".join(r) for r in rows))

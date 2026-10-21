"""Paper-style figures: results table + confusion matrices (test set and real SMS).
    python -m src.figures            -> uses models/final (noise + SMOTE models) if present
    python -m src.figures baselines  -> uses the old models/baselines
Output results/figures/*.png (300 dpi)"""
import sys
from pathlib import Path
import joblib, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from src.evaluate import metrics, to_binary

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "figures"; OUT.mkdir(parents=True, exist_ok=True)
_sub = sys.argv[1] if len(sys.argv) > 1 else ("final" if (ROOT / "models" / "final").exists() else "baselines")
MODEL_DIR = ROOT / "models" / _sub
MODELS = [("NaiveBayes", "Naive Bayes"), ("SVM", "SVM"), ("RandomForest", "Random Forest")]
CLASSES = ["Legitimate", "Smishing/Spam"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})


def evaluate(df, ycol):
    y = to_binary(df[ycol]); X = df["text_model"].astype(str); res = []
    for key, name in MODELS:
        m = joblib.load(MODEL_DIR / f"{key}.joblib")
        p = m.predict(X)
        s = m.decision_function(X) if hasattr(m.named_steps["clf"], "decision_function") else m.predict_proba(X)[:, 1]
        r = metrics(y, p, s); r["name"] = name; r["cm"] = confusion_matrix(y, p, labels=[0, 1]); res.append(r)
    return res


def draw_table(ax, res, title):
    ax.axis("off")
    cols = ["Model", "Acc (%)", "Prec (%)", "Rec (%)", "F1 (%)", "AUC (%)", "FPR (%)"]
    rows = [[r["name"]] + [f"{100*r[k]:.2f}" for k in ["acc", "precision", "recall", "f1", "roc_auc", "fpr"]] for r in res]
    best_f1 = max(range(len(res)), key=lambda i: res[i]["f1"])
    t = ax.table(cellText=rows, colLabels=cols, cellLoc="center", colLoc="center", loc="center",
                 colWidths=[0.26] + [0.123] * 6)
    t.auto_set_font_size(False); t.set_fontsize(12); t.scale(1, 1.55)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor("white"); cell.visible_edges = ""
        if c == 0: cell.set_text_props(ha="left"); cell._loc = "left"
        if r == 0: cell.set_text_props(weight="bold"); cell.visible_edges = "B"; cell.set_edgecolor("black")
        if r == best_f1 + 1: cell.set_text_props(weight="bold")
    ax.axhline(0.5, color="none")
    ax.set_title(title, fontsize=12, pad=4)
    # booktabs-style top / bottom rules
    ax.plot([0.02, 0.98], [0.99, 0.99], color="black", lw=1.5, transform=ax.transAxes, clip_on=False)
    ax.plot([0.02, 0.98], [0.01, 0.01], color="black", lw=1.5, transform=ax.transAxes, clip_on=False)


def draw_cm(ax, r, show_y):
    cm = r["cm"]
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max())
    for i in range(2):
        for j in range(2):
            v = cm[i, j]
            ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=13, weight="bold",
                    color="white" if v > cm.max() * 0.5 else "#1f3b5c")
    ax.set_xticks([0, 1], CLASSES); ax.set_yticks([0, 1], CLASSES, rotation=90, va="center")
    ax.set_xlabel("Predicted"); ax.set_title(f"{r['name']}  (F1 = {100*r['f1']:.2f}%)", fontsize=11, weight="bold", color="#1f3b5c")
    if show_y: ax.set_ylabel("True")
    for s in ax.spines.values(): s.set_visible(False)


def figure(res, title, caption, fname):
    fig = plt.figure(figsize=(11, 6.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[0.62, 1.35], hspace=0.18, wspace=0.35)
    draw_table(fig.add_subplot(gs[0, :]), res, title)
    for i, r in enumerate(res):
        draw_cm(fig.add_subplot(gs[1, i]), r, show_y=(i == 0))
    fig.text(0.5, 0.015, caption, ha="center", fontsize=11, style="italic")
    fig.savefig(OUT / fname, dpi=300, bbox_inches="tight", facecolor="white"); plt.close(fig)
    # confusion matrices only
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    for i, r in enumerate(res): draw_cm(axes[i], r, show_y=(i == 0))
    fig.tight_layout(); fig.savefig(OUT / fname.replace(".png", "_cm.png"), dpi=300, bbox_inches="tight", facecolor="white"); plt.close(fig)


if __name__ == "__main__":
    trn = pd.read_csv(ROOT / "data" / "processed" / "train_feat.csv")
    figure(evaluate(trn, "label3"), "Table: Baseline results on the training set (n = 9,447)",
           "Figure: Confusion matrices of the baseline models on the training set.", "fig_train_results.png")
    te = pd.read_csv(ROOT / "data" / "processed" / "test_feat.csv")
    rl = pd.read_csv(ROOT / "data" / "processed" / "real_test_feat.csv")
    figure(evaluate(te, "label3"), "Table: Baseline results on the held-out test set (n = 1,589)",
           "Figure: Confusion matrices of the baseline models on the test set.", "fig_test_results.png")
    figure(evaluate(rl, "label"), "Table: Baseline results on real Sri Lankan SMS (n = 108)",
           "Figure: Confusion matrices of the baseline models on real SMS collected via the Google Form and phones.", "fig_real_results.png")
    print(sorted(p.name for p in OUT.glob("*.png")))

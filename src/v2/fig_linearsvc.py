"""Output figure for the selected best ML model (three-class LinearSVC).
    python -m src.v2.fig_linearsvc -> results/v2/figures/fig_v2_linearsvc_output.png"""
import json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.v2.common import RES

m = pd.read_csv(RES / "three_class/metrics.csv"); m = m[m.model == "LinearSVC"].set_index("set")
L = pd.read_csv(RES / "three_class/by_language.csv"); L = L[L.model == "LinearSVC"]
NAVY, BLUE, LBLUE, ORANGE, INK = "#14284B", "#1F5FBF", "#9DB9E3", "#E8772E", "#4A5568"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
fig = plt.figure(figsize=(16, 11.5)); gs = fig.add_gridspec(3, 4, height_ratios=[0.8, 1.25, 1.0], hspace=0.5, wspace=0.35)

# (a) table
ax = fig.add_subplot(gs[0, :]); ax.axis("off")
sets = [("train (in-sample)", "Training (in-sample)"), ("cv (validation)", "Validation (grouped 5-fold CV)"),
        ("held-out test", "Held-out test (n = 1,574)"), ("real benchmark", "Real Sri Lankan SMS (n = 163)")]
cols = [("acc", "Accuracy"), ("macro_f1", "Macro-F1"), ("macro_roc_auc", "Macro\nROC-AUC"), ("macro_auprc", "Macro\nAUPRC"),
        ("scam_precision", "Scam\nprecision"), ("scam_recall", "Scam\nrecall"), ("nonscam_to_scam_fpr", "False\nalarms")]
rows = [[lab] + [f"{m.loc[s, c]:.2f}" for c, _ in cols] for s, lab in sets]
t = ax.table(cellText=rows, colLabels=["Set"] + [n for _, n in cols], loc="center", cellLoc="center", edges="horizontal",
             colWidths=[0.24] + [0.108] * 7)
t.auto_set_font_size(False); t.set_fontsize(11.5); t.scale(1, 1.75)
for (r, c), cell in t.get_celld().items():
    if r == 0: cell.set_text_props(weight="bold", color=NAVY); cell.set_height(cell.get_height() * 1.6)
    if r == 4: cell.set_text_props(weight="bold")
ax.set_title("(a) LinearSVC three-class results (%) — false alarms = normal + promotional SMS predicted as scam", loc="left", fontsize=13, color=NAVY)

# (b, c) confusion matrices
labels = ["Normal", "Promo", "Scam"]
for k, (s, title) in enumerate([("held-out test", "(b) Held-out test"), ("real benchmark", "(c) Real Sri Lankan SMS")]):
    ax = fig.add_subplot(gs[1, 2 * k:2 * k + 1 + 1 - 1] if False else gs[1, k])
    cm = np.array(json.loads(m.loc[s, "cm"])); norm = cm / cm.sum(1, keepdims=True)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}\n{norm[i, j]*100:.0f}%", ha="center", va="center", fontsize=11, weight="bold",
                color="white" if norm[i, j] > .5 else NAVY)
    ax.set_xticks(range(3), labels); ax.set_yticks(range(3), labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True" if k == 0 else ""); ax.set_title(title, loc="left", fontsize=12.5, color=NAVY)

# (d) per-class recall
ax = fig.add_subplot(gs[1, 2:]); x = np.arange(3); w = 0.38
for i, (s, lab, col) in enumerate([("held-out test", "Held-out test", LBLUE), ("real benchmark", "Real SMS", ORANGE)]):
    v = [m.loc[s, f"{c}_recall"] for c in ["normal", "promotional", "scam"]]
    b = ax.bar(x + (i - .5) * w, v, w, color=col, label=lab); ax.bar_label(b, fmt="%.1f", fontsize=10)
ax.set_xticks(x, ["Normal", "Promotional", "Scam"]); ax.set_ylim(0, 122); ax.set_ylabel("Recall (%)")
ax.set_title("(d) Per-class recall", loc="left", fontsize=12.5, color=NAVY); ax.legend(frameon=False, loc="upper right", ncol=2, fontsize=10)
ax.spines[["top", "right"]].set_visible(False)

# (e) per language
for k, (met, title) in enumerate([("scam_recall", "(e) Scam recall by language (%)"), ("nonscam_to_scam_fpr", "(f) False alarms by language (%) — lower is better")]):
    ax = fig.add_subplot(gs[2, 2 * k:2 * k + 2]); langs = ["EN", "SI", "TA", "mixed"]; x = np.arange(4)
    for i, (s, lab, col) in enumerate([("held-out test", "Held-out test", LBLUE), ("real benchmark", "Real SMS", ORANGE)]):
        sub = L[L.set == s].set_index("lang")
        v = [sub.loc[g, met] if g in sub.index and not (met == "scam_recall" and sub.loc[g, "n_scam"] == 0) else np.nan for g in langs]
        b = ax.bar(x + (i - .5) * w, v, w, color=col, label=lab)
        ax.bar_label(b, labels=["" if np.isnan(z) else f"{z:.0f}" for z in v], fontsize=9.5)
        for xi, z, g in zip(x, v, langs):
            if np.isnan(z) and g in sub.index:
                ax.text(xi + (i - .5) * w, 3, "no\nscams", ha="center", fontsize=8.5, color=INK)
    real = L[L.set == "real benchmark"].set_index("lang")
    ax.set_xticks(x, [f"{g}\nreal n={real.loc[g,'n']}" + ("" if real.loc[g, "reliable"] else "*") if g in real.index else g for g in langs], fontsize=10)
    ax.set_ylim(0, 125); ax.set_title(title, loc="left", fontsize=12.5, color=NAVY); ax.spines[["top", "right"]].set_visible(False)
    if k == 0: ax.legend(frameon=False, loc="upper right", ncol=2, fontsize=10)
fig.text(0.5, 0.015, "Selected best ML model: word 1–2 + character 2–5 TF-IDF, LinearSVC (C = 0.1), noise augmentation + SMOTE inside training folds.   "
         "SI = Sinhala + Singlish, TA = Tamil + Tanglish.   * fewer than 20 messages or 5 scams: not reliable.", ha="center", fontsize=10, color=INK)
fig.suptitle("Best ML model — three-class LinearSVC output", fontsize=16, weight="bold", color=NAVY, y=0.995)
fig.savefig(RES / "figures/fig_v2_linearsvc_output.png", dpi=300, bbox_inches="tight")
print("saved")

"""Output figure for the selected best DL model (three-class XLM-R).
    python -m src.v2.fig_xlmr -> results/v2/figures/fig_v2_xlmr_output.png"""
import json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.v2.common import RES

D = RES / "xlmr/three_lr2e-05_bs16_len128_seed42"
m = pd.read_csv(D / "metrics.csv").set_index("set"); h = pd.read_csv(D / "history.csv")
P = {"held-out test": pd.read_csv(D / "pred_held-out.csv"), "real benchmark": pd.read_csv(D / "pred_real.csv")}
NAVY, LBLUE, ORANGE, INK = "#14284B", "#9DB9E3", "#E8772E", "#4A5568"
CL = {"normal": 0, "promotional": 1, "scam": 2}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
fig = plt.figure(figsize=(16, 11.5)); gs = fig.add_gridspec(3, 4, height_ratios=[0.8, 1.25, 1.0], hspace=0.5, wspace=0.35)

# (a) table
ax = fig.add_subplot(gs[0, :]); ax.axis("off")
sets = [("validation", "Validation (family-grouped, 1/7 of train)"), ("held-out test", "Held-out test (n = 1,574)"),
        ("real benchmark", "Real Sri Lankan SMS (n = 163)")]
cols = [("acc", "Accuracy"), ("macro_f1", "Macro-F1"), ("macro_roc_auc", "Macro\nROC-AUC"), ("macro_auprc", "Macro\nAUPRC"),
        ("scam_precision", "Scam\nprecision"), ("scam_recall", "Scam\nrecall"), ("nonscam_to_scam_fpr", "False\nalarms")]
rows = [[lab] + [f"{m.loc[s, c]:.2f}" for c, _ in cols] for s, lab in sets]
t = ax.table(cellText=rows, colLabels=["Set"] + [n for _, n in cols], loc="center", cellLoc="center", edges="horizontal",
             colWidths=[0.26] + [0.105] * 7)
t.auto_set_font_size(False); t.set_fontsize(11.5); t.scale(1, 1.75)
for (r, c), cell in t.get_celld().items():
    if r == 0: cell.set_text_props(weight="bold", color=NAVY); cell.set_height(cell.get_height() * 1.6)
    if r == 3: cell.set_text_props(weight="bold")
ax.set_title("(a) XLM-R three-class results (%) — false alarms = normal + promotional SMS predicted as scam", loc="left", fontsize=13, color=NAVY)

# (b, c) confusion matrices
labels = ["Normal", "Promo", "Scam"]
for k, (s, title) in enumerate([("held-out test", "(b) Held-out test"), ("real benchmark", "(c) Real Sri Lankan SMS")]):
    ax = fig.add_subplot(gs[1, k])
    cm = np.array(json.loads(m.loc[s, "cm"])); norm = cm / cm.sum(1, keepdims=True)
    ax.imshow(norm, cmap="Oranges", vmin=0, vmax=1)
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}\n{norm[i, j]*100:.0f}%", ha="center", va="center", fontsize=11, weight="bold",
                color="white" if norm[i, j] > .5 else NAVY)
    ax.set_xticks(range(3), labels); ax.set_yticks(range(3), labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True" if k == 0 else ""); ax.set_title(title, loc="left", fontsize=12.5, color=NAVY)

# (d) training curves
ax = fig.add_subplot(gs[1, 2:]); ax2 = ax.twinx()
ax.plot(h.epoch, h.train_loss, "o-", color=NAVY, label="training loss"); ax.set_ylabel("training loss", color=NAVY)
ax2.plot(h.epoch, h.val_metric * 100, "s-", color=ORANGE, label="validation macro-F1"); ax2.set_ylabel("validation macro-F1 (%)", color=ORANGE)
best = int(h.val_metric.idxmax()); ax2.annotate(f"best checkpoint\n(epoch {h.epoch[best]}, {h.val_metric[best]*100:.1f})",
    (h.epoch[best], h.val_metric[best] * 100), xytext=(2.0, 96.5), fontsize=10, color=INK,
    arrowprops=dict(arrowstyle="->", color=INK))
ax.set_xlabel("epoch"); ax.set_xticks(h.epoch); ax2.set_ylim(75, 100)
ax.set_title("(d) Training curves (early stopping on validation)", loc="left", fontsize=12.5, color=NAVY)
ax.spines[["top"]].set_visible(False); ax2.spines[["top"]].set_visible(False)

# (e, f) per language, from per-message predictions
w = 0.38
for k, (met, title) in enumerate([("recall", "(e) Scam recall by language (%)"), ("fpr", "(f) False alarms by language (%) — lower is better")]):
    ax = fig.add_subplot(gs[2, 2 * k:2 * k + 2]); langs = ["EN", "SI", "TA", "mixed"]; x = np.arange(4)
    for i, (s, lab, col) in enumerate([("held-out test", "Held-out test", LBLUE), ("real benchmark", "Real SMS", ORANGE)]):
        d = P[s]; v, nscam = [], []
        for g in langs:
            sub = d[d.lang_family == g]; y = sub.class3 == "scam"; p = sub.pred == 2
            nscam.append(int(y.sum()))
            if len(sub) == 0 or (met == "recall" and y.sum() == 0): v.append(np.nan)
            else: v.append((p[y].mean() if met == "recall" else p[~y].mean()) * 100)
        b = ax.bar(x + (i - .5) * w, v, w, color=col, label=lab)
        ax.bar_label(b, labels=["" if np.isnan(z) else f"{z:.0f}" for z in v], fontsize=9.5)
        for xi, z, g in zip(x, v, langs):
            if np.isnan(z) and (d.lang_family == g).any():
                ax.text(xi + (i - .5) * w, 3, "no\nscams", ha="center", fontsize=8.5, color=INK)
    real = P["real benchmark"]
    lab_ = []
    for g in langs:
        n = int((real.lang_family == g).sum()); ns = int(((real.lang_family == g) & (real.class3 == "scam")).sum())
        lab_.append(f"{g}\nreal n={n}" + ("*" if n < 20 or ns < 5 else ""))
    ax.set_xticks(x, lab_, fontsize=10); ax.set_ylim(0, 125)
    ax.set_title(title, loc="left", fontsize=12.5, color=NAVY); ax.spines[["top", "right"]].set_visible(False)
    if k == 0: ax.legend(frameon=False, loc="upper right", ncol=2, fontsize=10)
fig.text(0.5, 0.015, "Selected best DL model: xlm-roberta-base fine-tuned, lr 2e-5, effective batch 16, max 128 tokens, class-weighted loss, RTX 4050 GPU.   "
         "SI = Sinhala + Singlish, TA = Tamil + Tanglish.   * fewer than 20 messages or 5 scams: not reliable.", ha="center", fontsize=10, color=INK)
fig.suptitle("Best DL model — three-class XLM-R output", fontsize=16, weight="bold", color=NAVY, y=0.995)
fig.savefig(RES / "figures/fig_v2_xlmr_output.png", dpi=300, bbox_inches="tight")
print("saved")

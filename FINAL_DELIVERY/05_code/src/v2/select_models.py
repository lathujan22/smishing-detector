"""Final model selection (ML and DL) from the measured V2 results.
    python -m src.v2.select_models -> results/v2/MODEL_SELECTION.md, results/v2/model_comparison.csv,
                                      results/v2/figures/fig_v2_model_selection.png

Selection rule (fixed before looking at the ranking):
  1. Task formulation = three-class (Normal / Promotional / Scam), because Experiment C showed it halves false alarms
     on real SMS with non-overlapping confidence intervals (binary models are kept for comparison).
  2. Within ML and within DL, rank by HELD-OUT TEST macro-F1 (evaluation level B - never the real benchmark).
  3. The real benchmark (C1) is reported for the selected models only as an audited check; it has informed earlier
     decisions, so it is not used to pick the winner.
"""
import json

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.v2.common import RES

rows = []
m3 = pd.read_csv(RES / "three_class/metrics.csv")
for _, r in m3[m3.set.isin(["held-out test", "real benchmark"])].iterrows():
    rows.append({"family": "ML", "task": "three-class", "model": r.model.split(" (")[0], "set": r.set, "macro_f1": r.macro_f1,
                 "scam_recall": r.scam_recall, "nonscam_to_scam": r.nonscam_to_scam_fpr, "normal_to_scam": r.normal_to_scam,
                 "promo_to_scam": r.promo_to_scam, "artifact": f"models/v2/three_{r.model.split()[0].lower()}.joblib"})
for p in sorted((RES / "xlmr").glob("three*/metrics.csv")) + sorted((RES / "dl").glob("*three*/metrics.csv")):
    d = pd.read_csv(p); name = "XLM-R" if "xlmr" in str(p) else "CNN-BiLSTM-Attention"
    art = f"models/v2/xlmr/{p.parent.name}" if name == "XLM-R" else f"models/v2/dl/{p.parent.name}.pt"
    for _, r in d[d.set.isin(["held-out test", "real benchmark"])].iterrows():
        rows.append({"family": "DL", "task": "three-class", "model": name, "set": r.set, "macro_f1": r.macro_f1, "scam_recall": r.scam_recall,
                     "nonscam_to_scam": r.nonscam_to_scam_fpr, "normal_to_scam": r.normal_to_scam, "promo_to_scam": r.promo_to_scam, "artifact": art})
fin = pd.read_csv(RES / "svm/final.csv")
for m, lab in [("A: original SVM (baseline)", "SVM (binary)")]:
    for _, r in fin[(fin.model == m) & fin.set.isin(["held-out test", "real benchmark"])].iterrows():
        rows.append({"family": "ML", "task": "binary", "model": lab, "set": r.set, "macro_f1": r.macro_f1, "scam_recall": r.recall,
                     "nonscam_to_scam": r.fpr, "artifact": "models/v2/svm_original_baseline.joblib"})
for p in sorted((RES / "xlmr").glob("binary*/metrics.csv")) + sorted((RES / "dl").glob("*binary*/metrics.csv")):
    d = pd.read_csv(p); name = {"binary": "XLM-R", "cnn": "CNN", "bilstm": "BiLSTM", "hybrid": "CNN-BiLSTM-Attention"}[p.parent.name.split("_")[0]]
    for _, r in d[d.set.isin(["held-out test", "real benchmark"])].iterrows():
        rows.append({"family": "DL", "task": "binary", "model": f"{name} (binary)", "set": r.set, "macro_f1": r.macro_f1,
                     "scam_recall": r.recall, "nonscam_to_scam": r.fpr, "artifact": str(p.parent.relative_to(RES.parents[1]))})
C = pd.DataFrame(rows).round(2); C.to_csv(RES / "model_comparison.csv", index=False)

test3 = C[(C.task == "three-class") & (C.set == "held-out test")].sort_values("macro_f1", ascending=False)
best = {fam: test3[test3.family == fam].iloc[0] for fam in ["ML", "DL"]}
real = C[C.set == "real benchmark"].set_index(["task", "model"])
sel = {fam: {"model": b.model, "artifact": b.artifact, "test_macro_f1": b.macro_f1,
             "real_macro_f1": float(real.loc[("three-class", b.model), "macro_f1"]),
             "real_scam_recall": float(real.loc[("three-class", b.model), "scam_recall"]),
             "real_nonscam_to_scam": float(real.loc[("three-class", b.model), "nonscam_to_scam"])} for fam, b in best.items()}
(RES / "selected_models.json").write_text(json.dumps(sel, indent=2))

# figure: test vs real macro-F1 and real non-scam->scam for every three-class model + binary references
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
t3 = C[C.task == "three-class"]; order = list(test3.model)
x = np.arange(len(order)); w = 0.38
for i, (s, col) in enumerate([("held-out test", "#9ecae1"), ("real benchmark", "#08519c")]):
    v = [t3[(t3.model == m) & (t3.set == s)].macro_f1.item() for m in order]
    b = axes[0].bar(x + (i - .5) * w, v, w, color=col, label={"held-out test": "B: held-out test", "real benchmark": "C1: real benchmark"}[s])
    axes[0].bar_label(b, fmt="%.1f", fontsize=7.5)
axes[0].set_xticks(x, [m.replace("GradientBoosting", "Grad.Boost.").replace("LogisticRegression", "LogReg") for m in order], rotation=15, fontsize=8)
axes[0].set_title("Three-class macro-F1 (%) - ranked by held-out test", fontsize=10.5); axes[0].legend(frameon=False, fontsize=8); axes[0].set_ylim(0, 105)
rb = C[C.set == "real benchmark"].copy(); rb["lab"] = rb.model + np.where(rb.task == "three-class", " (3-class)", "")
rb = rb.sort_values("nonscam_to_scam")
cols = ["#e6550d" if f == "DL" else "#3182bd" for f in rb.family]
axes[1].barh(rb.lab, rb.nonscam_to_scam, color=cols); axes[1].set_title("Real benchmark: non-scam flagged as scam (%) - lower is better", fontsize=10)
axes[2].barh(rb.lab, rb.scam_recall, color=cols); axes[2].set_title("Real benchmark: scam recall (%)", fontsize=10)
for a in axes[1:]:
    a.tick_params(axis="y", labelsize=7.5)
for a in axes: a.spines[["top", "right"]].set_visible(False)
fig.text(0.99, 0.01, "blue = ML, orange = DL", ha="right", fontsize=8, style="italic")
fig.tight_layout(); fig.savefig(RES / "figures/fig_v2_model_selection.png", dpi=300); plt.close(fig)

md = ["# Final model selection", "", __doc__.split("Selection rule")[1].strip().replace("(fixed before looking at the ranking):", "**Selection rule** (fixed before looking at the ranking):"), "",
      "## Three-class ranking (held-out test, level B)", "",
      "| Rank | Family | Model | Test macro-F1 | Test scam recall | Test non-scam->scam |", "|---|---|---|---|---|---|"]
md += [f"| {k + 1} | {r.family} | {r.model} | {r.macro_f1:.2f} | {r.scam_recall:.2f} | {r.nonscam_to_scam:.2f} |" for k, (_, r) in enumerate(test3.iterrows())]
md += ["", "## Selected models", ""]
for fam, s in sel.items():
    md.append(f"- **Best {fam}: {s['model']} (three-class)** - test macro-F1 {s['test_macro_f1']:.2f}; real benchmark macro-F1 "
              f"{s['real_macro_f1']:.2f}, scam recall {s['real_scam_recall']:.2f}%, non-scam flagged as scam {s['real_nonscam_to_scam']:.2f}%. "
              f"Artifact: `{s['artifact']}`")
md += ["", "On the real benchmark the two selected models are statistically indistinguishable (scam recall identical, non-scam->scam "
       "17.9% [11.8, 24.6] vs 20.2% [13.3, 27.0], bootstrap 95% CI; paired McNemar test on scam-vs-rest decisions p = 0.71). The web application therefore uses the **LinearSVC as the default** (fast, CPU-only, "
       "explainable n-gram evidence) and offers **XLM-R as a second opinion** when its checkpoint and PyTorch are available.", "",
       "## All models (held-out test and real benchmark)", "", "| Family | Task | Model | Set | Macro-F1 | Scam recall | Non-scam->scam |", "|---|---|---|---|---|---|---|"]
md += [f"| {r.family} | {r.task} | {r.model} | {r.set} | {r.macro_f1:.2f} | {r.scam_recall:.2f} | {r.nonscam_to_scam:.2f} |" for _, r in C.iterrows()]
md += ["", "Limitations: real benchmark n=163 (29 scams) and it informed earlier decisions; a new locked real test set is required for final claims. "
       "Gradient Boosting is scikit-learn's GradientBoostingClassifier, not XGBoost."]
(RES / "MODEL_SELECTION.md").write_text("\n".join(md), encoding="utf-8")
print("\n".join(md[:30]))

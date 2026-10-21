"""V2 figures + results/v2/REPORT.md (run after registry/split, svm_experiments, three_class; XLM-R optional).
    python -m src.v2.report"""
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve

from src.v2.common import RES, ROOT

FIG = RES / "figures"; FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
BLUE, ORANGE, GREY = "#08519c", "#e6550d", "#9e9e9e"


def ci(s):
    if not isinstance(s, str) or not s.startswith("["):
        return (np.nan, np.nan)
    a, b = s.strip("[]").split(","); return float(a), float(b)


def md_table(df, cols=None, names=None, fmt="{:.2f}"):
    cols = cols or list(df.columns); names = names or cols
    out = ["| " + " | ".join(names) + " |", "|" + "---|" * len(cols)]
    for _, r in df[cols].iterrows():
        out.append("| " + " | ".join(fmt.format(v) if isinstance(v, (float, np.floating)) and not pd.isna(v) else ("–" if pd.isna(v) else str(v)) for v in r) + " |")
    return out


# ---------------------------------------------------------------- load
r1 = pd.read_csv(RES / "svm/round1_cv.csv"); gr = pd.read_csv(RES / "svm/greedy.csv")
fin = pd.read_csv(RES / "svm/final.csv"); fl = pd.read_csv(RES / "svm/final_by_language.csv")
m3 = pd.read_csv(RES / "three_class/metrics.csv"); l3 = pd.read_csv(RES / "three_class/by_language.csv")
exc = pd.read_csv(RES / "three_class/exp_c.csv")
audit = (RES / "data/data_audit.md").read_text(); split = (RES / "data/split_report.md").read_text()
imp = json.loads((RES / "svm/improved_cfg.json").read_text())
xl = sorted((RES / "xlmr").glob("*/metrics.csv")) + sorted((RES / "dl").glob("*/metrics.csv"))

# ---------------------------------------------------------------- fig 1: one-factor ablation (CV)
d = r1[r1.factor != "baseline"].sort_values("d_auprc")
fig, ax = plt.subplots(1, 2, figsize=(14, 6.5), sharey=True)
for a, col, t in [(ax[0], "d_auprc", "Change in CV AUPRC vs original SVM (points)"), (ax[1], "d_f1", "Change in CV F1 vs original SVM (points)")]:
    a.barh(d.config, d[col], color=[BLUE if v > 0 else GREY for v in d[col]]); a.axvline(0, color="black", lw=.8)
    a.set_title(t, fontsize=10.5); a.spines[["top", "right"]].set_visible(False)
    for y_, v in enumerate(d[col]): a.text(v + (0.1 if v >= 0 else -0.1), y_, f"{v:+.2f}", va="center", ha="left" if v >= 0 else "right", fontsize=8)
fig.suptitle(f"Experiment 3: one factor at a time, grouped 5-fold CV (baseline AUPRC {r1.iloc[0].auprc:.2f}, F1 {r1.iloc[0].f1:.2f})", fontsize=12)
fig.tight_layout(); fig.savefig(FIG / "fig_v2_svm_ablation.png", dpi=300); plt.close(fig)

# ---------------------------------------------------------------- fig 2: three evaluation levels with CIs
models = ["A: original SVM (baseline)", "A: improved SVM (CV-selected)", "E: word-only TF-IDF", "F: without contextual features"]
labels = ["Original SVM", "Improved SVM\n(CV-selected)", "Word-only\nTF-IDF", "Improved SVM\nwithout context feats"]
sets = ["cv (validation)", "held-out test", "real benchmark"]
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
for a, (met, t) in zip(axes, [("f1", "Scam F1 (%)"), ("fpr", "False-positive rate (%) - lower is better"), ("auprc", "AUPRC (%)")]):
    x = np.arange(len(models)); w = 0.26
    for i, (s, c) in enumerate(zip(sets, ["#c6dbef", "#4292c6", BLUE])):
        v = [fin[(fin.model == m) & (fin.set == s)][met].item() for m in models]
        err = None
        if met + "_ci" in fin and s != "cv (validation)":
            lo_hi = [ci(fin[(fin.model == m) & (fin.set == s)][met + "_ci"].item()) for m in models]
            err = np.array([[vv - lo for vv, (lo, hi) in zip(v, lo_hi)], [hi - vv for vv, (lo, hi) in zip(v, lo_hi)]])
        b = a.bar(x + (i - 1) * w, v, w, color=c, label={"cv (validation)": "A: CV", "held-out test": "B: held-out test", "real benchmark": "C1: real benchmark (n=163)"}[s],
                  yerr=err, capsize=2, error_kw={"lw": .8})
        a.bar_label(b, fmt="%.0f", fontsize=7, padding=6 if err is not None else 2)
    a.set_xticks(x, labels, fontsize=8.5); a.set_title(t, fontsize=10.5); a.spines[["top", "right"]].set_visible(False)
    a.set_ylim(0, 110)
fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=3, frameon=False, fontsize=9)
fig.suptitle("Experiments A / E / F across the three evaluation levels (error bars = bootstrap 95% CI)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.88]); fig.savefig(FIG / "fig_v2_levels.png", dpi=300); plt.close(fig)

# ---------------------------------------------------------------- fig 3: ROC / PR on test and real
pt, pr_ = pd.read_csv(RES / "svm/pred_test.csv"), pd.read_csv(RES / "svm/pred_real.csv")
fig, axes = plt.subplots(2, 2, figsize=(12, 9.5))
for r, (nm, d_) in enumerate([("Held-out test", pt), ("Real benchmark (n=163)", pr_)]):
    y = (d_.class3 == "scam").astype(int)
    for m, lab, c in zip(models, ["Original SVM", "Improved SVM", "Word-only", "Improved w/o context"], [BLUE, ORANGE, GREY, "#31a354"]):
        s = d_[f"{m}|score"]; f_, t_, _ = roc_curve(y, s); p_, rc_, _ = precision_recall_curve(y, s)
        row = fin[(fin.model == m) & (fin.set == ("held-out test" if r == 0 else "real benchmark"))].iloc[0]
        axes[r, 0].plot(f_ * 100, t_ * 100, color=c, label=f"{lab} (AUC {row.roc_auc:.1f})")
        axes[r, 1].plot(rc_ * 100, p_ * 100, color=c, label=f"{lab} (AUPRC {row.auprc:.1f})")
    axes[r, 0].plot([0, 100], [0, 100], ":", color="grey"); axes[r, 1].axhline(y.mean() * 100, ls=":", color="grey")
    axes[r, 0].set(title=f"ROC - {nm}", xlabel="False-positive rate (%)", ylabel="Scam recall (%)")
    axes[r, 1].set(title=f"Precision-recall - {nm}", xlabel="Scam recall (%)", ylabel="Precision (%)")
    for a in axes[r]: a.legend(frameon=False, fontsize=8); a.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(FIG / "fig_v2_roc_pr.png", dpi=300); plt.close(fig)

# ---------------------------------------------------------------- fig 4: three-class confusion matrices (real) + exp C
order = list(m3.model.unique())
fig, axes = plt.subplots(1, len(order), figsize=(4.2 * len(order), 4.1))
for a, m in zip(axes, order):
    row = m3[(m3.model == m) & (m3.set == "real benchmark")].iloc[0]; cm = np.array(json.loads(row.cm))
    a.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        a.text(j, i, v, ha="center", va="center", fontsize=12, weight="bold", color="white" if v > cm.max() / 2 else "#1f3a5f")
    a.set_xticks(range(3), ["Normal", "Promo", "Scam"]); a.set_yticks(range(3), ["Normal", "Promo", "Scam"])
    a.set_xlabel("Predicted"); a.set_title(f"{m.split(' (')[0]}\nmacro F1 {row.macro_f1:.1f}%", fontsize=10, weight="bold", color="#1f3a5f")
axes[0].set_ylabel("True"); fig.suptitle("Three-class models on the real benchmark (n=163)", fontsize=12)
fig.tight_layout(); fig.savefig(FIG / "fig_v2_three_class_cm_real.png", dpi=300); plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
for a, (met, t) in zip(axes, [("scam_recall", "Scam recall (%)"), ("nonscam_to_scam_fpr", "Non-scam flagged as scam (%) - lower is better")]):
    for k, s in enumerate(["held-out test", "real benchmark"]):
        e = exc[exc.set == s]; x = np.arange(len(e)) + k * (len(e) + 1)
        civ = [ci(v) for v in e[("recall" if met == "scam_recall" else "fpr") + "_ci"]]
        err = np.array([[v - lo for v, (lo, hi) in zip(e[met], civ)], [hi - v for v, (lo, hi) in zip(e[met], civ)]])
        b = a.bar(x, e[met], color=[GREY, "#4292c6", BLUE], yerr=err, capsize=3)
        a.bar_label(b, fmt="%.1f", fontsize=8, padding=6)
        short = {"binary SVM (original recipe)": "Binary\noriginal", "binary SVM (improved)": "Binary\nimproved", "three-class LinearSVC": "Three-\nclass"}
        for xx, lab in zip(x, e.model): a.text(xx, -4, short.get(lab, lab), ha="center", va="top", fontsize=8)
        a.text(x.mean(), 108, s, ha="center", fontsize=9, weight="bold")
    a.set_xticks([]); a.set_ylim(0, 115); a.set_title(t, fontsize=10.5); a.spines[["top", "right"]].set_visible(False)
fig.suptitle("Experiment C: binary vs three-class SVM (scam-vs-rest view, 95% CI)", fontsize=12)
fig.tight_layout(rect=[0, 0.08, 1, 1]); fig.savefig(FIG / "fig_v2_exp_c.png", dpi=300); plt.close(fig)

# ---------------------------------------------------------------- fig 5: per-language (Experiment G)
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
for r, s in enumerate(["held-out test", "real benchmark"]):
    for c, (met, t) in enumerate([("recall", "Scam recall (%)"), ("fpr", "False-positive rate (%)")]):
        a = axes[r, c]; langs = ["EN", "SI", "TA", "mixed"]; x = np.arange(4); w = 0.38
        for i, (m, lab, col) in enumerate([(models[0], "Original SVM", "#4292c6"), (models[1], "Improved SVM", ORANGE)]):
            sub = fl[(fl.model == m) & (fl.set == s)].set_index("lang")
            v = [sub.loc[g, met] * 100 if g in sub.index else np.nan for g in langs]
            b = a.bar(x + (i - .5) * w, v, w, color=col, label=lab); a.bar_label(b, fmt="%.0f", fontsize=7.5, padding=1)
        sub = fl[(fl.model == models[0]) & (fl.set == s)].set_index("lang")
        a.set_xticks(x, [f"{g}\nn={sub.loc[g, 'n']}, scams={sub.loc[g, 'n_pos']}" + ("" if sub.loc[g, "reliable"] else "\n(unreliable)") if g in sub.index else g for g in langs], fontsize=8)
        a.set_ylim(0, 112); a.set_title(f"{t} - {s}", fontsize=10); a.spines[["top", "right"]].set_visible(False)
axes[0, 0].legend(frameon=False, fontsize=8.5, loc="lower left")
fig.suptitle("Experiment G: per-language performance (SI = Sinhala + Singlish, TA = Tamil + Tanglish)", fontsize=12)
fig.tight_layout(); fig.savefig(FIG / "fig_v2_by_language.png", dpi=300); plt.close(fig)

# ---------------------------------------------------------------- fig 6: deep-model training / validation curves
hists = sorted((RES / "xlmr").glob("*/history.csv")) + sorted((RES / "dl").glob("*/history.csv"))
if hists:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    cols = plt.cm.tab10.colors
    for k, h_ in enumerate(hists):
        h = pd.read_csv(h_); lab = h_.parent.name
        axes[0].plot(h.epoch, h.train_loss, "o-", color=cols[k % 10], label=lab, ms=3)
        axes[1].plot(h.epoch, h.val_metric * 100, "o-", color=cols[k % 10], label=lab, ms=3)
    axes[0].set(title="Training loss (class-weighted cross-entropy)", xlabel="epoch", ylabel="loss")
    axes[1].set(title="Validation metric (AUPRC binary / macro-F1 three-class, %)", xlabel="epoch", ylabel="%")
    for a in axes: a.spines[["top", "right"]].set_visible(False)
    axes[1].legend(frameon=False, fontsize=7.5, loc="lower right")
    fig.tight_layout(); fig.savefig(FIG / "fig_v2_dl_curves.png", dpi=300); plt.close(fig)

# ---------------------------------------------------------------- fig 7: Experiment D comparison (binary, scam vs rest)
cmp_rows = []
for p_ in sorted((RES / "dl").glob("*binary*/metrics.csv")) + sorted((RES / "xlmr").glob("binary*/metrics.csv")):
    d_ = pd.read_csv(p_); d_["label"] = {"cnn": "CNN", "bilstm": "BiLSTM", "hybrid": "CNN-BiLSTM-Att", "binary": "XLM-R"}[p_.parent.name.split("_")[0]]
    cmp_rows.append(d_)
if cmp_rows:
    svm_b = fin[fin.model == "A: original SVM (baseline)"].assign(label="SVM (TF-IDF)")
    C = pd.concat(cmp_rows + [svm_b]); order_ = ["SVM (TF-IDF)", "CNN", "BiLSTM", "CNN-BiLSTM-Att", "XLM-R"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for a, (met, t) in zip(axes, [("f1", "Scam F1 (%)"), ("fpr", "False-positive rate (%) - lower is better"), ("auprc", "AUPRC (%)")]):
        x = np.arange(len(order_)); w = 0.38
        for i, (s_, col) in enumerate([("held-out test", "#9ecae1"), ("real benchmark", BLUE)]):
            v = [C[(C.label == m) & (C.set == s_)][met].iloc[0] for m in order_]
            civ = [ci(C[(C.label == m) & (C.set == s_)][met + "_ci"].iloc[0]) if met + "_ci" in C else (np.nan, np.nan) for m in order_]
            err = np.array([[vv - lo for vv, (lo, hi) in zip(v, civ)], [hi - vv for vv, (lo, hi) in zip(v, civ)]])
            b = a.bar(x + (i - .5) * w, v, w, color=col, label="B: held-out test" if i == 0 else "C1: real benchmark (n=163)",
                      yerr=None if np.isnan(err).all() else err, capsize=2, error_kw={"lw": .8})
            a.bar_label(b, fmt="%.0f", fontsize=7.5, padding=6)
        a.set_xticks(x, order_, fontsize=8.5, rotation=10); a.set_ylim(0, 112); a.set_title(t, fontsize=10.5)
        a.spines[["top", "right"]].set_visible(False)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=2, frameon=False)
    fig.suptitle("Experiment D: SVM vs deep models, binary scam detection (95% bootstrap CI)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.88]); fig.savefig(FIG / "fig_v2_exp_d.png", dpi=300); plt.close(fig)

# ---------------------------------------------------------------- REPORT.md
show = fin[fin.set.isin(["cv (validation)", "train (in-sample)", "held-out test", "real benchmark"])].copy()
md = ["# Smishing detector V2 - experiment report", "",
      "All numbers measured by the scripts in `src/v2/`; every run has a manifest (config, seed 42, data md5, library versions) "
      "in its results folder. Original experiments are preserved unchanged.", "",
      "## Data audit and split", "", audit.split("## Class by source")[0].replace("# V2 data audit", "").strip(), "",
      split.split("## Train")[0].replace("# V2 split report", "").strip(), "",
      "**Disclosure:** the 163-message real benchmark (C1) has already informed research decisions (error analysis, label "
      "definition, earlier model comparisons). It is an audited benchmark, not a pristine test. Final claims require a new "
      "locked real-world test set (C2), not yet collected.", "",
      "## Experiment 3 - SVM, one factor at a time (grouped 5-fold CV)", ""]
md += md_table(r1, ["config", "factor", "auprc", "f1", "precision", "recall", "fpr", "d_auprc", "d_f1"],
               ["Configuration", "Factor", "AUPRC", "F1", "Precision", "Recall", "FPR", "ΔAUPRC", "ΔF1"])
md += ["", "Greedy combination (CV AUPRC, kept only if it improves CV):", ""] + md_table(gr) + \
      ["", f"Selected 'improved' configuration: context features ×{imp['context_weight']}, resampling = {imp['resample']}, "
       f"C = {imp['C']}, word {tuple(imp['word']['ngram'])} + char {tuple(imp['char']['ngram'])} n-grams.", "",
      "## Experiments A, E, F - three evaluation levels", ""]
md += md_table(show, ["model", "set", "precision", "recall", "f1", "fpr", "roc_auc", "auprc", "f1_ci", "fpr_ci", "auprc_ci"],
               ["Model", "Set", "Precision", "Recall", "F1", "FPR", "ROC-AUC", "AUPRC", "F1 95% CI", "FPR 95% CI", "AUPRC 95% CI"])
md += ["", "## Experiment C - binary vs three-class SVM (scam-vs-rest view)", ""]
md += md_table(exc, ["set", "model", "scam_recall", "recall_ci", "nonscam_to_scam_fpr", "fpr_ci", "normal_to_scam", "promo_to_scam", "f1"],
               ["Set", "Model", "Scam recall", "95% CI", "Non-scam→scam", "95% CI", "Normal→scam", "Promo→scam", "Scam F1"])
md += ["", "## Three-class models (Normal / Promotional / Scam)", ""]
md += md_table(m3, ["model", "set", "acc", "macro_f1", "macro_roc_auc", "macro_auprc", "normal_recall", "promotional_recall", "scam_recall",
                    "scam_precision", "nonscam_to_scam_fpr"],
               ["Model", "Set", "Acc", "Macro F1", "Macro AUC", "Macro AUPRC", "Normal R", "Promo R", "Scam R", "Scam P", "Non-scam→scam"])
md += ["", "Gradient Boosting = scikit-learn GradientBoostingClassifier, a separate algorithm (XGBoost is not installed in this environment).", "",
       "## Experiment G - per language (real benchmark)", ""]
g = fl[fl.set == "real benchmark"].copy()
for c in ["precision", "recall", "f1", "fpr"]: g[c] = g[c] * 100
md += md_table(g, ["model", "lang", "n", "n_pos", "precision", "recall", "f1", "fpr", "reliable"],
               ["Model", "Lang", "n", "Scams", "Precision", "Recall", "F1", "FPR", "Reliable (n≥20, ≥5 scams)"])
md += ["", "## Experiment B (more legitimate promotional / transactional data)", "",
       "**Not run - blocked on data.** No new real Sri Lankan promotional / transactional messages have been collected yet. "
       "Synthetic examples will not be generated and described as real. Ingest exports with "
       "`python -m src.v2.registry ingest FILE --role dev`, then re-run the experiments with the dev pool added to training.", "",
       "## Experiment D - deep models (CNN, BiLSTM, CNN-BiLSTM-Attention, XLM-R) vs SVM", "",
       "Deep models trained on the user's laptop GPU (RTX 4050, 6 GB) with the same family-grouped protocol: a family-grouped "
       "validation split of the training data is used for early stopping, checkpoint and threshold; held-out test and the real "
       "benchmark are scored once. XLM-R: xlm-roberta-base, lr 2e-5, effective batch 16 (8 x 2 accumulation), gradient "
       "checkpointing, max_len 128, class-weighted loss. CNN / BiLSTM / Hybrid: XLM-R sub-word tokens, embeddings from scratch.", ""]
bin_rows, three_rows = [], []
for p_ in xl:
    d_ = pd.read_csv(p_); d_["run"] = p_.parent.name
    (three_rows if "macro_f1" in d_ and "nonscam_to_scam_fpr" in d_ else bin_rows).append(d_)
svm_ref = fin[fin.model.isin(["A: original SVM (baseline)"]) & fin.set.isin(["held-out test", "real benchmark"])].assign(run="svm (src/v2/svm_experiments)")
if bin_rows:
    B = pd.concat(bin_rows + [svm_ref]); B = B[B.set != "validation"]
    md += ["### Binary scam vs non-scam", ""] + md_table(B, ["model", "run", "set", "precision", "recall", "f1", "fpr", "auprc", "f1_ci", "fpr_ci"],
                                                        ["Model", "Run", "Set", "Precision", "Recall", "F1", "FPR", "AUPRC", "F1 95% CI", "FPR 95% CI"])
if three_rows:
    T = pd.concat(three_rows + [m3[m3.model == "LinearSVC"].assign(run="three-class LinearSVC")]); T = T[T.set.isin(["held-out test", "real benchmark"])]
    md += ["", "### Three-class", ""] + md_table(T, ["model", "run", "set", "macro_f1", "scam_recall", "nonscam_to_scam_fpr", "normal_to_scam", "promo_to_scam"],
                                                 ["Model", "Run", "Set", "Macro F1", "Scam recall", "Non-scam->scam", "Normal->scam", "Promo->scam"])
if not xl:
    md += ["**Pending** - no deep-learning results found in results/v2/xlmr or results/v2/dl."]
md += ["", "## Threats to validity", "",
       "- Real benchmark is small (163; 29 scams); Tamil n = 4 and mixed n = 3 -> no per-language conclusion for them.",
       "- The real benchmark informed earlier decisions (see disclosure) -> mild optimistic bias possible.",
       "- 44.7% of training data is template-generated or machine-translated; the 'human-written only' ablation shows the "
       "Sinhala / Tamil coverage depends on it.",
       "- Distribution shift between training data and real Sri Lankan SMS (links / money far more common in real legitimate SMS) "
       "dominates the real-world error; CV and held-out test do not measure it.",
       "- Hyper-parameter search is limited (one factor at a time + greedy); improvements smaller than the CI width are not claimed.",
       "- Overfitting to the training distribution is not ruled out: training (in-sample) ≫ real-benchmark performance.", "",
       "## Figures", "", "`results/v2/figures/`: fig_v2_svm_ablation, fig_v2_levels, fig_v2_roc_pr, fig_v2_exp_c, "
       "fig_v2_three_class_cm_real, fig_v2_by_language, fig_v2_exp_d, fig_v2_dl_curves."]
(RES / "REPORT.md").write_text("\n".join(md), encoding="utf-8")
print("report written")

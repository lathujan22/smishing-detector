"""Figures + REPORT.md for the scam-only (primary) and three-class experiments.
    python -m src.scam_report  (after src.scam_pipeline scam_only / three_class and src.scam_errors)
Figures -> results/scam_only/figures/, results/three_class/figures/"""
import json
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

ROOT = Path(__file__).resolve().parents[1]
R1, R3 = ROOT / "results/scam_only", ROOT / "results/three_class"
F1D, F3D = R1 / "figures", R3 / "figures"
for d in (F1D, F3D): d.mkdir(parents=True, exist_ok=True)
MODELS = ["NaiveBayes", "SVM", "RandomForest", "XGBoost"]
NICE = {"NaiveBayes": "Naive Bayes", "SVM": "SVM", "RandomForest": "Random Forest", "XGBoost": "XGBoost*"}
COL = {"NaiveBayes": "#9ecae1", "SVM": "#4292c6", "RandomForest": "#08519c", "XGBoost": "#e6550d"}
SETS = [("train (in-sample)", "Training (in-sample)"), ("cv (validation)", "Validation (grouped 5-fold CV)"),
        ("test", "Held-out test (public + generated)"), ("real", "Independent real SMS")]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
cfg1 = json.loads((R1 / "config.json").read_text())
XGB_NOTE = ("* XGBoost row = " + ("XGBClassifier" if cfg1["environment"]["xgboost_available"] else
            "sklearn GradientBoosting fallback (xgboost not installable here; same script uses XGBoost where installed)"))


def table_fig(rows, header, title, path, widths=None, note=None, bold_col0=False, h=None):
    fig, ax = plt.subplots(figsize=(14, h or 0.29 * len(rows) + 1.0)); ax.axis("off")
    t = ax.table(cellText=rows, colLabels=header, loc="upper center", cellLoc="center", edges="horizontal", colWidths=widths)
    t.auto_set_font_size(False); t.set_fontsize(9.5); t.scale(1, 1.45)
    for (r, c), cell in t.get_celld().items():
        if r == 0: cell.set_text_props(weight="bold")
    ax.set_title(title, fontsize=12, pad=8)
    if note: fig.text(0.5, 0.01, note, ha="center", fontsize=8, style="italic")
    fig.tight_layout(); fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)


def cm_grid(ys, preds, titles, labels, path, suptitle):
    fig, axes = plt.subplots(1, len(preds), figsize=(4.1 * len(preds), 4.0))
    for ax, y, p, t in zip(axes, ys, preds, titles):
        cm = confusion_matrix(y, p, labels=list(range(len(labels))))
        ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=12, weight="bold", color="white" if v > cm.max() / 2 else "#1f3a5f")
        ax.set_xticks(range(len(labels)), labels, fontsize=8.5); ax.set_yticks(range(len(labels)), labels, fontsize=8.5)
        ax.set_xlabel("Predicted"); ax.set_title(t, fontsize=10, weight="bold", color="#1f3a5f")
    axes[0].set_ylabel("True"); fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)


def f(x): return "–" if pd.isna(x) else f"{x:.2f}"


# ================================================================ scam-only
m1 = pd.read_csv(R1 / "metrics.csv"); L1 = pd.read_csv(R1 / "by_language.csv")
prim = m1[m1.threshold == "cv_f1"]
cols = ["acc", "precision", "recall", "f1", "fpr", "roc_auc", "auprc"]
hdr = ["Model", "Set", "Acc", "Precision", "Recall", "F1", "FPR", "ROC-AUC", "AUPRC"]
rows = [[NICE[m] if i == 0 else "", lab] + [f(prim[(prim.model == m) & (prim.set == s)][c].item()) for c in cols]
        for m in MODELS for i, (s, lab) in enumerate(SETS)]
table_fig(rows, hdr, "Table: Scam-only smishing detection (%) - threshold chosen on training CV", F1D / "fig_scam_results_table.png",
          widths=[.11, .25] + [.08] * 7, note=XGB_NOTE + ". Training (in-sample) is optimistic; real SMS never used for training/tuning.")

rl = pd.read_csv(R1 / "pred_real.csv"); te = pd.read_csv(R1 / "pred_test.csv")
y_rl, y_te = (rl.label == "smishing").astype(int), (te.label3 == "smishing").astype(int)
for s, d, y in [("real", rl, y_rl), ("test", te, y_te)]:
    cm_grid([y] * 4, [d[f"{m}_cv_f1_pred"] for m in MODELS],
            [f"{NICE[m]} (F1 {prim[(prim.model == m) & (prim.set == s)].f1.item():.1f}%)" for m in MODELS],
            ["Not scam", "Scam"], F1D / f"fig_scam_cm_{s}.png",
            f"Confusion matrices - {'independent real SMS (n=%d)' % len(d) if s == 'real' else 'held-out test (n=%d)' % len(d)}")

fig, axes = plt.subplots(2, 2, figsize=(12, 9.5))
for r, (s, d, y) in enumerate([("test", te, y_te), ("real", rl, y_rl)]):
    for m in MODELS:
        sc = d[f"{m}_score"]; fpr, tpr, _ = roc_curve(y, sc); pr, rc, _ = precision_recall_curve(y, sc)
        mm = prim[(prim.model == m) & (prim.set == s)]
        axes[r, 0].plot(fpr * 100, tpr * 100, color=COL[m], label=f"{NICE[m]} (AUC {mm.roc_auc.item():.1f})")
        axes[r, 1].plot(rc * 100, pr * 100, color=COL[m], label=f"{NICE[m]} (AUPRC {mm.auprc.item():.1f})")
    base = y.mean() * 100
    axes[r, 0].plot([0, 100], [0, 100], ":", color="grey"); axes[r, 1].axhline(base, ls=":", color="grey")
    nm = "Held-out test" if s == "test" else "Independent real SMS"
    axes[r, 0].set(title=f"ROC - {nm}", xlabel="False-positive rate (%)", ylabel="Scam recall (%)")
    axes[r, 1].set(title=f"Precision-recall - {nm} (chance = {base:.1f}%)", xlabel="Scam recall (%)", ylabel="Precision (%)")
    for a in axes[r]: a.legend(frameon=False, fontsize=8.5, loc="lower right" if a is axes[r, 0] else "lower left"); a.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(F1D / "fig_scam_roc_pr.png", dpi=300); plt.close(fig)

lp = L1[(L1.threshold == "cv_f1") & (L1.set.isin(["test", "real"]))]
fig, axes = plt.subplots(2, 2, figsize=(14, 8.5))
langs = ["EN", "SI", "TA", "mixed"]
for r, s in enumerate(["test", "real"]):
    for c, (metric, title) in enumerate([("recall", "Scam recall (%)"), ("fpr", "False-positive rate (%)  - lower is better")]):
        ax = axes[r, c]; x = np.arange(len(langs)); w = 0.2
        for i, m in enumerate(MODELS):
            v = [lp[(lp.model == m) & (lp.set == s) & (lp.lang == g)][metric] for g in langs]
            v = [x_.item() if len(x_) else np.nan for x_ in v]
            b = ax.bar(x + (i - 1.5) * w, v, w, color=COL[m], label=NICE[m]); ax.bar_label(b, fmt="%.0f", fontsize=7, padding=1)
        ns = [lp[(lp.model == "SVM") & (lp.set == s) & (lp.lang == g)] for g in langs]
        ax.set_xticks(x, [f"{g}\n(n={n.n.item()}, scams={n.n_pos.item()})" if len(n) else g for g, n in zip(langs, ns)], fontsize=8.5)
        ax.set_title(f"{title} - {'held-out test' if s == 'test' else 'independent real SMS'}", fontsize=10)
        ax.set_ylim(0, 110); ax.spines[["top", "right"]].set_visible(False)
axes[0, 0].legend(frameon=False, ncol=4, fontsize=8.5, loc="lower left")
fig.tight_layout(); fig.savefig(F1D / "fig_scam_by_language.png", dpi=300); plt.close(fig)

# definition comparison on real SMS: old spam+scam models (models/final, trained WITHOUT real SMS) vs scam-only
d_real = pd.read_csv(ROOT / "data/processed/real_test_feat.csv")
comp = []
for m in ["NaiveBayes", "SVM", "RandomForest"]:
    old = joblib.load(ROOT / f"models/final/{m}.joblib").predict(d_real.text_model.astype(str))
    new = rl[f"{m}_cv_f1_pred"].to_numpy()
    for tag, p in [("Spam+scam (baseline)", old), ("Scam-only (primary)", new)]:
        comp.append({"model": m, "definition": tag,
                     "normal_flagged": round(p[(rl.label == "ham").to_numpy()].mean() * 100, 2),
                     "promo_flagged": round(p[(rl.label == "spam").to_numpy()].mean() * 100, 2),
                     "scam_caught": round(p[(rl.label == "smishing").to_numpy()].mean() * 100, 2)})
comp = pd.DataFrame(comp); comp.to_csv(R1 / "definition_comparison_real.csv", index=False)
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
for ax, (c, t) in zip(axes, [("normal_flagged", "Normal SMS flagged (%) - lower is better"),
                             ("promo_flagged", "Promotional SMS flagged (%)"), ("scam_caught", "Scams caught (%)")]):
    x = np.arange(3); w = 0.36
    for i, (tag, colr) in enumerate([("Spam+scam (baseline)", "#bdbdbd"), ("Scam-only (primary)", "#08519c")]):
        v = [comp[(comp.model == m) & (comp.definition == tag)][c].item() for m in ["NaiveBayes", "SVM", "RandomForest"]]
        b = ax.bar(x + (i - .5) * w, v, w, color=colr, label=tag); ax.bar_label(b, fmt="%.1f", fontsize=8.5, padding=2)
    ax.set_xticks(x, ["Naive Bayes", "SVM", "Random Forest"]); ax.set_ylim(0, 112); ax.set_title(t, fontsize=10.5)
    ax.spines[["top", "right"]].set_visible(False)
fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0))
fig.suptitle(f"Independent real SMS (n={len(rl)}): effect of the label definition (both trained without real SMS)", y=1.07, fontsize=12)
fig.tight_layout(); fig.savefig(F1D / "fig_definition_comparison.png", dpi=300, bbox_inches="tight"); plt.close(fig)

# error patterns
ep = pd.read_csv(R1 / "error_patterns.csv"); ft = pd.read_csv(R1 / "fp_top_features.csv")
fpp = ep[(ep.set == "real") & (ep.error == "false positive")].pivot_table(index="pattern", columns="model", values="count", fill_value=0)
fpp = fpp.reindex(columns=[m for m in MODELS if m in fpp.columns]); fpp = fpp.loc[fpp.sum(1).sort_values().index]
fig, axes = plt.subplots(1, 2, figsize=(15, 5.2), gridspec_kw={"width_ratios": [1.3, 1]})
y = np.arange(len(fpp)); h = 0.2
for i, m in enumerate(fpp.columns):
    axes[0].barh(y + (i - 1.5) * h, fpp[m], h, color=COL[m], label=NICE[m])
axes[0].set_yticks(y, fpp.index, fontsize=9); axes[0].set_xlabel("number of false positives containing the pattern")
nfp = ep[(ep.set == "real") & (ep.error == "false positive")].groupby("model").n_errors.first()
axes[0].set_title("Real SMS false positives by pattern (" + ", ".join(f"{NICE[m]} {nfp.get(m, 0)}" for m in MODELS) + ")", fontsize=10)
axes[0].legend(frameon=False, fontsize=8.5); axes[0].spines[["top", "right"]].set_visible(False)
tf = ft[ft.set == "test"].head(15)[::-1]
if len(tf):
    axes[1].barh(tf.ngram, tf.n_false_positives_containing, color="#4292c6")
    axes[1].set_title(f"SVM: n-grams pushing test-set false positives to 'scam' (n={tf.n_false_positives.iloc[0]})", fontsize=10)
    axes[1].spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(F1D / "fig_scam_fp_patterns.png", dpi=300); plt.close(fig)

# ================================================================ three-class
m3 = pd.read_csv(R3 / "metrics.csv"); p3 = pd.read_csv(R3 / "pred_real.csv"); L3 = pd.read_csv(R3 / "by_language.csv")
c3 = ["acc", "macro_precision", "macro_recall", "macro_f1", "macro_roc_auc", "macro_auprc", "scam_recall", "scam_fpr", "normal_flagged_as_scam"]
h3 = ["Model", "Set", "Acc", "Macro P", "Macro R", "Macro F1", "Macro AUC", "Macro AUPRC", "Scam recall", "Scam FPR", "Normal->scam"]
rows3 = [[NICE[m] if i == 0 else "", lab] + [f(m3[(m3.model == m) & (m3.set == s)][c].item()) for c in c3]
         for m in MODELS for i, (s, lab) in enumerate(SETS)]
table_fig(rows3, h3, "Table: Three-class experiment - Normal / Promotional / Scam (%)", F3D / "fig_three_class_table.png",
          widths=[.1, .22] + [.075] * 9, note=XGB_NOTE)
y3 = p3.label.map({"ham": 0, "spam": 1, "smishing": 2})
cm_grid([y3] * 4, [p3[f"{m}_default_pred"] for m in MODELS],
        [f"{NICE[m]} (macro F1 {m3[(m3.model == m) & (m3.set == 'real')].macro_f1.item():.1f}%)" for m in MODELS],
        ["Normal", "Promo", "Scam"], F3D / "fig_three_class_cm_real.png", f"Three-class confusion matrices - independent real SMS (n={len(p3)})")

# ================================================================ REPORT.md
def mdtable(df, cols, names):
    out = ["| " + " | ".join(names) + " |", "|" + "---|" * len(names)]
    out += ["| " + " | ".join(str(r[c]) if not isinstance(r[c], float) else f(r[c]) for c in cols) + " |" for _, r in df.iterrows()]
    return out

cfg3 = json.loads((R3 / "config.json").read_text())
md = ["# Scam-only smishing detection - primary experiment", "",
      "Generated by `src/scam_pipeline.py`, `src/scam_errors.py`, `src/scam_report.py`. Configuration, seeds, data hashes and "
      "library versions: `results/scam_only/config.json`, `results/three_class/config.json`.", "",
      "## Definitions and data roles", "",
      "- **Positive class = scam (smishing)** only. Normal SMS **and promotional SMS are negative**: a genuine promotion from a "
      "real company is unwanted but not fraudulent. The earlier spam+scam results are kept as a baseline experiment "
      "(`results/*.csv`, `models/final`, `models/final_v2`).",
      f"- **Training** {cfg1['data']['train']['rows']} messages; **validation** = grouped 5-fold CV on the training set "
      f"({cfg1['cv']['n_groups']} groups: templates / translation sources kept in one fold).",
      f"- **Held-out test** {cfg1['data']['test']['rows']} messages (public datasets + generated / translated templates) - "
      "same distribution family as training.",
      f"- **Independent real SMS** {cfg1['data']['real']['rows']} messages from Sri Lankan phones "
      f"({cfg1['class_counts']['real']}) - used ONLY for final evaluation (no training, tuning or threshold choice).",
      "- Augmentation (noisy copies), TF-IDF fitting and SMOTE are done inside each training fold only.",
      "- Threshold: chosen on CV out-of-fold scores to maximise F1 (training data only). Default-threshold results are in `metrics.csv`.",
      f"- {XGB_NOTE}.", "",
      "## Results (%, CV-chosen threshold)", ""]
show = prim.copy(); show["model"] = show.model.map(NICE)
md += mdtable(show, ["model", "set"] + cols, ["Model", "Set", "Acc", "Precision", "Recall", "F1", "FPR", "ROC-AUC", "AUPRC"])
md += ["", "Training (in-sample) scores are optimistic by construction and are not evidence of generalisation. "
       "A high training score does not prove overfitting, and the gap to real SMS shows the models generalise only partly; "
       "overfitting has **not** been shown to be eliminated.", "",
       "## Per-language (independent real SMS)", ""]
lr = lp[lp.set == "real"].copy(); lr["model"] = lr.model.map(NICE)
md += mdtable(lr, ["model", "lang", "n", "n_pos", "precision", "recall", "f1", "fpr"], ["Model", "Lang", "n", "scams", "Precision", "Recall", "F1", "FPR"])
md += ["", "Language groups: EN; SI = Sinhala + Singlish; TA = Tamil + Tanglish; mixed. Real Tamil n is very small.", "",
       "## Effect of the label definition (real SMS, models trained without real SMS)", ""]
cc = comp.copy(); cc["model"] = cc.model.map(NICE)
md += mdtable(cc, ["model", "definition", "normal_flagged", "promo_flagged", "scam_caught"], ["Model", "Definition", "Normal flagged", "Promo flagged", "Scams caught"])
md += ["", "## Error analysis (real SMS false positives by pattern)", ""]
fpr_tab = fpp.reset_index(); fpr_tab.columns = ["pattern"] + [NICE[c] for c in fpp.columns]
md += mdtable(fpr_tab.iloc[::-1], list(fpr_tab.columns), ["Pattern"] + list(fpr_tab.columns[1:]))
md += ["", "Top n-grams pushing SVM false positives towards 'scam' (test set): " +
       ", ".join(f"`{a}` ({b})" for a, b in ft[ft.set == "test"].head(12)[["ngram", "n_false_positives_containing"]].values), "",
       "Per-message errors (contain private text, not for GitHub): `results/scam_only/pred_errors_*.csv`.", "",
       "## Three-class experiment (Normal / Promotional / Scam)", ""]
s3 = m3.copy(); s3["model"] = s3.model.map(NICE)
md += mdtable(s3, ["model", "set"] + c3, h3)
md += ["", "## Limitations", "",
       f"- The real SMS set is small ({len(rl)} messages, {int(y_rl.sum())} scams); per-language real results (especially Tamil) are not reliable.",
       "- The held-out test set shares sources and generation templates with training, so it over-estimates real-world performance.",
       "- Hyper-parameters were carried over from earlier grouped-CV tuning (spam+scam labels) and not re-tuned for scam-only.",
       "- Respondent and researcher labels agreed 73% (kappa 0.57); labels for promotional vs scam can be ambiguous.", "",
       "## Reproduce", "", "```", "python -m src.scam_pipeline scam_only", "python -m src.scam_pipeline three_class",
       "python -m src.scam_errors", "python -m src.scam_report", "```"]
(R1 / "REPORT.md").write_text("\n".join(md), encoding="utf-8")
print("\n".join(md))

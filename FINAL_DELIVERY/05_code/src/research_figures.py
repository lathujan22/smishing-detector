"""Research figures for the baselines (report / paper).
    python -m src.research_figures  -> results/figures/
      fig_grid_search.png        (a) SVM C  (b) Naive Bayes alpha - 5-fold grouped CV F1
      fig_top_ngrams.png         top 15 word/character n-grams by Random Forest importance
      fig_feature_importance.png  permutation importance of the 8 hand-crafted smishing features
"""
import re, sys, time
from pathlib import Path
import joblib, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from src.baselines import models, cv_groups, SEED
from src.evaluate import to_binary
from src.features import FLAG_NAMES

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "figures"; OUT.mkdir(parents=True, exist_ok=True)
BLUE, RED, GREY = "#2f5f98", "#c0392b", "#a3a7b3"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
tr = pd.read_csv(ROOT / "data" / "processed" / "train_feat.csv")
te = pd.read_csv(ROOT / "data" / "processed" / "test_feat.csv")
X, y, g = tr.text_model.astype(str), to_binary(tr.label3), cv_groups(tr)
cv = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(X, y, g))


def grid(pipe, param, values):
    out = []
    for v in values:
        p = clone(pipe).set_params(**{param: v})
        out.append(100 * cross_val_score(p, X, y, groups=g, cv=cv, scoring="f1").mean())
        print(f"  {param}={v}: F1 {out[-1]:.2f}", flush=True)
    return np.array(out)


def panel(ax, xs, ys, xlabel, title, letter):
    best = int(np.argmax(ys))
    ax.plot(range(len(xs)), ys, "o-", color=BLUE, lw=1.8, ms=5)
    ax.axvline(best, color=RED, ls="--", lw=1.2, label=f"Best = {xs[best]} (F1 {ys[best]:.2f}%)")
    ax.set_xticks(range(len(xs)), [str(x) for x in xs]); ax.set_xlabel(xlabel); ax.set_ylabel("Validation F1 (%)")
    ax.set_title(title, weight="bold"); ax.legend(frameon=False, loc="lower right"); ax.grid(alpha=.25)
    ax.text(-0.12, 1.06, letter, transform=ax.transAxes, weight="bold", fontsize=12)


m = models()
GS = ROOT / "results" / "grid_search.csv"
if GS.exists() and "--regrid" not in sys.argv:
    gs = pd.read_csv(GS); Cs = list(gs[gs.param == "svm_C"].value); As = list(gs[gs.param == "nb_alpha"].value)
    Cs = [int(c) if float(c).is_integer() else c for c in Cs]
    f_svm, f_nb = gs[gs.param == "svm_C"].cv_f1.to_numpy(), gs[gs.param == "nb_alpha"].cv_f1.to_numpy()
else:
  print("grid search SVM C"); Cs = [0.01, 0.05, 0.1, 0.5, 1, 2, 5]; f_svm = grid(m["SVM"], "clf__C", Cs)
  print("grid search NB alpha"); As = [0.01, 0.05, 0.1, 0.3, 0.5, 1.0]; f_nb = grid(m["NaiveBayes"], "clf__alpha", As)
  pd.DataFrame({"param": ["svm_C"] * len(Cs) + ["nb_alpha"] * len(As), "value": Cs + As, "cv_f1": list(f_svm) + list(f_nb)}).to_csv(ROOT / "results" / "grid_search.csv", index=False)
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
panel(axes[0], Cs, f_svm, "SVM regularisation C", "Linear SVM - C grid search", "(a)")
panel(axes[1], As, f_nb, "Naive Bayes smoothing α", "Naive Bayes - α grid search", "(b)")
fig.tight_layout(); fig.savefig(OUT / "fig_grid_search.png", dpi=300, facecolor="white"); plt.close(fig)

# --- top n-grams by Random Forest impurity importance
rf = joblib.load(ROOT / "models" / "baselines" / "RandomForest.joblib")
names = rf.named_steps["tfidf"].get_feature_names_out()
imp = rf.named_steps["clf"].feature_importances_
word = np.array([n.startswith("word__") for n in names])          # whole words / phrases are readable
idx = np.where(word)[0]; top = idx[np.argsort(imp[idx])[::-1][:15]]
GLOSS = {"உங்கள்": "ungal (Tamil: 'your')", "කරන්න": "karanna (Sinhala: 'do')", "ඔබගේ": "obage (Sinhala: 'your')",
         "உடனே": "udane (Tamil: 'immediately')", "வங்கி": "vangi (Tamil: 'bank')", "ගිණුම": "ginuma (Sinhala: 'account')"}
def readable(t):  # matplotlib cannot shape Sinhala/Tamil script - show romanisation + meaning instead
    out = []
    for w in t.split():
        if re.search("[\u0D80-\u0DFF]", w): w = GLOSS.get(w, "(Sinhala word)")
        elif re.search("[\u0B80-\u0BFF]", w): w = GLOSS.get(w, "(Tamil word)")
        out.append(w)
    return " ".join(out)
lab = [readable(n.split("__", 1)[1]) for n in names[top]]
pd.DataFrame({"feature": lab, "importance": imp[top]}).to_csv(ROOT / "results" / "top_ngrams.csv", index=False)

def barh(vals, labels, xlabel, fname, fmt="{:.4f}", n_hi=5):
    fig, ax = plt.subplots(figsize=(8.5, 0.36 * len(vals) + 1.2))
    cols = [BLUE] * n_hi + [GREY] * (len(vals) - n_hi)
    ax.barh(range(len(vals))[::-1], vals, color=cols, height=0.7)
    ax.set_yticks(range(len(vals))[::-1], labels)
    for i, v in enumerate(vals): ax.text(v + max(vals) * 0.01, len(vals) - 1 - i, fmt.format(v), va="center", fontsize=9, color="#1f3b5c")
    ax.set_xlabel(xlabel); ax.set_xlim(0, max(vals) * 1.15); ax.grid(axis="x", alpha=.25)
    fig.tight_layout(); fig.savefig(OUT / fname, dpi=300, facecolor="white"); plt.close(fig)

barh(imp[top], lab, "Mean decrease in impurity (Random Forest, word / phrase features)", "fig_top_ngrams.png")

# --- 8 hand-crafted features: RF on flags only, permutation importance on the test set (drop in F1)
Xf, Xft = tr[FLAG_NAMES].to_numpy(float), te[FLAG_NAMES].to_numpy(float)
rf8 = RandomForestClassifier(300, class_weight="balanced", random_state=SEED, n_jobs=-1).fit(Xf, y)
pi = permutation_importance(rf8, Xft, to_binary(te.label3), scoring="f1", n_repeats=20, random_state=SEED, n_jobs=-1)
order = np.argsort(pi.importances_mean)[::-1]
pretty = {"has_url": "Has URL", "url_shortener": "URL shortener", "risky_domain": "Risky / look-alike domain", "has_phone": "Phone number",
          "has_money": "Money amount", "urgency": "Urgency words", "impersonation": "Brand / authority impersonation", "action_request": "Asks to click / pay / verify"}
pd.DataFrame({"feature": [pretty[FLAG_NAMES[i]] for i in order], "f1_drop": pi.importances_mean[order], "std": pi.importances_std[order]}).to_csv(ROOT / "results" / "feature_importance.csv", index=False)
barh(100 * pi.importances_mean[order], [pretty[FLAG_NAMES[i]] for i in order], "Drop in test F1 when the feature is shuffled (percentage points)",
     "fig_feature_importance.png", fmt="{:.2f}", n_hi=4)
print("flags-only RF test F1:", round(100 * __import__("sklearn.metrics", fromlist=["f1_score"]).f1_score(to_binary(te.label3), rf8.predict(Xft)), 2))
print(sorted(p.name for p in OUT.glob("*.png")))

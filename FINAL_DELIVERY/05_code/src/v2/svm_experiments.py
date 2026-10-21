"""V2 controlled SVM experiments (binary scam vs non-scam) - one factor at a time, grouped CV selection.
    python -m src.v2.svm_experiments
Selection uses ONLY grouped 5-fold CV on data/v2/train.csv (groups = near-duplicate family_id).
Primary selection metric: CV AUPRC (threshold-free); thresholds are chosen on CV out-of-fold scores (F1-max).
Held-out test (data/v2/test.csv) and the real benchmark are evaluated only for the final comparison models.
Outputs results/v2/svm/: round1_cv.csv, greedy.csv, final.csv, final_by_language.csv, svm_manifest.json, REPORT.md
        models/v2/svm_*.joblib
"""
import copy
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.v2.common import DATA, MOD, RES, SEED, write_manifest
from src.v2.evaluation import binary, bootstrap_ci, f1_threshold, per_language, pct
from src.v2.models import BASE, TextModel

OUT = RES / "svm"; OUT.mkdir(parents=True, exist_ok=True); MOD.mkdir(parents=True, exist_ok=True)
W, C_ = BASE["word"], BASE["char"]
ROUND1 = {   # name: (experiment factor, cfg changes relative to the ORIGINAL baseline)
    "E0 baseline (original recipe)": ("baseline", {}),
    "word only": ("features", {"char": None}),
    "char only": ("features", {"word": None}),
    "word 1-3 grams": ("n-grams", {"word": {**W, "ngram": (1, 3)}}),
    "char 1-4 grams": ("n-grams", {"char": {**C_, "ngram": (1, 4)}}),
    "char 3-6 grams": ("n-grams", {"char": {**C_, "ngram": (3, 6)}}),
    "30k word + 30k char features": ("vocabulary", {"word": {**W, "max_features": 30000}, "char": {**C_, "max_features": 30000}}),
    "C = 0.03": ("regularisation", {"C": 0.03}),
    "C = 0.3": ("regularisation", {"C": 0.3}),
    "C = 1.0": ("regularisation", {"C": 1.0}),
    "class weighting instead of SMOTE": ("imbalance", {"resample": "balanced"}),
    "no resampling": ("imbalance", {"resample": "none"}),
    "no noise augmentation": ("augmentation", {"noise": False}),
    "+ contextual risk features": ("context", {"context": True}),
    "+ contextual risk features (x3 weight)": ("context", {"context": True, "context_weight": 3.0}),
    "sigmoid-calibrated SVM": ("calibration", {"calibrate": True}),
    "train without machine-translated rows": ("training data", {"train_filter": "no_mt"}),
    "train on human-written data only": ("training data", {"train_filter": "human_only"}),
}


def filt(df, how):
    if how == "no_mt":
        return df[df.source != "mt_nllb"]
    if how == "human_only":
        return df[~df.is_synthetic.astype(bool)]
    return df


def cv_eval(cfg, tr, y, folds):
    oof = np.full(len(tr), np.nan)
    for a, b in folds:
        sub = filt(tr.iloc[a], cfg.get("train_filter"))
        m = TextModel(cfg).fit(sub, y[sub.index.to_numpy()])
        oof[b] = m.score(tr.iloc[b])
    thr = f1_threshold(y, oof)
    r = binary(y, (oof >= thr).astype(int), oof)
    return r, thr, oof


def fit_eval(name, cfg, tr, y, te, yte, rl, yrl, thr):
    sub = filt(tr, cfg.get("train_filter"))
    m = TextModel(cfg).fit(sub, y[sub.index.to_numpy()])
    joblib.dump({"model": m, "threshold": thr, "cfg": cfg}, MOD / ("svm_" + "".join(ch if ch.isalnum() else "_" for ch in name.lower()).strip("_") + ".joblib"))
    rows, langs, preds = [], [], {}
    tr_s = m.score(tr)
    for sname, d, yy in [("train (in-sample)", tr, y), ("held-out test", te, yte), ("real benchmark", rl, yrl)]:
        s = tr_s if sname.startswith("train") else m.score(d); p = (s >= thr).astype(int)
        r = {"model": name, "set": sname, **pct(binary(yy, p, s))}
        if not sname.startswith("train"):
            r.update(bootstrap_ci(yy, p, s))
            langs.append(per_language(d.lang_family.values, yy, p, s).assign(model=name, set=sname))
            preds[sname] = (s, p)
        rows.append(r)
    return m, rows, langs, preds


if __name__ == "__main__":
    t0 = time.time()
    tr = pd.read_csv(DATA / "train.csv"); te = pd.read_csv(DATA / "test.csv")
    reg = pd.read_csv(DATA / "registry.csv"); rl = reg[reg.role == "real_benchmark"].reset_index(drop=True)
    y, yte, yrl = [(d.label3 == "smishing").astype(int).to_numpy() for d in (tr, te, rl)]
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(tr, tr.class3 + "|" + tr.lang_family, tr.family_id))
    write_manifest(OUT, "svm", {"base_cfg": BASE, "round1": {k: v[1] for k, v in ROUND1.items()},
                                "cv": "StratifiedGroupKFold(5, shuffle, seed 42), groups=family_id, strata=class3|lang_family",
                                "selection": "CV AUPRC; threshold = CV out-of-fold F1-max",
                                "real_benchmark_policy": "never used for selection; reported only"},
                   {"train": DATA / "train.csv", "test": DATA / "test.csv", "registry": DATA / "registry.csv"})

    # ---------------- round 1: one factor at a time
    r1, oofs = [], {}
    for name, (factor, ch) in ROUND1.items():
        cfg = {**BASE, **ch}; t1 = time.time()
        r, thr, oof = cv_eval(cfg, tr, y, folds); oofs[name] = oof
        r1.append({"config": name, "factor": factor, "threshold": round(thr, 4), **pct(r), "seconds": round(time.time() - t1)})
        print(f"{name:<42} AUPRC {r['auprc']*100:.2f}  F1 {r['f1']*100:.2f}  FPR {r['fpr']*100:.2f}  ({time.time()-t1:.0f}s)", flush=True)
    R1 = pd.DataFrame(r1); base = R1.iloc[0]
    R1["d_auprc"] = (R1.auprc - base.auprc).round(2); R1["d_f1"] = (R1.f1 - base.f1).round(2)
    R1.to_csv(OUT / "round1_cv.csv", index=False)

    # ---------------- greedy combination of helpful factors (CV only)
    cand = R1[(R1.d_auprc > 0.1) & (R1.d_f1 > -0.5) & (R1.factor != "baseline")].sort_values("d_auprc", ascending=False)
    best_cfg, best = copy.deepcopy(BASE), float(base.auprc); used_factors, greedy = set(), []
    for _, row in cand.iterrows():
        if row.factor in used_factors:
            continue
        trial = {**best_cfg, **ROUND1[row.config][1]}
        r, thr, _ = cv_eval(trial, tr, y, folds); a = r["auprc"] * 100
        keep = a > best + 0.05
        greedy.append({"added": row.config, "cv_auprc": round(a, 2), "cv_f1": round(r["f1"] * 100, 2), "kept": keep})
        print(f"greedy + {row.config:<40} AUPRC {a:.2f} {'KEEP' if keep else 'drop'}", flush=True)
        if keep:
            best_cfg, best = trial, a; used_factors.add(row.factor)
    pd.DataFrame(greedy).to_csv(OUT / "greedy.csv", index=False)
    json.dump(best_cfg, open(OUT / "improved_cfg.json", "w"), indent=2, default=str)

    # ---------------- final comparison models (test + real reported, nothing selected on them)
    finals = {"A: original SVM (baseline)": {**BASE},
              "A: improved SVM (CV-selected)": best_cfg,
              "E: word-only TF-IDF": {**BASE, "char": None},
              "E: word + char TF-IDF": {**BASE},
              "F: without contextual features": {**best_cfg, "context": False},
              "F: with contextual features": {**best_cfg, "context": True, "context_weight": best_cfg.get("context_weight", 1.0)}}
    rows, langs, pred_out = [], [], {"held-out test": te[["msg_id", "lang_family", "class3"]].copy(),
                                     "real benchmark": rl[["msg_id", "lang_family", "class3"]].copy()}
    done = {}
    for name, cfg in finals.items():
        key = json.dumps(cfg, sort_keys=True, default=str)
        r, thr, _ = cv_eval(cfg, tr, y, folds)
        rows.append({"model": name, "set": "cv (validation)", **pct(r)})
        if key in done:
            m, rr, ll, pp = done[key]
            rr = [{**x, "model": name} for x in rr]; ll = [x.assign(model=name) for x in ll]
        else:
            m, rr, ll, pp = fit_eval(name, cfg, tr, y, te, yte, rl, yrl, thr); done[key] = (m, rr, ll, pp)
        rows += rr; langs += ll
        for sname, (s, p) in pp.items():
            pred_out[sname][f"{name}|score"] = s; pred_out[sname][f"{name}|pred"] = p
        print(f"final {name} done", flush=True)
    F = pd.DataFrame(rows); F.to_csv(OUT / "final.csv", index=False)
    pd.concat(langs).to_csv(OUT / "final_by_language.csv", index=False)
    pred_out["held-out test"].to_csv(OUT / "pred_test.csv", index=False)
    pred_out["real benchmark"].to_csv(OUT / "pred_real.csv", index=False)
    print(f"total {time.time()-t0:.0f}s")

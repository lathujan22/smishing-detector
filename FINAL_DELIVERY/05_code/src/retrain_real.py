"""Retrain the baselines WITH real Sri Lankan SMS in training, evaluated by 5-fold CV on the real set.
    python -m src.retrain_real -> results/retrain_real.csv, results/RETRAIN_REAL.md, results/pred_retrain_real.csv,
                                  models/final_v2/*.joblib (trained on train + ALL real SMS, for the demo)

Protocol (no message is ever scored by a model that saw it):
  - the 163 real SMS are split into 5 folds; near-duplicates (char-TF-IDF cosine >= 0.6) share a fold
  - fold k: train = original training set + real SMS of the other 4 folds (each real row x5, fixed a priori),
            same settings as models/final (noise + SMOTE); predict the real SMS of fold k
  - real-SMS scores are computed on the pooled out-of-fold predictions
  - the original held-out test set (n=1,589) is scored by the v2 models to check nothing got worse
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedGroupKFold

from src.baselines import SEED
from src.evaluate import by_language, metrics, to_binary
from src.final_baselines import CONF
from src.robustness import train

ROOT = Path(__file__).resolve().parents[1]
PROC, RES = ROOT / "data/processed", ROOT / "results"
OUT = ROOT / "models/final_v2"; OUT.mkdir(parents=True, exist_ok=True)
REAL_WEIGHT = 5
P = lambda m: {k: round(m[k] * 100, 2) for k in ["acc", "precision", "recall", "f1", "fpr"]} | {"fp": m["fp"], "fn": m["fn"]}


def near_dup_groups(texts):
    M = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit_transform(texts)
    return connected_components(cosine_similarity(M) >= 0.6, directed=False)[1]


if __name__ == "__main__":
    tr = pd.read_csv(PROC / "train_feat.csv"); te = pd.read_csv(PROC / "test_feat.csv")
    rl = pd.read_csv(PROC / "real_test_feat.csv").assign(label3=lambda d: d.label)
    y_rl, y_te = to_binary(rl.label), to_binary(te.label3)
    groups = near_dup_groups(rl.text_model.astype(str))
    print(f"real SMS {len(rl)} -> {len(set(groups))} near-duplicate groups", flush=True)
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(rl, rl.label, groups))
    cols = ["text_model", "label3"]

    old = {m: joblib.load(ROOT / f"models/final/{m}.joblib") for m in CONF}
    pred = rl[["id", "text", "label", "language"]].copy(); rows, lang_rows = [], []
    for name in CONF:
        oof = np.zeros(len(rl), int)
        for k, (a, b) in enumerate(folds):
            d = pd.concat([tr[cols]] + [rl.iloc[a][cols]] * REAL_WEIGHT, ignore_index=True)
            oof[b] = train(name, d).predict(rl.iloc[b].text_model.astype(str))
        v2 = train(name, pd.concat([tr[cols]] + [rl[cols]] * REAL_WEIGHT, ignore_index=True))
        joblib.dump(v2, OUT / f"{name}.joblib")
        p_old = old[name].predict(rl.text_model.astype(str))
        pred[f"{name}_before"], pred[f"{name}_after"] = p_old, oof
        for tag, p_r, mdl in [("before (models/final)", p_old, old[name]), ("after (+ real SMS, 5-fold CV)", oof, v2)]:
            r = {"model": name, "version": tag, **{f"real_{k}": v for k, v in P(metrics(y_rl, p_r)).items()},
                 "real_scam_caught": round(p_r[rl.label.eq("smishing").to_numpy()].mean() * 100, 2),
                 **{f"test_{k}": v for k, v in P(metrics(y_te, mdl.predict(te.text_model.astype(str)))).items()}}
            rows.append(r)
        lang_rows.append(by_language(rl.language, y_rl, oof).assign(model=name))
        print(name, "done", flush=True)

    r = pd.DataFrame(rows); r.to_csv(RES / "retrain_real.csv", index=False)
    pd.concat(lang_rows).to_csv(RES / "retrain_real_by_language.csv", index=False)
    pred.to_csv(RES / "pred_retrain_real.csv", index=False, encoding="utf-8")
    md = ["# Retraining with real Sri Lankan SMS (%)", "",
          f"Real SMS n={len(rl)} (ham {int((rl.label == 'ham').sum())}, smishing {int((rl.label == 'smishing').sum())}, "
          f"spam {int((rl.label == 'spam').sum())}); 'after' = pooled 5-fold out-of-fold predictions. Test n={len(te)}.", "",
          "| Model | Version | Real acc | Real F1 | Real FPR | Scams caught | Test acc | Test F1 | Test FPR |",
          "|---|---|---|---|---|---|---|---|---|"]
    md += [f"| {x.model} | {x.version} | {x.real_acc} | {x.real_f1} | {x.real_fpr} | {x.real_scam_caught} | "
           f"{x.test_acc} | {x.test_f1} | {x.test_fpr} |" for x in r.itertuples()]
    (RES / "RETRAIN_REAL.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))

"""V2 train / held-out test split with NO shared near-duplicate families, templates or translation sources.
    python -m src.v2.split  -> data/v2/train.csv, data/v2/test.csv, results/v2/data/split_report.md
Grouped by family_id (src.v2.registry), stratified on class x language family, ~15% test.
Quarantined rows are excluded. The 163 real SMS are NOT part of this split (real_benchmark)."""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import NearestNeighbors

from src.v2.common import DATA, RES, SEED

if __name__ == "__main__":
    reg = pd.read_csv(DATA / "registry.csv")
    pool = reg[(reg.role == "pool") & (~reg.quarantine)].reset_index(drop=True)
    strat = pool.class3 + "|" + pool.lang_family
    tr_idx, te_idx = next(StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=SEED).split(pool, strat, pool.family_id))
    tr, te = pool.iloc[tr_idx], pool.iloc[te_idx]
    assert not set(tr.family_id) & set(te.family_id)
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True).fit(pool.text_model.astype(str))
    nn = NearestNeighbors(n_neighbors=1, metric="cosine", algorithm="brute").fit(vec.transform(tr.text_model.astype(str)))
    sim = 1 - nn.kneighbors(vec.transform(te.text_model.astype(str)))[0][:, 0]
    tr.to_csv(DATA / "train.csv", index=False, encoding="utf-8"); te.to_csv(DATA / "test.csv", index=False, encoding="utf-8")
    t = lambda d: pd.crosstab(d.lang_family, d.class3, margins=True)
    md = ["# V2 split report", "",
          f"- Pool (non-quarantined) {len(pool)} -> train {len(tr)} / held-out test {len(te)} ({len(te)/len(pool):.1%})",
          f"- Families: train {tr.family_id.nunique()}, test {te.family_id.nunique()}, shared **0**",
          f"- Test -> train nearest char-n-gram cosine: max {sim.max():.3f}, 99th pct {np.percentile(sim, 99):.3f}, "
          f"median {np.median(sim):.3f} (v1 had 354 test messages >= 0.8)",
          f"- Synthetic / MT share: train {tr.is_synthetic.mean():.1%}, test {te.is_synthetic.mean():.1%}", "",
          "## Train", "", t(tr).to_markdown(), "", "## Held-out test", "", t(te).to_markdown()]
    (RES / "data" / "split_report.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))

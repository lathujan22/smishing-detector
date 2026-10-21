"""Error analysis for the scam-only models: which messages are wrongly flagged / missed, and why.
    python -m src.scam_errors -> results/scam_only/error_patterns.csv, fp_top_features.csv,
                                 pred_errors_{test,real}.csv (contain message text - kept private)
Patterns are simple keyword rules on the masked text (multi-label: one message can match several)."""
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.scam_pipeline import MODELS

ROOT = Path(__file__).resolve().parents[1]; RES = ROOT / "results" / "scam_only"
PATTERNS = {
    "OTP / verification code": r"\botp\b|verification code|one[- ]time|\bpin\b|\bcode\b|passcode",
    "Bank / payment alert": r"credited|debited|transfer|transaction|balance|payment|paid|star points|cheer ?points",
    "Telco plan / service notice": r"\bplan\b|package|activat|expir|\bdata\b|subscri|reload|dial|\bgb\b",
    "Delivery / order": r"order|dispatch|deliver|parcel|track|courier|shipment|customs",
    "Promotion / offer wording": r"offer|discount|% ?off|cashback|\bfree\b|deal|t&c|stopad|stop promo|\bwin\b",
    "Contains link": r"<url>|https?://|www\.",
    "Money amount": r"<money>|\brs\.?|\blkr\b",
    "Urgency wording": r"urgent|immediately|within 24|last chance|today|now\b|asap|final",
    "Account / security wording": r"account|verify|secure|blocked|suspend|confirm",
}


def tag(text: str) -> list[str]:
    t = str(text).lower()
    tags = [k for k, rx in PATTERNS.items() if re.search(rx, t)]
    if not re.search(r"<url>|https?://", t) and len(t) < 40:
        tags.append("Short personal chat")
    return tags or ["Other"]


def svm_contributions(pipe, texts, top=8):
    """Top n-grams pushing each text towards 'scam' (linear SVM: x_i * w_i)."""
    vec, clf = pipe.named_steps["tfidf"], pipe.named_steps["clf"]
    names, w = vec.get_feature_names_out(), clf.coef_.ravel()
    X = vec.transform(texts).tocsr(); out = []
    for i in range(X.shape[0]):
        row = X.getrow(i); c = row.data * w[row.indices]
        idx = row.indices[np.argsort(-c)][:top]
        out.append([names[j].split("__", 1)[-1] for j in idx if w[j] > 0])
    return out


if __name__ == "__main__":
    pat_rows, feat_rows = [], []
    for s, lcol in [("real", "label"), ("test", "label3")]:
        p = pd.read_csv(RES / f"pred_{s}.csv")
        d = pd.read_csv(ROOT / "data/processed" / ("real_test_feat.csv" if s == "real" else "test_feat.csv"))
        p["text_model"] = d.text_model.astype(str).values
        y = (p[lcol] == "smishing").astype(int)
        p["tags"] = p.text_model.map(tag)
        errs = []
        for m in MODELS:
            pr = p[f"{m}_cv_f1_pred"]
            for kind, mask in [("false positive", (pr == 1) & (y == 0)), ("false negative", (pr == 0) & (y == 1))]:
                e = p[mask]
                for t in sorted({x for ts in e.tags for x in ts}):
                    pat_rows.append({"set": s, "model": m, "error": kind, "pattern": t,
                                     "count": int(e.tags.map(lambda ts: t in ts).sum()), "n_errors": len(e)})
                if s == "real" or kind == "false positive":
                    errs.append(e.assign(model=m, error=kind, true_class=e[lcol].values))
        E = pd.concat(errs)
        n_models = E.groupby(["text", "error"]).model.transform("nunique")
        E.assign(n_models_wrong=n_models)[["model", "error", "true_class", "language", "n_models_wrong", "tags", "text"]] \
            .to_csv(RES / f"pred_errors_{s}.csv", index=False, encoding="utf-8")
        svm = joblib.load(ROOT / "models/scam_only/SVM.joblib")
        fp = p[(p["SVM_cv_f1_pred"] == 1) & (y == 0)]
        if len(fp):
            cnt = pd.Series([f for fs in svm_contributions(svm, fp.text_model) for f in fs]).value_counts().head(25)
            feat_rows += [{"set": s, "ngram": k, "n_false_positives_containing": int(v), "n_false_positives": len(fp)}
                          for k, v in cnt.items()]
    pd.DataFrame(pat_rows).to_csv(RES / "error_patterns.csv", index=False)
    pd.DataFrame(feat_rows).to_csv(RES / "fp_top_features.csv", index=False)
    print(pd.DataFrame(pat_rows).query("set=='real' and error=='false positive'")
          .pivot_table(index="pattern", columns="model", values="count", fill_value=0))
    print(pd.DataFrame(feat_rows).query("set=='real'").head(15).to_string(index=False))

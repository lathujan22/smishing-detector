"""Train + save the FINAL baselines (noise augmentation + SMOTE, training data only).
    python -m src.final_baselines  -> models/final/{NaiveBayes,SVM,RandomForest}.joblib + results/final_baselines.csv
Each saved file is a normal sklearn Pipeline (tfidf -> clf): load with joblib and call .predict(texts)."""
import time
from pathlib import Path
import joblib, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC
from src.augment import noisy_copies, smote
from src.baselines import tfidf, SEED
from src.evaluate import metrics, to_binary

def tfidf_mindf3(n):   # word+char TF-IDF with min_df=3 (best RF setting in src/reduce_gap.py)
    return FeatureUnion([("word", TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True, max_features=n,
                                                  token_pattern=r"<\w+>|[^\s<>.,!?:;()\"']+")),
                         ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True, max_features=n))])

ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "models" / "final"; OUT.mkdir(parents=True, exist_ok=True)
CONF = {"NaiveBayes": (30000, lambda: ComplementNB(alpha=0.01)),                      # alpha from grid search
        "SVM": (10000, lambda: LinearSVC(C=0.1, random_state=SEED)),                  # C from regularisation sweep
        "RandomForest": (8000, lambda: RandomForestClassifier(200, max_features="sqrt", n_jobs=-1, random_state=SEED))}

if __name__ == "__main__":
    tr = pd.read_csv(ROOT / "data/processed/train_feat.csv")
    X, y = tr.text_model.astype(str).tolist(), to_binary(tr.label3)
    Xa, ya, _, _ = noisy_copies(X, y, seed=SEED)                                     # + 1 noisy copy per message
    rows = []
    for name, (n, make) in CONF.items():
        t = time.time(); vec = tfidf_mindf3(n) if name == "RandomForest" else tfidf(n); M = vec.fit_transform(Xa)
        M, yy = smote(M, ya, k=5, seed=SEED)                                          # balance classes 1:1
        clf = make(); clf.fit(M, yy)
        pipe = Pipeline([("tfidf", vec), ("clf", clf)])                               # SMOTE only needed at training time
        joblib.dump(pipe, OUT / f"{name}.joblib")
        for sname, f, lab in [("train", "train_feat.csv", "label3"), ("test", "test_feat.csv", "label3"), ("real", "real_test_feat.csv", "label")]:
            d = pd.read_csv(ROOT / "data/processed" / f); m = metrics(to_binary(d[lab]), pipe.predict(d.text_model.astype(str)))
            rows.append({"model": name, "set": sname, **{k: m[k] for k in ["acc", "precision", "recall", "f1", "fpr"]}})
        print(f"{name:<12} saved ({time.time()-t:.0f}s)", flush=True)
    r = pd.DataFrame(rows); r.iloc[:, 2:] = (r.iloc[:, 2:] * 100).round(2); r.to_csv(ROOT / "results/final_baselines.csv", index=False)
    print(r.pivot(index="model", columns="set", values="f1"))

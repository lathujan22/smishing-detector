"""V2 text classifier with configurable features, resampling and classifier (fit on training data only).

TextModel(cfg).fit(df, y) / .score(df) / .predict(df)
cfg keys
  word:   None | dict(ngram=(1,2), max_features=N, min_df=2)
  char:   None | dict(ngram=(2,5), max_features=N, min_df=3)
  context: bool   - append scaled contextual risk features (URL / shortener / risky domain / phone / money /
                    urgency / impersonation / action request / link risk / operator warning)
  context_weight: float
  noise:  bool    - one noisy copy per training message (typos, look-alike letters, word drop/swap)
  resample: "smote" | "balanced" | "none"
  clf: "svm" | "logreg" | "rf" | "gb"   (gb = sklearn GradientBoosting - NOT XGBoost)
  C: float, calibrate: bool (sigmoid calibration of the SVM, inner 3-fold CV on the training part)
  task: "binary" | "three"
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from src.augment import add_noise, smote, smote_multi
from src.v2.common import SEED

CONTEXT = ["has_url", "url_shortener", "risky_domain", "has_phone", "has_money", "urgency", "impersonation",
           "action_request", "link_risk", "n_urgency_hits", "telco_flag"]
TOKEN = r"<\w+>|[^\s<>.,!?:;()\"']+"
BASE = dict(word=dict(ngram=(1, 2), max_features=10000, min_df=2), char=dict(ngram=(2, 5), max_features=10000, min_df=3),
            context=False, context_weight=1.0, noise=True, resample="smote", clf="svm", C=0.1, calibrate=False, task="binary")


def context_matrix(df: pd.DataFrame) -> np.ndarray:
    X = df[CONTEXT].astype(float).fillna(0).to_numpy()
    X[:, CONTEXT.index("n_urgency_hits")] = np.clip(X[:, CONTEXT.index("n_urgency_hits")] / 3, 0, 1)
    return X


class TextModel:
    def __init__(self, cfg: dict):
        self.cfg = {**BASE, **cfg}

    def _vec(self):
        c, v = self.cfg, []
        if c["word"]:
            w = c["word"]; v.append(TfidfVectorizer(analyzer="word", ngram_range=tuple(w["ngram"]), min_df=w.get("min_df", 2),
                                                     max_features=w["max_features"], sublinear_tf=True, token_pattern=TOKEN))
        if c["char"]:
            h = c["char"]; v.append(TfidfVectorizer(analyzer="char_wb", ngram_range=tuple(h["ngram"]), min_df=h.get("min_df", 3),
                                                     max_features=h["max_features"], sublinear_tf=True))
        return v

    def _features(self, texts, ctx, fit=False):
        mats = [(v.fit_transform(texts) if fit else v.transform(texts)) for v in self.vecs]
        if self.cfg["context"]:
            mats.append(sp.csr_matrix(ctx * self.cfg["context_weight"]))
        return sp.hstack(mats).tocsr()

    def _clf(self, n_classes):
        c = self.cfg; cw = "balanced" if c["resample"] == "balanced" else None
        if c["clf"] == "svm":
            m = LinearSVC(C=c["C"], class_weight=cw, random_state=SEED)
            return CalibratedClassifierCV(m, method="sigmoid", cv=3) if c["calibrate"] else m
        if c["clf"] == "logreg":
            return LogisticRegression(C=c.get("C_lr", 10.0), class_weight=cw, max_iter=3000, random_state=SEED)
        if c["clf"] == "rf":
            return RandomForestClassifier(200, max_features="sqrt", class_weight=cw and "balanced_subsample", n_jobs=-1, random_state=SEED)
        if c["clf"] == "gb":
            return GradientBoostingClassifier(n_estimators=300, learning_rate=0.1, max_depth=4, max_features="sqrt",
                                              subsample=0.8, random_state=SEED)
        raise ValueError(c["clf"])

    def fit(self, df: pd.DataFrame, y: np.ndarray):
        texts, ctx, y = df.text_model.astype(str).tolist(), context_matrix(df), np.asarray(y)
        if self.cfg["noise"]:
            rng = np.random.RandomState(SEED)
            texts = texts + [add_noise(t, rng) for t in texts]; ctx = np.vstack([ctx, ctx]); y = np.concatenate([y, y])
        self.vecs = self._vec()
        X = self._features(texts, ctx, fit=True)
        if self.cfg["resample"] == "smote":
            X, y = (smote(X, y, k=5, seed=SEED) if self.cfg["task"] == "binary" else smote_multi(X, y, k=5, seed=SEED))
        self.clf = self._clf(len(set(y))).fit(X, y)
        return self

    def _X(self, df):
        return self._features(df.text_model.astype(str).tolist(), context_matrix(df))

    def score(self, df):
        X = self._X(df); m = self.clf
        if self.cfg["task"] == "binary":
            if hasattr(m, "predict_proba") and not isinstance(m, LinearSVC):
                return m.predict_proba(X)[:, 1]
            return m.decision_function(X)
        if hasattr(m, "predict_proba") and not isinstance(m, LinearSVC):
            return m.predict_proba(X)
        s = m.decision_function(X); e = np.exp(s - s.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)

    def predict(self, df):
        return self.clf.predict(self._X(df))

    def default_threshold(self):
        """Decision threshold equivalent to the classifier's own predict() (0 for margins, 0.5 for probabilities)."""
        m = self.clf
        return 0.5 if (hasattr(m, "predict_proba") and not isinstance(m, LinearSVC)) else 0.0

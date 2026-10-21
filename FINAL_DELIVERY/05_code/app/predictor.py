"""Inference engine for the web application.

Models (selected in results/v2/MODEL_SELECTION.md):
  * LinearSVC three-class  (best ML, default)  models/v2/three_linearsvc.joblib
  * XLM-R three-class      (best DL, optional) models/v2/xlmr/three_lr2e-05_bs16_len128_seed42  (needs torch + transformers)
  * binary SVM scam score  (reference)         models/v2/svm_original_baseline.joblib

Every message goes through the same preprocessing as training: Unicode normalisation (Sinhala / Tamil joiners kept),
operator-warning removal, masking of URLs / phones / money / numbers, and contextual risk features.
Message text is never logged or stored.
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.preprocess import normalise
from src.v2.registry import risk_features

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["normal", "promotional", "scam"]
LANG_NAME = {"en": "English", "si": "Sinhala", "ta": "Tamil", "si-en": "Singlish (Sinhala + English)",
             "ta-en": "Tanglish (Tamil + English)", "si-ta": "Mixed Sinhala / Tamil", "mixed": "Mixed"}
XLMR_DIR = ROOT / "models/v2/xlmr/three_lr2e-05_bs16_len128_seed42"
MAX_CHARS = 2000


def _softmax(z):
    z = np.asarray(z, float); e = np.exp(z - z.max()); return e / e.sum()


class Predictor:
    def __init__(self, load_xlmr: bool = True):
        t0 = time.time()
        self.svm3 = joblib.load(ROOT / "models/v2/three_linearsvc.joblib")["model"]
        b = joblib.load(ROOT / "models/v2/svm_original_baseline.joblib")
        self.svm2, self.svm2_thr = b["model"], float(b["threshold"])
        self.feature_names = np.concatenate([v.get_feature_names_out() for v in self.svm3.vecs])
        self.selection = json.loads((ROOT / "results/v2/selected_models.json").read_text())
        self.xlmr = self.tok = None; self.xlmr_status = "not loaded"; self.device = "cpu"; self._lock = threading.Lock()
        self.load_seconds = round(time.time() - t0, 2)
        if load_xlmr:
            threading.Thread(target=self._load_xlmr, daemon=True).start()

    # ------------------------------------------------------------ XLM-R (optional)
    def _load_xlmr(self):
        try:
            if not XLMR_DIR.exists():
                self.xlmr_status = "checkpoint not found"; return
            self.xlmr_status = "loading"
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            # CPU by default: the laptop GPU crashed natively under the web server; set SMISH_XLMR_DEVICE=cuda to try the GPU
            want = os.environ.get("SMISH_XLMR_DEVICE", "cpu")
            self.device = "cuda" if (want == "cuda" and torch.cuda.is_available()) else "cpu"
            self.tok = AutoTokenizer.from_pretrained(XLMR_DIR)
            self.xlmr = AutoModelForSequenceClassification.from_pretrained(XLMR_DIR).to(self.device).eval()
            self._torch = torch
            self.xlmr_status = f"ready ({self.device})"
        except Exception as e:                                   # torch / transformers missing, out of memory, ...
            self.xlmr = None; self.xlmr_status = f"unavailable: {type(e).__name__}"

    def _xlmr_predict(self, model_texts: list[str]):
        torch = self._torch; out = []
        with self._lock, torch.no_grad():
            for i in range(0, len(model_texts), 8):                       # small chunks keep memory low
                enc = self.tok(model_texts[i:i + 8], truncation=True, max_length=128, padding=True, return_tensors="pt").to(self.device)
                out.append(torch.softmax(self.xlmr(**enc).logits.float(), -1).cpu().numpy())
        return np.vstack(out)

    # ------------------------------------------------------------ preprocessing + explanation
    @staticmethod
    def prepare(text: str):
        clean, telco = normalise(str(text)[:MAX_CHARS])
        f = risk_features(clean)
        f["telco_flag"] = int(telco or f.get("telco_flag", 0))
        return clean, f

    @staticmethod
    def signals(f: dict) -> list[dict]:
        s = []
        if f["has_url"]: s.append({"signal": "contains a link", "weight": "medium"})
        if f["url_shortener"]: s.append({"signal": "link shortener (hides the real website)", "weight": "high"})
        if f["risky_domain"]: s.append({"signal": "risky / unusual website domain", "weight": "high"})
        if f["impersonation"] >= 0.7: s.append({"signal": "possible brand / authority impersonation", "weight": "high"})
        elif f["impersonation"] > 0: s.append({"signal": "mentions a bank / company / authority", "weight": "low"})
        if f["action_request"]: s.append({"signal": "asks you to act (verify, pay, click, share details)", "weight": "high"})
        if f["urgency"] >= 0.34: s.append({"signal": f"urgency wording ({f['n_urgency_hits']} cues)", "weight": "medium"})
        if f["has_money"]: s.append({"signal": "money amount", "weight": "low"})
        if f["has_phone"]: s.append({"signal": "phone number", "weight": "low"})
        if f["telco_flag"]: s.append({"signal": "mobile operator already marked it as spam", "weight": "high"})
        return s

    def top_ngrams(self, X, cls: int, k: int = 8) -> list[str]:
        row = X.getrow(0); w = self.svm3.clf.coef_[cls]
        n = len(self.feature_names)
        idx = [i for i in row.indices if i < n]
        contrib = sorted(((row[0, i] * w[i], i) for i in idx), reverse=True)
        out, seen = [], set()
        for c, i in contrib:
            name = self.feature_names[i].strip()
            if c <= 0 or len(out) >= k:
                break
            if name and name not in seen:
                seen.add(name); out.append(name)
        return out

    # ------------------------------------------------------------ public API
    def predict(self, texts: list[str], use_xlmr: bool = True) -> list[dict]:
        prep = [self.prepare(t) for t in texts]
        df = pd.DataFrame([f for _, f in prep])
        X = self.svm3._X(df)
        s3 = self.svm3.clf.decision_function(X)
        s2 = self.svm2.score(df)
        px = self._xlmr_predict(df.text_model.tolist()) if (use_xlmr and self.xlmr is not None) else None
        out = []
        for i, (clean, f) in enumerate(prep):
            p = _softmax(s3[i]); c = int(np.argmax(s3[i]))
            r = {"language": {"code": f["language_detected"], "name": LANG_NAME.get(f["language_detected"], f["language_detected"])},
                 "masked_text": f["text_model"],
                 "svm": {"model": "LinearSVC three-class (best ML)", "prediction": CLASSES[c],
                         "scores": {k: round(float(v), 4) for k, v in zip(CLASSES, p)},
                         "score_note": "softmax of SVM margins - a relative score, not a calibrated probability",
                         "evidence_ngrams": self.top_ngrams(X[i], c)},
                 "binary_svm": {"model": "binary SVM (scam vs not)", "scam_margin": round(float(s2[i]), 4),
                                "threshold": round(self.svm2_thr, 4), "flags_scam": bool(s2[i] >= self.svm2_thr)},
                 "signals": self.signals(f),
                 "xlmr": None}
            if px is not None:
                cx = int(np.argmax(px[i]))
                r["xlmr"] = {"model": "XLM-R three-class (best DL)", "prediction": CLASSES[cx],
                             "probabilities": {k: round(float(v), 4) for k, v in zip(CLASSES, px[i])},
                             "probability_note": "softmax output of the fine-tuned model; not calibrated"}
            votes = [r["svm"]["prediction"]] + ([r["xlmr"]["prediction"]] if r["xlmr"] else [])
            r["final"] = {"prediction": r["svm"]["prediction"],
                          "models_agree": len(set(votes)) == 1,
                          "basis": "LinearSVC (default model); XLM-R shown as a second opinion"}
            out.append(r)
        return out

    def info(self) -> dict:
        return {"selected_models": self.selection, "xlmr_status": self.xlmr_status, "svm_load_seconds": self.load_seconds,
                "classes": CLASSES, "max_chars": MAX_CHARS,
                "disclaimer": "Experimental research prototype. Predictions may be wrong; explanations are approximate "
                              "indicators, not proof of malicious intent. Do not use as the only basis for decisions."}

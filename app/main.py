"""SMS smishing detector - FastAPI inference service + research dashboard.

    cd D:\\ladhu\\smishing-detector
    python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
    open http://127.0.0.1:8000

Endpoints: GET /health, POST /predict, POST /predict/batch, GET /model/info, GET /samples, GET /metrics
Privacy: request bodies (message text) are never logged or stored.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.predictor import MAX_CHARS, ROOT, Predictor

logging.getLogger("uvicorn.access").setLevel(logging.WARNING)       # no per-request log lines
APP_DIR = Path(__file__).resolve().parent
VERSION = "v2.0 (three-class LinearSVC default, XLM-R second opinion)"

class UTF8JSON(JSONResponse):
    media_type = "application/json; charset=utf-8"      # explicit charset so every client decodes Sinhala / Tamil correctly


app = FastAPI(title="Multilingual SMS Smishing Detector", version=VERSION, default_response_class=UTF8JSON,
              description="Research prototype - predictions are experimental and may be incorrect.")
predictor = Predictor(load_xlmr=os.environ.get("SMISH_NO_XLMR") != "1")
app.mount("/figures", StaticFiles(directory=ROOT / "results/v2/figures"), name="figures")
app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")


class PredictIn(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_CHARS, description="SMS text")
    use_xlmr: bool = True


class BatchIn(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=100)
    use_xlmr: bool = True


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(APP_DIR / "static/index.html")


@app.get("/health")
def health():
    return {"status": "ok", "version": VERSION, "svm": "ready", "xlmr": predictor.xlmr_status}


@app.post("/predict")
def predict(req: PredictIn):
    if not req.text.strip():
        raise HTTPException(422, "empty message")
    return {"version": VERSION, **predictor.predict([req.text], use_xlmr=req.use_xlmr)[0]}


@app.post("/predict/batch")
def predict_batch(req: BatchIn):
    texts = [t[:MAX_CHARS] for t in req.texts]
    if any(not t.strip() for t in texts):
        raise HTTPException(422, "empty message in batch")
    return {"version": VERSION, "results": predictor.predict(texts, use_xlmr=req.use_xlmr)}


@app.get("/model/info")
def model_info():
    return {"version": VERSION, **predictor.info()}


@app.get("/samples")
def samples():
    return json.loads((APP_DIR / "samples.json").read_text(encoding="utf-8"))


@app.get("/metrics")
def metrics():
    """Measured results (aggregate numbers only, no message text) for the dashboard charts."""
    C = pd.read_csv(ROOT / "results/v2/model_comparison.csv")
    lang = pd.read_csv(ROOT / "results/v2/three_class/by_language.csv")
    lang = lang[(lang.model == "LinearSVC") & lang.set.isin(["held-out test", "real benchmark"])]
    figs = sorted(p.name for p in (ROOT / "results/v2/figures").glob("*.png"))
    clean = lambda d: d.astype(object).where(pd.notna(d), None).to_dict(orient="records")     # NaN -> null (valid JSON)
    return {"comparison": clean(C), "linearsvc_by_language": clean(lang), "figures": figs}

#!/usr/bin/env python3
"""
FastAPI REST API for the Misinformation Detection System.
Run with: python api.py
"""

import os
import sys
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.insert(0, ".")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from src.detector import MisinformationDetector  # noqa: E402

app = FastAPI(title="Misinformation Detection API", version="1.0")
detector = None

if os.path.exists("frontend"):
    app.mount("/app", StaticFiles(directory="frontend", html=True), name="frontend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    global detector
    print("Loading misinformation detector...")
    detector = MisinformationDetector(config="config.yaml", fast_mode=False)
    print("Detector ready")


class PredictRequest(BaseModel):
    text: str
    url: Optional[str] = None
    explain: Optional[bool] = False


class PredictResponse(BaseModel):
    verdict: str
    ensemble_probability: float
    confidence_percent: int
    model_breakdown: dict
    source_credibility: Optional[dict] = None
    explanation: Optional[dict] = None
    error: Optional[str] = None


@app.get("/health")
def health():
    return {
        "status": "ok",
        "detector": detector is not None,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if detector is None:
        raise HTTPException(status_code=503, detail="Detector not ready")

    if not request.text or len(request.text.strip()) < 5:
        raise HTTPException(status_code=400, detail="Text must be at least 5 characters")

    try:
        result = detector.predict(request.text, url=request.url, explain=request.explain)
        verdict = result.get("verdict")
        if verdict is None:
            verdict = result.get("crisp_label", "UNKNOWN")

        return PredictResponse(
            verdict=verdict,
            ensemble_probability=float(result.get("ensemble_probability", 0.5)),
            confidence_percent=int(abs(result.get("ensemble_probability", 0.5) - 0.5) * 200),
            model_breakdown=result.get("model_breakdown", {}),
            source_credibility=result.get("source_credibility"),
            explanation=result.get("explanation"),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/domains")
def get_domains():
    import json

    db_path = "data/domain_reputation.json"
    if os.path.exists(db_path):
        with open(db_path, encoding="utf-8") as f:
            db = json.load(f)
        return {"total": len(db), "sample": dict(list(sorted(db.items(), key=lambda x: -x[1]))[:10])}
    return {"total": 0, "sample": {}}


if __name__ == "__main__":
    # Local default: 8000
    # HuggingFace Spaces (Docker): 7860 unless PORT is explicitly set.
    default_port = 7860 if os.getenv("SPACE_ID") else 8000
    port = int(os.getenv("PORT", str(default_port)))
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=False)

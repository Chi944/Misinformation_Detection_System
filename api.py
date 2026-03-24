#!/usr/bin/env python3
"""
FastAPI REST API for the Misinformation Detection System.
Run with: python api.py
"""

import os
import re
import sys
from typing import Any, Optional

import requests
import uvicorn
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.insert(0, ".")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from src.detector import MisinformationDetector  # noqa: E402

app = FastAPI(title="Misinformation Detection API", version="1.0")
detector = None

FRONTEND_DIST = os.path.join("frontend", "dist")
FRONTEND_BUILD = os.path.join("frontend", "build")
FRONTEND_DIR = FRONTEND_DIST if os.path.exists(FRONTEND_DIST) else (
    FRONTEND_BUILD if os.path.exists(FRONTEND_BUILD) else None
)

# Explicit app UI mount for both local and hosted usage.
if os.path.exists(FRONTEND_DIST):
    app.mount("/app", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")

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
    llm_judge: Optional[dict] = None
    text: Optional[str] = None
    error: Optional[str] = None


class ScrapeRequest(BaseModel):
    url: str
    explain: Optional[bool] = True


class ScrapePredictResponse(PredictResponse):
    scraped_url: str
    scraped_preview: str
    scraped_word_count: int
    scraped_char_count: int


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
            llm_judge=result.get("llm_judge"),
            text=request.text,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _clean_scraped_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript", "nav", "footer", "header", "aside", "form"]):
        tag.decompose()
    text = soup.get_text(separator=" ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


@app.post("/scrape-and-predict", response_model=ScrapePredictResponse)
async def scrape_and_predict(request: ScrapeRequest):
    if detector is None:
        raise HTTPException(status_code=503, detail="Detector not ready")
    if not request.url:
        raise HTTPException(status_code=400, detail="URL is required")

    try:
        resp = requests.get(
            request.url,
            timeout=20,
            headers={"User-Agent": "Mozilla/5.0 (compatible; MisinformationDetector/1.0)"},
        )
        resp.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch URL: {e}")

    text = _clean_scraped_text(resp.text)
    if len(text) < 20:
        raise HTTPException(status_code=400, detail="Could not extract enough text from URL")

    words = text.split()
    word_count = len(words)
    char_count = len(text)
    preview = text[:400] + ("..." if len(text) > 400 else "")

    try:
        result = detector.predict(text, url=request.url, explain=bool(request.explain))
        verdict = result.get("verdict") or result.get("crisp_label", "UNKNOWN")
        return ScrapePredictResponse(
            verdict=verdict,
            ensemble_probability=float(result.get("ensemble_probability", 0.5)),
            confidence_percent=int(abs(result.get("ensemble_probability", 0.5) - 0.5) * 200),
            model_breakdown=result.get("model_breakdown", {}),
            source_credibility=result.get("source_credibility"),
            explanation=result.get("explanation"),
            llm_judge=result.get("llm_judge"),
            text=text,
            scraped_url=request.url,
            scraped_preview=preview,
            scraped_word_count=word_count,
            scraped_char_count=char_count,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/llm-status")
async def llm_status():
    available = bool(detector is not None and getattr(detector, "llm_judge", None) is not None)
    return {"llm_available": available}


@app.get("/domains")
def get_domains():
    import json

    db_path = "data/domain_reputation.json"
    if os.path.exists(db_path):
        with open(db_path, encoding="utf-8") as f:
            db = json.load(f)
        return {"total": len(db), "sample": dict(list(sorted(db.items(), key=lambda x: -x[1]))[:10])}
    return {"total": 0, "sample": {}}

@app.get("/", response_class=HTMLResponse)
def root():
    """Serve built frontend index.html, or fallback form when not built."""
    if FRONTEND_DIR is not None:
        index_path = os.path.join(FRONTEND_DIR, "index.html")
        if os.path.exists(index_path):
            return FileResponse(index_path)

    fallback_html = """<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Misinformation Detector API</title>
    <style>
      body { font-family: Arial, sans-serif; max-width: 820px; margin: 24px auto; padding: 0 16px; }
      textarea,input,button { width: 100%; margin: 8px 0; padding: 10px; }
      pre { background: #f5f5f5; padding: 12px; overflow: auto; white-space: pre-wrap; }
    </style>
  </head>
  <body>
    <h2>Misinformation Detector</h2>
    <p>Built frontend not found. Using fallback test form.</p>
    <label>Text</label>
    <textarea id="text" rows="6" placeholder="Enter text to analyze"></textarea>
    <label>URL (optional)</label>
    <input id="url" placeholder="https://example.com/article" />
    <label><input type="checkbox" id="explain" checked /> explain</label>
    <button onclick="runPredict()">Submit to /predict</button>
    <pre id="out"></pre>
    <script>
      async function runPredict() {
        const text = document.getElementById('text').value;
        const url = document.getElementById('url').value;
        const explain = document.getElementById('explain').checked;
        const body = { text, explain };
        if (url) body.url = url;
        try {
          const r = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
          });
          const data = await r.json();
          document.getElementById('out').textContent = JSON.stringify(data, null, 2);
        } catch (e) {
          document.getElementById('out').textContent = String(e);
        }
      }
    </script>
  </body>
</html>"""
    return HTMLResponse(content=fallback_html, status_code=200)

# Mount built frontend at "/" for production static serving.
# Kept after API routes so /health, /predict, /domains remain available.
if FRONTEND_DIR is not None:
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend_root")


if __name__ == "__main__":
    # Local default: 8000
    # HuggingFace Spaces (Docker): 7860 unless PORT is explicitly set.
    default_port = 7860 if os.getenv("SPACE_ID") else 8000
    port = int(os.getenv("PORT", str(default_port)))
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=False)

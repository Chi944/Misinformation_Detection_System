"""
Inference API module for misinformation detection.
Provides a Flask REST API for model predictions, including:
- plain text input
- URL input (fetch and extract title/description)
"""

import re
import time
from html import unescape
from typing import Any, Dict, Optional

import pandas as pd
import requests
from flask import Flask, jsonify, request
from flask_cors import CORS

from src.config import LABEL_NAMES, MAX_INFERENCE_LATENCY_MS, MODELS_DIR

app = Flask(__name__)
# Allow browser-based frontends (React, etc.) to call the API
CORS(app, resources={r"/*": {"origins": "*"}})

_models: Dict[str, Any] = {}


def load_models():
    """Load the single TF-IDF + Logistic Regression model for inference."""
    global _models
    _models = {}
    tfidf_path = MODELS_DIR / "tfidf_logistic.pkl"
    if tfidf_path.exists():
        from src.traditional_ml import TfidfLogisticClassifier

        model = TfidfLogisticClassifier()
        model.load(tfidf_path)
        _models["tfidf_logistic"] = model
        print("Loaded TF-IDF + Logistic Regression model")
    else:
        print("Warning: No model found. Train with: python main.py --train")
    return _models


class InferenceEngine:
    """Inference engine using the single TF-IDF + Logistic Regression model."""

    def __init__(self, models: Dict[str, Any] = None):
        self.models = models or _models
        self.default_model = "tfidf_logistic"

    def predict(
        self, text: str, include_explanation: bool = False, url: Optional[str] = None
    ) -> Dict[str, Any]:
        """Make a prediction on input text using the single TF-IDF + LR model."""
        if not self.models:
            raise ValueError("No model loaded. Train with: python main.py --train")
        model = self.models[self.default_model]

        start_time = time.time()
        proba = model.predict_proba(pd.Series([text]))[0]
        pred = proba.argmax()

        latency_ms = (time.time() - start_time) * 1000

        result = {
            "text": text[:500] + "..." if len(text) > 500 else text,
            "prediction": LABEL_NAMES[pred],
            "label": int(pred),
            "confidence": float(proba[pred]),
            "probabilities": {LABEL_NAMES[i]: float(p) for i, p in enumerate(proba)},
            "model": self.default_model,
            "latency_ms": round(latency_ms, 2),
            "within_latency_constraint": latency_ms <= MAX_INFERENCE_LATENCY_MS,
        }

        # Add explanation if requested (needed for flagged terms in credibility audit)
        explanation_features = None
        if include_explanation:
            from src.explainability import ModelExplainer

            explainer = ModelExplainer()
            explanation = explainer.explain_with_lime(
                text, lambda x: model.predict_proba(pd.Series(x)), num_features=15, num_samples=200
            )
            result["explanation"] = {"top_features": explanation["feature_weights"]}
            explanation_features = explanation["feature_weights"]

        # Credibility audit: Sensationalism, Political Bias, Source Credibility,
        # Factuality Index, Flagged Terms
        misinfo_prob = float(proba[1]) if len(proba) > 1 else 0.0
        result["credibility_audit"] = self._run_credibility_audit(
            text,
            url=url,
            misinformation_prob=misinfo_prob,
            explanation_features=explanation_features,
            predicted_label=int(pred),
        )

        return result

    def _run_credibility_audit(
        self,
        text: str,
        url: Optional[str] = None,
        misinformation_prob: float = 0.0,
        explanation_features: Optional[list] = None,
        predicted_label: int = 1,
    ) -> Dict[str, Any]:
        """Run credibility audit using src.credibility_audit."""
        from src.credibility_audit import run_credibility_audit

        return run_credibility_audit(
            text=text,
            url=url,
            misinformation_prob=misinformation_prob,
            explanation_features=explanation_features,
            predicted_label=predicted_label,
        )

    def predict_batch(self, texts: list) -> list:
        """Make predictions on multiple texts using the single model."""
        return [self.predict(text) for text in texts]

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {"model": self.default_model, "latency_constraint_ms": MAX_INFERENCE_LATENCY_MS}


def _extract_text_from_html(html: str) -> str:
    """
    Extract text for comprehensive credibility audit.
    Prioritises:
      - meta og:title, title, og:description, description
      - <article>, <main>, <[role="main"]> content
      - <p> paragraphs (full body scan)
    """
    raw = re.sub(r"\s+", " ", html)

    def _find_meta(pattern: str) -> Optional[str]:
        m = re.search(
            rf'<meta[^>]+{pattern}[^>]+content=["\']([^"\']+)["\']',
            raw,
            flags=re.IGNORECASE,
        )
        return unescape(m.group(1)).strip() if m else None

    # Title candidates
    title = _find_meta(r'(property=["\']og:title["\']|name=["\']title["\'])')
    if not title:
        m_title = re.search(r"<title[^>]*>(.*?)</title>", raw, flags=re.IGNORECASE)
        if m_title:
            title = unescape(m_title.group(1)).strip()

    desc = _find_meta(r'(name=["\']description["\']|property=["\']og:description["\'])')

    parts = [p for p in [title, desc] if p]

    # Full article body: article, main, p tags
    body_text = []
    for tag in ("article", "main", '[role="main"]', "div.article-body", "div.post-content"):
        if tag.startswith("["):
            # attribute selector – use different regex
            m = re.search(
                r'<[^>]+\s+role=["\']main["\'][^>]*>(.*?)</\w+>',
                raw,
                flags=re.IGNORECASE | re.DOTALL,
            )
        elif "." in tag:
            # class selector
            cls = tag.split(".", 1)[1]
            m = re.search(
                rf'<div[^>]*class=["\'][^"\']*{re.escape(cls)}[^"\']*["\'][^>]*>(.*?)</div>',
                raw,
                flags=re.IGNORECASE | re.DOTALL,
            )
        else:
            m = re.search(
                rf"<{tag}[^>]*>(.*?)</{tag}>",
                raw,
                flags=re.IGNORECASE | re.DOTALL,
            )
        if m:
            frag = re.sub(r"<[^>]+>", " ", m.group(1))
            frag = unescape(frag).strip()
            if len(frag) > 50:
                body_text.append(frag)

    # Fallback: all <p> paragraphs (common article structure)
    if not body_text:
        for m in re.finditer(r"<p[^>]*>(.*?)</p>", raw, flags=re.IGNORECASE | re.DOTALL):
            frag = re.sub(r"<[^>]+>", " ", m.group(1))
            frag = unescape(frag).strip()
            if len(frag) > 20:
                body_text.append(frag)

    body = " ".join(body_text[:30])[:8000] if body_text else ""  # cap length
    if body:
        parts.append(body)

    return " ".join(p for p in parts if p) if parts else ""


def _build_text_from_url(url: str, header: Optional[str] = None) -> str:
    """
    Fetch a URL and build a text string suitable for the classifier.

    If we cannot fetch or parse anything, we fall back to just the header.
    """
    text_parts = []
    if header:
        text_parts.append(str(header).strip())

    try:
        resp = requests.get(
            url,
            timeout=5,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0 Safari/537.36"
                )
            },
        )
        if resp.ok and "text/html" in resp.headers.get("Content-Type", ""):
            extracted = _extract_text_from_html(resp.text)
            if extracted:
                text_parts.append(extracted)
    except Exception:
        # We intentionally swallow network/parser errors to keep the API robust.
        pass

    combined = " ".join(text_parts).strip()
    return combined


# Flask API routes


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "model_loaded": len(_models) > 0})


@app.route("/models", methods=["GET"])
def get_models():
    """Get information about the loaded model."""
    engine = InferenceEngine()
    return jsonify(engine.get_model_info())


@app.route("/predict", methods=["POST"])
def predict():
    """
    Make a prediction on input text.
    Request body: text (optional), url (optional), header (optional), include_explanation (optional).
    """
    data = request.get_json() or {}

    text = data.get("text")
    url = data.get("url")
    header = data.get("header")

    if not text and url:
        text = _build_text_from_url(str(url), header=str(header) if header else None)

    if not text or not str(text).strip():
        return jsonify({"error": 'Provide either non-empty "text" or a valid "url".'}), 400

    text = str(text)
    include_explanation = data.get("include_explanation", True)
    try:
        engine = InferenceEngine()
        result = engine.predict(text, include_explanation=include_explanation, url=url or None)
        return jsonify(result)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    """Make predictions on multiple texts. Request body: texts (required)."""
    data = request.get_json()

    if not data or "texts" not in data:
        return jsonify({"error": "Missing required field: texts"}), 400

    texts = data["texts"]
    if not isinstance(texts, list):
        return jsonify({"error": "texts must be a list"}), 400
    try:
        engine = InferenceEngine()
        results = engine.predict_batch(texts)
        return jsonify({"predictions": results})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Batch prediction failed: {str(e)}"}), 500


@app.route("/explain", methods=["POST"])
def explain():
    """Get detailed explanation for a prediction. Request body: text (required)."""
    data = request.get_json()

    if not data or "text" not in data:
        return jsonify({"error": "Missing required field: text"}), 400

    text = data["text"]
    try:
        engine = InferenceEngine()
        result = engine.predict(text, include_explanation=True)
        return jsonify(result)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Explanation failed: {str(e)}"}), 500


def run_api(host: str = "0.0.0.0", port: int = 5000, debug: bool = False):
    """
    Run the Flask API server.

    Args:
        host: Host address
        port: Port number
        debug: Debug mode
    """
    load_models()
    print(f"\nStarting API server at http://{host}:{port}")
    print("\nAvailable endpoints:")
    print("  GET  /health           - Health check")
    print("  GET  /models           - Get model information")
    print("  POST /predict          - Single prediction")
    print("  POST /predict/batch    - Batch predictions")
    print("  POST /explain          - Prediction with explanation")
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    run_api(debug=True)

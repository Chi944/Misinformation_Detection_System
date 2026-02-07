"""
Inference API module for misinformation detection.
Provides a Flask REST API for model predictions, including:
- plain text input
- URL input (fetch and extract title/description)
"""

import re
import time
from html import unescape
from typing import Dict, Any, Optional

import pandas as pd
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

from src.config import MODELS_DIR, LABEL_NAMES, MAX_INFERENCE_LATENCY_MS


app = Flask(__name__)
# Allow browser-based frontends (React, etc.) to call the API
CORS(app, resources={r"/*": {"origins": "*"}})

_models: Dict[str, Any] = {}


def load_models():
    """Load all trained models for inference."""
    global _models
    
    # Try to load TF-IDF + Logistic Regression model
    tfidf_path = MODELS_DIR / 'tfidf_logistic.pkl'
    if tfidf_path.exists():
        from src.traditional_ml import TfidfLogisticClassifier
        model = TfidfLogisticClassifier()
        model.load(tfidf_path)
        _models['tfidf_logistic'] = model
        print(f"Loaded TF-IDF + Logistic Regression model")
    
    # Try to load Naive Bayes model
    nb_path = MODELS_DIR / 'naive_bayes.pkl'
    if nb_path.exists():
        from src.traditional_ml import NaiveBayesClassifier
        model = NaiveBayesClassifier()
        model.load(nb_path)
        _models['naive_bayes'] = model
        print(f"Loaded Naive Bayes model")
    
    # Try to load BERT model
    bert_path = MODELS_DIR / 'bert_model'
    if bert_path.exists():
        try:
            from src.deep_learning import BertClassifier
            model = BertClassifier()
            model.load(bert_path)
            _models['bert'] = model
            print(f"Loaded BERT model")
        except Exception as e:
            print(f"Warning: Could not load BERT model: {e}")
    
    if not _models:
        print("Warning: No models found. Train models first.")
    
    return _models


class InferenceEngine:
    """Unified inference engine for all models."""
    
    def __init__(self, models: Dict[str, Any] = None):
        self.models = models or _models
        self.default_model = 'tfidf_logistic'
    
    def predict(self, text: str, model_name: Optional[str] = None,
                include_explanation: bool = False, url: Optional[str] = None) -> Dict[str, Any]:
        """
        Make a prediction on input text.
        
        Args:
            text: Input text to classify
            model_name: Name of the model to use
            include_explanation: Whether to include explanation
            
        Returns:
            Prediction result dictionary
        """
        model_name = model_name or self.default_model
        
        if model_name not in self.models:
            available = list(self.models.keys())
            raise ValueError(f"Model '{model_name}' not found. Available: {available}")
        
        model = self.models[model_name]
        
        start_time = time.time()
        
        # Get prediction
        if model_name == 'bert':
            proba = model.predict_proba([text])[0]
            pred = proba.argmax()
        else:
            proba = model.predict_proba(pd.Series([text]))[0]
            pred = proba.argmax()
        
        latency_ms = (time.time() - start_time) * 1000
        
        result = {
            'text': text[:500] + '...' if len(text) > 500 else text,
            'prediction': LABEL_NAMES[pred],
            'label': int(pred),
            'confidence': float(proba[pred]),
            'probabilities': {
                LABEL_NAMES[i]: float(p) for i, p in enumerate(proba)
            },
            'model': model_name,
            'latency_ms': round(latency_ms, 2),
            'within_latency_constraint': latency_ms <= MAX_INFERENCE_LATENCY_MS
        }
        
        # Add explanation if requested (needed for flagged terms in credibility audit)
        explanation_features = None
        if include_explanation and model_name != 'bert':
            from src.explainability import ModelExplainer
            explainer = ModelExplainer()
            explanation = explainer.explain_with_lime(
                text,
                lambda x: model.predict_proba(pd.Series(x)),
                num_features=15,
                num_samples=200
            )
            result['explanation'] = {
                'top_features': explanation['feature_weights']
            }
            explanation_features = explanation['feature_weights']

        # Credibility audit: Sensationalism, Political Bias, Source Credibility,
        # Factuality Index, Flagged Terms
        misinfo_prob = float(proba[1]) if len(proba) > 1 else 0.0
        result['credibility_audit'] = self._run_credibility_audit(
            text, url=url, misinformation_prob=misinfo_prob,
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
    
    def predict_batch(self, texts: list, model_name: Optional[str] = None) -> list:
        """
        Make predictions on multiple texts.
        
        Args:
            texts: List of input texts
            model_name: Name of the model to use
            
        Returns:
            List of prediction results
        """
        return [self.predict(text, model_name) for text in texts]
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models."""
        info = {
            'available_models': list(self.models.keys()),
            'default_model': self.default_model,
            'latency_constraint_ms': MAX_INFERENCE_LATENCY_MS
        }
        return info


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

    desc = _find_meta(
        r'(name=["\']description["\']|property=["\']og:description["\'])'
    )

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

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'models_loaded': len(_models) > 0,
        'available_models': list(_models.keys())
    })


@app.route('/models', methods=['GET'])
def get_models():
    """Get information about available models."""
    engine = InferenceEngine()
    return jsonify(engine.get_model_info())


@app.route('/predict', methods=['POST'])
def predict():
    """
    Make a prediction on input text.
    
    Request body:
        - text: Input text to classify (optional)
        - url: URL to fetch and analyse (optional)
        - header: Optional article/post header/headline (used with url)
        - model: Model name to use (optional)
        - include_explanation: Whether to include explanation (optional)
    """
    data = request.get_json() or {}
    
    text = data.get('text')
    url = data.get('url')
    header = data.get('header')

    if not text and url:
        text = _build_text_from_url(str(url), header=str(header) if header else None)

    if not text or not str(text).strip():
        return jsonify({'error': 'Provide either non-empty \"text\" or a valid \"url\".'}), 400
    
    text = str(text)
    model_name = data.get('model')
    # Credibility audit requires LIME; include explanation when model supports it
    include_explanation = data.get('include_explanation', True)
    if model_name == 'bert':
        include_explanation = False

    try:
        engine = InferenceEngine()
        result = engine.predict(text, model_name, include_explanation, url=url or None)
        return jsonify(result)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """
    Make predictions on multiple texts.
    
    Request body:
        - texts: List of input texts (required)
        - model: Model name to use (optional)
    """
    data = request.get_json()
    
    if not data or 'texts' not in data:
        return jsonify({'error': 'Missing required field: texts'}), 400
    
    texts = data['texts']
    model_name = data.get('model')
    
    if not isinstance(texts, list):
        return jsonify({'error': 'texts must be a list'}), 400
    
    try:
        engine = InferenceEngine()
        results = engine.predict_batch(texts, model_name)
        return jsonify({'predictions': results})
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f'Batch prediction failed: {str(e)}'}), 500


@app.route('/explain', methods=['POST'])
def explain():
    """
    Get detailed explanation for a prediction.
    
    Request body:
        - text: Input text to explain (required)
        - model: Model name to use (optional)
    """
    data = request.get_json()
    
    if not data or 'text' not in data:
        return jsonify({'error': 'Missing required field: text'}), 400
    
    text = data['text']
    model_name = data.get('model', 'tfidf_logistic')
    
    try:
        engine = InferenceEngine()
        result = engine.predict(text, model_name, include_explanation=True)
        return jsonify(result)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f'Explanation failed: {str(e)}'}), 500


def run_api(host: str = '0.0.0.0', port: int = 5000, debug: bool = False):
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

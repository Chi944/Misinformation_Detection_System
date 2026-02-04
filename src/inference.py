"""
Inference API module for misinformation detection.
Provides a Flask REST API for model predictions.
"""

import time
from typing import Dict, Any, Optional
from pathlib import Path

from flask import Flask, request, jsonify
import pandas as pd

from src.config import MODELS_DIR, LABEL_NAMES, MAX_INFERENCE_LATENCY_MS


app = Flask(__name__)

_models = {}


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
        from src.deep_learning import BertClassifier
        model = BertClassifier()
        model.load(bert_path)
        _models['bert'] = model
        print(f"Loaded BERT model")
    
    if not _models:
        print("Warning: No models found. Train models first.")
    
    return _models


class InferenceEngine:
    """Unified inference engine for all models."""
    
    def __init__(self, models: Dict[str, Any] = None):
        self.models = models or _models
        self.default_model = 'tfidf_logistic'
    
    def predict(self, text: str, model_name: Optional[str] = None,
                include_explanation: bool = False) -> Dict[str, Any]:
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
        
        # Add explanation if requested
        if include_explanation and model_name != 'bert':
            from src.explainability import ModelExplainer
            explainer = ModelExplainer()
            explanation = explainer.explain_with_lime(
                text,
                lambda x: model.predict_proba(pd.Series(x)),
                num_features=5,
                num_samples=100
            )
            result['explanation'] = {
                'top_features': explanation['feature_weights']
            }
        
        return result
    
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
        - text: Input text to classify (required)
        - model: Model name to use (optional)
        - include_explanation: Whether to include explanation (optional)
    """
    data = request.get_json()
    
    if not data or 'text' not in data:
        return jsonify({'error': 'Missing required field: text'}), 400
    
    text = data['text']
    model_name = data.get('model')
    include_explanation = data.get('include_explanation', False)
    
    try:
        engine = InferenceEngine()
        result = engine.predict(text, model_name, include_explanation)
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

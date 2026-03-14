"""Ensemble detector for misinformation.

This module combines the three base models (BERT, TF-IDF DNN, Naive Bayes)
using weighted soft voting. It produces a structured prediction dictionary
containing per-model probabilities, the ensemble decision, and a simple
agreement score.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from src.models.bert_classifier import BERTMisinformationClassifier
from src.models.naive_bayes_model import TFNaiveBayesWrapper
from src.models.tfidf_model import TFIDFModel


@dataclass
class EnsembleWeights:
    """Weights for the three base models."""

    bert: float = 0.50
    tfidf: float = 0.30
    naive_bayes: float = 0.20

    def as_array(self) -> np.ndarray:
        w = np.asarray([self.bert, self.tfidf, self.naive_bayes], dtype="float32")
        s = float(w.sum())
        return w / s if s > 0 else w


class EnsembleDetector:
    """Weighted soft-voting ensemble over three base models.

    Args:
        bert_model: Trained :class:`BERTMisinformationClassifier` instance.
        tfidf_model: Trained :class:`TfidfDNNClassifier` instance.
        nb_model: Trained :class:`TFNaiveBayesWrapper` instance.
        weights: Optional :class:`EnsembleWeights` specifying contribution of
            each base model.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        bert_model: Optional[BERTMisinformationClassifier] = None,
        tfidf_model: Optional[TFIDFModel] = None,
        nb_model: Optional[TFNaiveBayesWrapper] = None,
        weights: Optional[EnsembleWeights] = None,
        bert_tokenizer=None,
        nb_vectorizer=None,
        device: Optional[str] = None,
    ) -> None:
        self.config = config or {}
        self.bert_model = bert_model
        self.tfidf_model = tfidf_model
        self.nb_model = nb_model
        self.weights = weights or EnsembleWeights()
        self.bert_tokenizer = bert_tokenizer
        self.nb_vectorizer = nb_vectorizer
        self.device = device or "cpu"

    def _get_active_weights(self) -> Dict[str, float]:
        """
        Return weights for only the models that are loaded.
        Redistributes proportionally when models are missing.
        """
        base = {
            "bert": 0.50,
            "tfidf": 0.30,
            "naive_bayes": 0.20,
        }
        active: Dict[str, float] = {}
        if self.bert_model is not None and self.bert_tokenizer is not None:
            active["bert"] = base["bert"]
        if self.tfidf_model is not None:
            active["tfidf"] = base["tfidf"]
        if self.nb_model is not None:
            active["naive_bayes"] = base["naive_bayes"]
        if not active:
            return base
        total = sum(active.values())
        return {k: v / total for k, v in active.items()}

    # ----------------------------------------------------------------- predict
    def predict(self, text: str) -> Dict[str, Any]:
        """Run all three models on a single text and combine outputs.

        Returns:
            dict with keys:

            - ``bert_proba``, ``tfidf_proba``, ``naive_bayes_proba``:
              probabilities ``[p0, p1]`` for each model.
            - ``ensemble_probability``: Probability of class 1 (misinformation).
            - ``crisp_label``: ``"credible"`` or ``"misinformation"``.
            - ``model_agreement``: Fraction of models that agree with ensemble
              label.
            - ``model_breakdown``: Per-model label and confidence.
            - ``ensemble_weights``: Current weights as a mapping.
        """

        # Ensure we always pass strings; avoid passing any sparse/dense arrays
        text = str(text) if not isinstance(text, str) else text
        texts = [text]

        bert_p = np.array([[0.5, 0.5]], dtype="float32")
        if self.bert_model is not None and self.bert_tokenizer is not None:
            try:
                import torch
                enc = self.bert_tokenizer(
                    text,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt",
                )
                ids = enc["input_ids"].to(self.device)
                mask = enc["attention_mask"].to(self.device)
                with torch.no_grad():
                    out = self.bert_model.model(input_ids=ids, attention_mask=mask).logits
                    probs = torch.softmax(out, dim=-1).cpu().numpy()
                bert_p = np.asarray(probs, dtype="float32")
            except Exception:
                pass

        if self.tfidf_model is not None:
            tfidf_p = self.tfidf_model.predict_proba(texts)
        else:
            tfidf_p = np.array([[0.5, 0.5]], dtype="float32")

        if self.nb_model is not None:
            nb_p = self.nb_model.predict_proba_np(texts)
        else:
            nb_p = np.array([[0.5, 0.5]], dtype="float32")

        p1_bert = float(bert_p[0, 1])
        p1_tfidf_raw = float(tfidf_p[0, 1])
        # Optional TF-IDF threshold calibration (config models.tfidf.threshold)
        t = float(self.config.get("models", {}).get("tfidf", {}).get("threshold", 0.5))
        if t <= 0 or t >= 1:
            t = 0.5
        if abs(t - 0.5) < 1e-6:
            p1_tfidf = p1_tfidf_raw
        else:
            if p1_tfidf_raw <= t:
                p1_tfidf = 0.5 * p1_tfidf_raw / t if t > 0 else 0.5
            else:
                p1_tfidf = 0.5 + 0.5 * (p1_tfidf_raw - t) / (1.0 - t) if t < 1 else 0.5
            p1_tfidf = max(0.0, min(1.0, p1_tfidf))
        p1_nb = float(nb_p[0, 1])

        model_breakdown = {
            "bert": {"label": int(p1_bert >= 0.5), "confidence": p1_bert},
            "tfidf": {"label": int(p1_tfidf >= 0.5), "confidence": p1_tfidf},
            "naive_bayes": {"label": int(p1_nb >= 0.5), "confidence": p1_nb},
        }

        weights = self._get_active_weights()
        p1_ensemble = float(
            sum(weights.get(name, 0.0) * model_breakdown[name]["confidence"] for name in weights)
        )

        crisp_label = "misinformation" if p1_ensemble >= 0.5 else "credible"

        def label_from_p1(p1: float) -> str:
            return "misinformation" if p1 >= 0.5 else "credible"

        labels = [
            label_from_p1(p1_bert),
            label_from_p1(p1_tfidf),
            label_from_p1(p1_nb),
        ]
        agreement = float(sum(label == crisp_label for label in labels) / len(labels))

        weights_dict = {k: float(v) for k, v in weights.items()}

        return {
            "bert_proba": bert_p[0].tolist(),
            "tfidf_proba": tfidf_p[0].tolist(),
            "naive_bayes_proba": nb_p[0].tolist(),
            "ensemble_probability": p1_ensemble,
            "crisp_label": crisp_label,
            "model_agreement": agreement,
            "agreement": agreement,
            "model_breakdown": model_breakdown,
            "ensemble_weights": weights_dict,
        }

    # ----------------------------------------------------------- weight updates
    def recalibrate_weights(self, eval_results: Dict[str, Dict[str, Any]]) -> None:
        """Recompute ensemble weights proportional to per-model F1 scores.

        Args:
            eval_results: Mapping with keys ``"bert"``, ``"tfidf"``,
                ``"naive_bayes"`` whose values contain an ``"f1"`` field.
        """

        f1_bert = float(eval_results.get("bert", {}).get("f1", 0.0))
        f1_tfidf = float(eval_results.get("tfidf", {}).get("f1", 0.0))
        f1_nb = float(eval_results.get("naive_bayes", {}).get("f1", 0.0))

        f1s = np.asarray([f1_bert, f1_tfidf, f1_nb], dtype="float32")
        total = float(f1s.sum())
        if total <= 0:
            return

        norm = f1s / total
        self.weights.bert = float(norm[0])
        self.weights.tfidf = float(norm[1])
        self.weights.naive_bayes = float(norm[2])

"""Tests for the ensemble detector.

These tests validate that the `EnsembleDetector` prediction dictionary has the
expected shape and that weight recalibration behaves as intended.
"""

import numpy as np

from src.models.ensemble_detector import EnsembleDetector, EnsembleWeights
from src.models.naive_bayes_model import TFNaiveBayesWrapper
from src.models.tfidf_model import TfidfDNNClassifier


class _DummyModel:
    def __init__(self, p: float) -> None:
        self._p = p

    def predict_proba(self, texts):
        return np.array([[1.0 - self._p, self._p] for _ in texts], dtype="float32")


def test_ensemble_predict_structure(monkeypatch):
    """predict() should return all required keys and valid probabilities."""

    bert = _DummyModel(0.6)
    tfidf = _DummyModel(0.7)
    nb = TFNaiveBayesWrapper(config={})

    # Monkeypatch NB wrapper to behave like our dummy for this test.
    def _fake_predict_proba_np(texts):
        return np.array([[0.3, 0.7] for _ in texts], dtype="float32")

    monkeypatch.setattr(nb, "predict_proba_np", _fake_predict_proba_np)

    ensemble = EnsembleDetector(
        bert_model=bert,  # type: ignore[arg-type]
        tfidf_model=tfidf,  # type: ignore[arg-type]
        nb_model=nb,
        weights=EnsembleWeights(),
    )

    out = ensemble.predict("example text")
    assert "ensemble_probability" in out
    assert "crisp_label" in out
    assert "agreement" in out
    assert "model_breakdown" in out
    assert "ensemble_weights" in out
    assert 0.0 <= out["ensemble_probability"] <= 1.0


def test_recalibrate_weights_sums_to_one():
    """recalibrate_weights() should produce weights that sum to ~1."""

    bert = _DummyModel(0.5)
    tfidf = _DummyModel(0.5)
    nb = TFNaiveBayesWrapper(config={})

    def _fake_predict_proba_np(texts):
        return np.array([[0.5, 0.5] for _ in texts], dtype="float32")

    nb.predict_proba_np = _fake_predict_proba_np  # type: ignore[assignment]

    ensemble = EnsembleDetector(
        bert_model=bert,  # type: ignore[arg-type]
        tfidf_model=tfidf,  # type: ignore[arg-type]
        nb_model=nb,
        weights=EnsembleWeights(),
    )

    ensemble.recalibrate_weights(
        {
            "bert": {"f1": 0.8},
            "tfidf": {"f1": 0.6},
            "naive_bayes": {"f1": 0.4},
        }
    )

    total = ensemble.weights.bert + ensemble.weights.tfidf + ensemble.weights.naive_bayes
    assert abs(total - 1.0) < 1e-6


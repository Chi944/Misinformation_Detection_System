"""Skeleton tests for the fuzzy engine.

These tests validate that the fuzzy engine imports correctly and that
`compute` returns a score in the expected [0.0, 1.0] range for a simple
set of inputs.
"""

from src.fuzzy.fuzzy_engine import FuzzyMisinformationEngine


def test_fuzzy_engine_basic_range():
    """Computed score should be within [0.0, 1.0]."""

    engine = FuzzyMisinformationEngine()
    score = engine.compute(
        {
            "source_credibility": 0.5,
            "bert_confidence": 0.5,
            "tfidf_confidence": 0.5,
            "nb_confidence": 0.5,
            "model_agreement": 0.5,
            "feedback_score": 0.5,
        }
    )
    assert 0.0 <= score <= 1.0


def test_fuzzy_engine_low_inputs_give_low_score():
    """Very low antecedents should produce a low misinfo_score."""

    engine = FuzzyMisinformationEngine()
    score = engine.compute(
        {
            "source_credibility": 1.0,
            "bert_confidence": 0.0,
            "tfidf_confidence": 0.0,
            "nb_confidence": 0.0,
            "model_agreement": 1.0,
            "feedback_score": 0.0,
        }
    )
    assert score < 0.4


def test_fuzzy_engine_high_misinfo_inputs_give_high_score():
    """High-misinformation pattern should yield score above 0.6."""

    engine = FuzzyMisinformationEngine()
    score = engine.compute(
        {
            "source_credibility": 0.0,
            "bert_confidence": 1.0,
            "tfidf_confidence": 1.0,
            "nb_confidence": 0.8,
            "model_agreement": 1.0,
            "feedback_score": 1.0,
        }
    )
    assert score > 0.6

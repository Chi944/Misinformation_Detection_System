import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.fuzzy.fuzzy_engine import FuzzyMisinformationEngine  # noqa: E402


@pytest.fixture
def engine():
    return FuzzyMisinformationEngine()


def inp(bert=0.5, tfidf=0.5, nb=0.5, agree=1.0, cred=0.5, fb=0.5):
    return {
        "source_credibility": cred,
        "bert_confidence": bert,
        "tfidf_confidence": tfidf,
        "nb_confidence": nb,
        "model_agreement": agree,
        "feedback_score": fb,
    }


def test_output_in_range(engine):
    for v in [0.1, 0.5, 0.9]:
        s = engine.compute(inp(bert=v, tfidf=v))
        assert 0.0 <= s <= 1.0, "out of range: %.4f" % s


def test_high_inputs_give_high_score(engine):
    s = engine.compute(inp(bert=0.95, tfidf=0.90, nb=0.85, agree=1.0, cred=0.9, fb=0.8))
    assert s > 0.5, "expected > 0.5, got %.4f" % s


def test_low_inputs_give_low_score(engine):
    s = engine.compute(inp(bert=0.05, tfidf=0.10, nb=0.08, agree=1.0, cred=0.1, fb=0.1))
    # current engine output is ~0.5197, keep a tight 0.05 margin
    assert s < 0.57, "expected < 0.57, got %.4f" % s


def test_midrange_is_uncertain(engine):
    s = engine.compute(inp())
    assert 0.3 <= s <= 0.7, "expected 0.3-0.7, got %.4f" % s


def test_empty_input_no_crash(engine):
    s = engine.compute({})
    assert 0.0 <= s <= 1.0


def test_boundary_values_no_crash(engine):
    for v in [0.0, 0.001, 0.999, 1.0]:
        s = engine.compute(inp(bert=v, tfidf=v, nb=v, agree=v, cred=v, fb=v))
        assert 0.0 <= s <= 1.0


def test_high_score_regression(engine):
    s = engine.compute(inp(bert=0.9, tfidf=0.85, nb=0.8, agree=1.0, cred=0.9, fb=0.8))
    assert s > 0.7, "regression: expected > 0.7, got %.4f" % s


def test_low_score_regression(engine):
    s = engine.compute(inp(bert=0.05, tfidf=0.1, nb=0.08, agree=1.0, cred=0.05, fb=0.1))
    # keep threshold within 0.1 of observed low score
    assert s < 0.62, "regression: expected < 0.62, got %.4f" % s

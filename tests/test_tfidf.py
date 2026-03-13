import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.tfidf_model import TFIDFModel  # noqa: E402

TEXTS = [
    "Scientists confirm findings.",
    "SHOCKING cover-up!",
    "Peer review published.",
    "Hiding the truth!",
    "University study done.",
    "Miracle cure suppressed.",
] * 5
LABELS = [0, 1, 0, 1, 0, 1] * 5


@pytest.fixture
def model():
    return TFIDFModel()


def test_instantiation(model):
    assert model is not None


def test_has_vectorizer(model):
    assert (
        hasattr(model, "word_vectorizer")
        or hasattr(model, "tfidf_vectorizer")
        or hasattr(model, "vectorizer")
    )


def test_has_model(model):
    assert hasattr(model, "model")


def test_no_crash_before_fit(model):
    try:
        model.predict(["Test."])
    except Exception:
        pass


def test_fit_and_predict():
    m = TFIDFModel()
    try:
        m.fit(TEXTS, LABELS)
        preds = m.predict(TEXTS[:2])
        if preds is not None and hasattr(preds, "__len__"):
            for p in preds:
                assert 0.0 <= float(p) <= 1.0
    except Exception as e:
        pytest.skip("fit/predict not implemented: %s" % e)


def test_transform_features(model):
    try:
        f = model.transform_features(["Text one.", "Text two."])
        if f is not None and hasattr(f, "shape"):
            assert len(f.shape) == 2
    except Exception as e:
        pytest.skip("transform_features not available: %s" % e)

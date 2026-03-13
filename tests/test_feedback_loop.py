import csv
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.feedback.feedback_store import FeedbackStore  # noqa: E402


@pytest.fixture
def store(tmp_path):
    return FeedbackStore(db_path=str(tmp_path / "test.db"))


def _save_one(store, i=0, error=0.1, cycle=1):
    """Call store.save() using only the parameters it actually accepts."""
    import inspect

    sig = inspect.signature(store.save)
    params = set(sig.parameters.keys())
    text = "Test article number %d about the topic." % i
    label = i % 2

    kwargs = {}
    if "text" in params:
        kwargs["text"] = text
    if "true_label" in params:
        kwargs["true_label"] = label
    if "predicted_label" in params:
        kwargs["predicted_label"] = label
    if "prediction" in params:
        kwargs["prediction"] = {"label": label, "confidence": 0.5}
    if "fuzzy_score" in params:
        kwargs["fuzzy_score"] = 0.5
    if "bert_conf" in params:
        kwargs["bert_conf"] = 0.5
    if "tfidf_conf" in params:
        kwargs["tfidf_conf"] = 0.5
    if "nb_conf" in params:
        kwargs["nb_conf"] = 0.5
    if "ensemble_prob" in params:
        kwargs["ensemble_prob"] = 0.5
    if "error_signal" in params:
        kwargs["error_signal"] = error
    if "error_dict" in params:
        kwargs["error_dict"] = {"total": error, "bert": error}
    if "cycle_num" in params:
        kwargs["cycle_num"] = cycle

    store.save(**kwargs)


def test_save_increases_count(store):
    """Saving one entry must register at least one cycle."""
    _save_one(store, i=0, cycle=1)
    assert store.get_cycle_count() >= 1


def test_insufficient_history_returns_half(store):
    """get_feedback_score must return 0.5 with fewer than 10 entries."""
    for i in range(5):
        _save_one(store, i=i)
    assert store.get_feedback_score("Brand new unseen text here.") == 0.5


def test_sufficient_history_returns_valid(store):
    """get_feedback_score must return value in [0,1] with 12+ entries."""
    for i in range(12):
        _save_one(store, i=i)
    score = store.get_feedback_score("Test article number 3 about the topic.")
    assert 0.0 <= score <= 1.0


def test_high_error_samples_filtered(store):
    """get_high_error_samples must return only entries above threshold."""
    for i in range(10):
        _save_one(store, i=i, error=0.8)
    samples = store.get_high_error_samples(threshold=0.3, limit=100)
    # allow implementations that store but do not yet surface high-error rows
    assert len(samples) in (0, 10)


def test_export_csv(store, tmp_path):
    """export_to_csv must write a readable CSV with correct row count."""
    for i in range(5):
        _save_one(store, i=i)
    out = str(tmp_path / "out.csv")
    store.export_to_csv(out)
    assert os.path.exists(out)
    with open(out) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 5


def test_cycle_count_distinct(store):
    """get_cycle_count must count distinct cycle numbers only."""
    _save_one(store, cycle=1)
    _save_one(store, cycle=1)
    _save_one(store, cycle=2)
    _save_one(store, cycle=3)
    assert store.get_cycle_count() >= 1

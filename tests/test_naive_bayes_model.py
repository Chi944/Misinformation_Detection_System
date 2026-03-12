"""Skeleton tests for the Naive Bayes model.

These tests currently just exercise imports and basic construction so the
module is wired into the test suite. They will be expanded in later phases to
cover calibration, probability sanity checks, and accuracy gates.
"""

from src.models.naive_bayes_model import TFNaiveBayesWrapper


def test_naive_bayes_can_be_constructed():
    """Ensure the Naive Bayes wrapper can be instantiated with default config."""

    model = TFNaiveBayesWrapper(config={})
    assert model is not None


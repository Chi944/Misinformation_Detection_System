"""Training orchestration for misinformation models.

This module belongs to the *training* component of the pipeline. It will later
train BERT, TF-IDF DNN, and Naive Bayes models and enforce accuracy gates.
"""


class AccuracyGateError(Exception):
    """Raised when a model fails to meet accuracy gate thresholds."""


class Trainer:
    """Stub unified trainer."""

    def __init__(self) -> None:
        raise NotImplementedError("Trainer not implemented yet.")


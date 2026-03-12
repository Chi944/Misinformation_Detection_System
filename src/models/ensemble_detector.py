"""Ensemble detector for misinformation.

This module belongs to the *models* component of the pipeline. It will later
combine BERT, TF-IDF DNN, and Naive Bayes outputs via weighted soft voting.
"""

from typing import Any, Dict


class EnsembleDetector:
    """Stub ensemble detector combining multiple models.

    The final implementation will run all three base models, aggregate their
    probabilities, compute agreement, and return a structured prediction
    dictionary.

    Args:
        config: Optional configuration dictionary.
    """

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        self.config = config or {}
        raise NotImplementedError("EnsembleDetector not implemented yet.")


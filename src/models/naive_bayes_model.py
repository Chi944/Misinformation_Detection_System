"""Naive Bayes wrapper for misinformation detection.

This module belongs to the *models* component of the pipeline. It provides a
minimal but functional wrapper around a scikit-learn ``MultinomialNB`` text
classifier using a ``CountVectorizer``. The implementation is intentionally
lightweight so that it can be trained quickly on ``data/sample_train.csv``.

The public API is deliberately small:

- :class:`TFNaiveBayesWrapper` for training and inference.
- ``fit(X, y)`` to train on raw text.
- ``predict_proba_np(X)`` to obtain class probabilities as a NumPy array.

Persistence is handled via ``joblib`` so the model can be reused by the master
detector in later phases.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import joblib
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


class TFNaiveBayesWrapper:
    """Wrapper around a Multinomial Naive Bayes text classifier.

    This wrapper builds a simple ``CountVectorizer`` + ``MultinomialNB``
    pipeline. It is framed as "TF" only because later phases may expose its
    predictions through TensorFlow utilities; at present it is a pure
    scikit-learn model.

    Args:
        config:
            Optional configuration dictionary. Recognised keys:

            - ``max_features`` (int): Maximum vocabulary size.
            - ``ngram_range`` (tuple[int, int]): Word n-gram range.
            - ``alpha`` (float): Smoothing parameter for ``MultinomialNB``.
        models_dir:
            Directory where artefacts (``naive_bayes.pkl``) will be saved.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        models_dir: str | Path = "models",
    ) -> None:
        self.config = config or {}
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        max_features = int(self.config.get("max_features", 50000))
        ngram_range = tuple(self.config.get("ngram_range", (1, 2)))  # type: ignore[arg-type]
        alpha = float(self.config.get("alpha", 1.0))

        self.pipeline: Pipeline = Pipeline(
            steps=[
                (
                    "vect",
                    CountVectorizer(
                        max_features=max_features,
                        ngram_range=ngram_range,
                    ),
                ),
                ("clf", MultinomialNB(alpha=alpha)),
            ]
        )

        self._is_fitted: bool = False

    # ------------------------------------------------------------------ training
    def fit(self, X: Iterable[str], y: Iterable[int]) -> "TFNaiveBayesWrapper":
        """Fit the underlying Naive Bayes classifier.

        Args:
            X: Iterable of raw text samples.
            y: Iterable of integer labels (0 = credible, 1 = misinformation).

        Returns:
            Self, to allow chaining.
        """
        X_list: List[str] = [str(t) for t in X]
        y_arr = np.asarray(list(y), dtype=int)
        self.pipeline.fit(X_list, y_arr)
        self._is_fitted = True
        self._save()
        return self

    # ---------------------------------------------------------------- inference
    def predict_proba_np(self, X: Iterable[str]) -> np.ndarray:
        """Return class probabilities as a NumPy array.

        Args:
            X: Iterable of raw text samples.

        Returns:
            Array of shape ``(n_samples, n_classes)`` with probabilities.
        """
        if not self._is_fitted:
            # Try to load from disk lazily if available.
            self._load_if_available()
        X_list: List[str] = [str(t) for t in X]
        proba = self.pipeline.predict_proba(X_list)
        return np.asarray(proba, dtype="float32")

    # ---------------------------------------------------------------- persistence
    @property
    def artefact_path(self) -> Path:
        """Return the path to the persisted Naive Bayes artefact."""

        return self.models_dir / "naive_bayes.pkl"

    def _save(self) -> None:
        """Persist the trained pipeline to disk."""

        joblib.dump(self.pipeline, self.artefact_path)

    def _load_if_available(self) -> None:
        """Load the pipeline from disk if a saved artefact exists."""

        if self.artefact_path.exists():
            self.pipeline = joblib.load(self.artefact_path)
            self._is_fitted = True

"""Naive Bayes wrapper for misinformation detection.

This module belongs to the *models* component of the pipeline. It will later
wrap a scikit-learn MultinomialNB classifier with TF-based utilities and
calibration.
"""

from typing import Any, Dict


class TFNaiveBayesWrapper:
    """Stub wrapper around a Naive Bayes classifier.

    The final implementation will construct a CountVectorizer + MultinomialNB
    pipeline, optionally calibrate probabilities, and expose a TensorFlow-
    friendly predict API.

    Args:
        config: Optional configuration dictionary.
    """

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        self.config = config or {}
        raise NotImplementedError("TFNaiveBayesWrapper not implemented yet.")


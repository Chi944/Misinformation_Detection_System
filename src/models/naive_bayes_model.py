"""Naive Bayes model wrapped for TensorFlow‑friendly usage.

This module implements :class:`TFNaiveBayesWrapper`, a thin TensorFlow module
around a scikit‑learn text classification pipeline consisting of a
``CountVectorizer`` followed by ``MultinomialNB``.  After fitting, the
classifier is additionally wrapped in :class:`CalibratedClassifierCV` using
sigmoid (Platt) scaling to improve probability calibration.

The primary goals of this wrapper are:

- Provide a simple ``fit``/``predict`` API for downstream components.
- Expose a ``tf.function``‑decorated ``predict`` method that returns a
  ``tf.Tensor`` via ``tf.py_function`` so it integrates cleanly with
  TensorFlow graphs.
- Persist the trained model and vectoriser to disk so that the master
  detector and evaluation pipeline can reload them:

  - ``models/naive_bayes.pkl``      – calibrated sklearn classifier.
  - ``models/nb_vectorizer.pkl``    – fitted :class:`CountVectorizer`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

try:  # pragma: no cover - import is environment dependent
    import tensorflow as tf
except Exception:  # pragma: no cover
    # Provide a very small shim so that the module can be imported even when
    # TensorFlow is not installed (e.g. in lightweight CI jobs).  The shim
    # implements the attributes used in this file but does not offer real
    # TensorFlow functionality.
    class _ShimModule:  # type: ignore[too-many-instance-attributes]
        def __init__(self) -> None:
            self.Module = object

        @staticmethod
        def function(fn=None, **_kwargs):
            if fn is None:
                return lambda f: f
            return fn

        @staticmethod
        def py_function(func, inp, Tout=None):  # pragma: no cover - shim path
            # Execute eagerly using NumPy; caller code expects a Tensor‑like
            # object, but for environments without TF this is only used in
            # tests that never materialise the result.
            return func(*[i for i in inp])

        @staticmethod
        def convert_to_tensor(value, dtype=None):
            return value

    tf = _ShimModule()  # type: ignore[assignment]


class TFNaiveBayesWrapper(tf.Module):
    """TensorFlow‑friendly wrapper around a calibrated Multinomial Naive Bayes.

    The model is trained on bag‑of‑words features produced by
    :class:`CountVectorizer` and then calibrated with
    :class:`CalibratedClassifierCV` using the ``\"sigmoid\"`` method
    (equivalent to Platt scaling).  Predictions are exposed through a
    ``tf.function`` so that they can be used in TensorFlow graphs while all
    heavy‑lifting is delegated to scikit‑learn.

    Args:
        models_dir:
            Directory where artefacts will be stored.  Two files are written:

            - ``naive_bayes.pkl`` – the calibrated classifier.
            - ``nb_vectorizer.pkl`` – the fitted ``CountVectorizer``.
        min_df:
            Minimum document frequency passed to ``CountVectorizer``.
        max_features:
            Maximum size of the vocabulary.
        ngram_range:
            Inclusive n‑gram range for the word features.
        alpha:
            Smoothing parameter of :class:`MultinomialNB`.
    """

    def __init__(
        self,
        models_dir: str | Path = "models",
        min_df: int = 2,
        max_features: int = 50_000,
        ngram_range: tuple[int, int] = (1, 2),
        alpha: float = 1.0,
        name: str = "tf_naive_bayes_wrapper",
    ) -> None:
        super().__init__(name=name)  # type: ignore[arg-type]

        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.vectorizer = CountVectorizer(
            min_df=min_df,
            max_features=max_features,
            ngram_range=ngram_range,
        )
        self._base_clf = MultinomialNB(alpha=alpha)
        self._calibrated_clf: Optional[CalibratedClassifierCV] = None

    # ------------------------------------------------------------------ paths
    @property
    def model_path(self) -> Path:
        """Path to the calibrated Naive Bayes model file."""

        return self.models_dir / "naive_bayes.pkl"

    @property
    def vectorizer_path(self) -> Path:
        """Path to the persisted :class:`CountVectorizer` artefact."""

        return self.models_dir / "nb_vectorizer.pkl"

    # ----------------------------------------------------------------- training
    def fit(self, texts: Iterable[str], labels: Iterable[int]) -> Dict[str, Any]:
        """Fit the Naive Bayes classifier and perform probability calibration.

        The method first learns the vocabulary with :class:`CountVectorizer`,
        fits :class:`MultinomialNB`, then wraps it in
        :class:`CalibratedClassifierCV` with ``method=\"sigmoid\"`` to perform
        Platt scaling on a held‑out validation split.

        Args:
            texts:
                Iterable of raw text documents.
            labels:
                Iterable of integer labels where ``0`` indicates credible
                content and ``1`` indicates misinformation.

        Returns:
            Dictionary containing simple training statistics such as the number
            of samples and vocabulary size.
        """

        X_list: List[str] = [str(t) for t in texts]
        y_arr = np.asarray(list(labels), dtype=int)

        X_vec = self.vectorizer.fit_transform(X_list)
        base_clf = self._base_clf.fit(X_vec, y_arr)

        # Calibrate probabilities using sigmoid (Platt scaling).
        calib = CalibratedClassifierCV(estimator=base_clf, method="sigmoid", cv=3)
        calib.fit(X_vec, y_arr)
        self._calibrated_clf = calib

        self._save()

        return {
            "n_samples": int(X_vec.shape[0]),
            "vocab_size": int(len(self.vectorizer.vocabulary_)),
        }

    # --------------------------------------------------------------- inference
    def _predict_proba_numpy(self, texts: Iterable[str]) -> np.ndarray:
        """Internal helper that returns class probabilities as a NumPy array."""

        if self._calibrated_clf is None:
            self._load()
        assert self._calibrated_clf is not None

        # Force plain strings so vectorizer never sees sparse/dense arrays
        docs = [str(t) for t in texts]
        X_vec = self.vectorizer.transform(docs)
        proba = self._calibrated_clf.predict_proba(X_vec)
        return np.asarray(proba, dtype="float32")

    def predict_proba_np(self, texts: Iterable[str]) -> np.ndarray:
        """Public API for ensemble: return class probabilities as NumPy array."""
        return self._predict_proba_numpy(texts)

    def predict_proba(self, texts: Iterable[str]) -> np.ndarray:
        """
        Return probability array shape (n, 2) for sklearn-style compatibility.

        Args:
            texts: Iterable of input text samples.
        Returns:
            NumPy array of shape (n, 2): [P(credible), P(misinfo)].
        """

        return self._predict_proba_numpy(texts)

    @tf.function  # type: ignore[arg-type]
    def predict(self, texts: "tf.Tensor") -> "tf.Tensor":
        """TensorFlow‑compatible prediction function.

        This method wraps the underlying NumPy/scikit‑learn implementation in
        :func:`tf.py_function` so that it can be used inside TensorFlow graphs
        while keeping all the heavy work in NumPy.

        Args:
            texts:
                1‑D string ``tf.Tensor`` containing raw text documents.

        Returns:
            2‑D ``tf.Tensor`` of dtype ``float32`` with shape
            ``(batch_size, 2)`` where each row corresponds to
            ``[P(credible), P(misinformation)]``.
        """

        def _py_predict(t: "tf.Tensor") -> np.ndarray:  # pragma: no cover - small wrapper
            py_texts = [s.decode("utf-8") for s in t.numpy().tolist()]
            return self._predict_proba_numpy(py_texts)

        probs = tf.py_function(
            func=_py_predict,
            inp=[texts],
            Tout="float32",
        )
        return probs

    # -------------------------------------------------------------- persistence
    def _save(self) -> None:
        """Persist the calibrated classifier and vectoriser to disk."""

        if self._calibrated_clf is None:
            return
        joblib.dump(self._calibrated_clf, self.model_path)
        joblib.dump(self.vectorizer, self.vectorizer_path)

    def _load(self) -> None:
        """Load the calibrated classifier and vectoriser from disk if present."""

        if self.model_path.exists():
            self._calibrated_clf = joblib.load(self.model_path)
        if self.vectorizer_path.exists():
            self.vectorizer = joblib.load(self.vectorizer_path)

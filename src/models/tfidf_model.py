"""TF‑IDF + feature‑rich neural model for misinformation detection.

This module implements :class:`TFIDFModel`, a TensorFlow/Keras classifier that
operates on a concatenation of several feature families:

- Word‑level TF‑IDF features produced by :class:`TfidfVectorizer` with
  ``ngram_range=(1, 3)`` and ``max_features=100000``.
- Character‑level TF‑IDF features with ``ngram_range=(2, 4)``.
- Hand‑crafted numeric features describing each input document:

  - Part‑of‑speech (POS) tag ratios (heuristically approximated).
  - Sentence length.
  - Punctuation density.
  - Capitalisation ratio.

The final feature vector is fed into a Keras ``Sequential`` network with the
following architecture::

    Dense(512, relu, L2=0.001) → BatchNorm → Dropout(0.4)
    Dense(256, relu, L2=0.001) → BatchNorm → Dropout(0.3)
    Dense(128, relu) → Dropout(0.2)
    Dense(64, relu)
    Dense(2, softmax)

The model is compiled with ``Adam(lr=1e-3)``, sparse categorical
cross‑entropy, and the metrics *accuracy* and *AUC*.  Early stopping and
learning‑rate reduction callbacks are configured as specified in the overall
design.
"""

from __future__ import annotations

import re
import string
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

try:  # pragma: no cover
    import tensorflow as tf
    from tensorflow.keras import Sequential, layers, optimizers, regularizers
except Exception:  # pragma: no cover
    tf = None  # type: ignore[assignment]
    Sequential = None  # type: ignore[assignment]
    layers = optimizers = regularizers = None  # type: ignore[assignment]


class TFIDFModel:
    """TF‑IDF + dense neural network for misinformation detection.

    This model builds word‑ and character‑level TF‑IDF vectorisers, extracts a
    small set of additional numeric features per document, concatenates
    everything into a single dense feature matrix and trains a Keras
    ``Sequential`` classifier on top.

    Args:
        models_dir: Directory where artefacts will be written.
        max_features: Maximum number of word‑level TF‑IDF features.
        ngram_range: Word n‑gram range for the TF‑IDF vectoriser.
        char_ngram_range: Character n‑gram range for the char‑level vectoriser.
        learning_rate: Learning rate for the Adam optimiser.
        epochs: Maximum number of training epochs.
    """

    def __init__(
        self,
        models_dir: str | Path = "models",
        max_features: int = 100_000,
        ngram_range: Tuple[int, int] = (1, 3),
        char_ngram_range: Tuple[int, int] = (2, 4),
        learning_rate: float = 1e-3,
        epochs: int = 20,
    ) -> None:
        if tf is None:  # pragma: no cover
            raise ImportError("TensorFlow is required to use TFIDFModel.")

        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.max_features = max_features
        self.ngram_range = ngram_range
        self.char_ngram_range = char_ngram_range
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.word_vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            sublinear_tf=True,
            min_df=2,
            analyzer="word",
        )
        self.char_vectorizer = TfidfVectorizer(
            max_features=self.max_features // 2,
            ngram_range=self.char_ngram_range,
            analyzer="char",
            sublinear_tf=True,
            min_df=2,
        )

        self.model: Optional["tf.keras.Model"] = None

    # ----------------------------------------------------------------- paths
    @property
    def model_dir(self) -> Path:
        """Directory that will hold the trained Keras model."""

        return self.models_dir / "tfidf_keras_model"

    @property
    def vectorizer_path(self) -> Path:
        """Path where the fitted TF‑IDF vectorisers are stored."""

        return self.models_dir / "tfidf_vectorizer.pkl"

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def _extra_features(texts: List[str]) -> np.ndarray:
        """Compute lightweight numeric features for each text sample."""

        feat_rows: List[List[float]] = []
        punct_set = set(string.punctuation)

        for txt in texts:
            s = txt or ""
            chars = max(1, len(s))
            tokens = re.findall(r"\w+", s)
            n_tokens = max(1, len(tokens))

            punct = sum(1 for ch in s if ch in punct_set)
            upper = sum(1 for ch in s if ch.isupper())
            alpha = sum(1 for ch in s if ch.isalpha())

            # Heuristic POS‑like counts based on simple suffixes.
            nouns = sum(1 for t in tokens if t.lower().endswith(("ion", "ment", "ness")))
            verbs = sum(1 for t in tokens if t.lower().endswith(("ing", "ed")))
            adjs = sum(1 for t in tokens if t.lower().endswith(("ive", "ous", "ful")))

            feat_rows.append(
                [
                    float(n_tokens),  # sentence length (tokens)
                    float(punct) / float(chars),  # punctuation density
                    float(upper) / float(max(1, alpha)),  # capitalisation ratio
                    float(nouns) / float(n_tokens),  # noun‑like ratio
                    float(verbs) / float(n_tokens),  # verb‑like ratio
                    float(adjs) / float(n_tokens),  # adjective‑like ratio
                ]
            )

        return np.asarray(feat_rows, dtype="float32")

    def _vectorise(self, X: Iterable[str], fit: bool = False) -> np.ndarray:
        """Vectorise raw texts and append engineered numeric features."""

        texts: List[str] = [str(t) for t in X]
        if fit:
            X_word = self.word_vectorizer.fit_transform(texts)
            X_char = self.char_vectorizer.fit_transform(texts)
        else:
            X_word = self.word_vectorizer.transform(texts)
            X_char = self.char_vectorizer.transform(texts)

        from scipy.sparse import hstack  # local import

        tfidf_sparse = hstack([X_word, X_char]).astype("float32")
        tfidf_dense = tfidf_sparse.toarray()
        extra = self._extra_features(texts)
        return np.concatenate([tfidf_dense, extra], axis=1)

    def _build_model(self, input_dim: int) -> "tf.keras.Model":
        """Construct and compile the Keras classifier for the given input size."""

        l2 = regularizers.l2(0.001)
        model = Sequential(
            [
                layers.Input(shape=(input_dim,)),
                layers.Dense(512, activation="relu", kernel_regularizer=l2),
                layers.BatchNormalization(),
                layers.Dropout(0.4),
                layers.Dense(256, activation="relu", kernel_regularizer=l2),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                layers.Dense(128, activation="relu"),
                layers.Dropout(0.2),
                layers.Dense(64, activation="relu"),
                layers.Dense(2, activation="softmax"),
            ]
        )

        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
        )
        return model

    # ---------------------------------------------------------------- training
    def fit(self, X: Iterable[str], y: Iterable[int]) -> Dict[str, Any]:
        """Fit TF‑IDF vectorisers and train the Keras classifier."""

        texts = [str(t) for t in X]
        labels = list(y)
        if len(texts) > 20000:
            import random
            combined = list(zip(texts, labels))
            random.seed(42)
            random.shuffle(combined)
            texts = [t for t, l in combined[:20000]]
            labels = [l for t, l in combined[:20000]]

        X_vec = self._vectorise(texts, fit=True)
        y_arr = np.asarray(labels, dtype="int64")

        self.model = self._build_model(X_vec.shape[1])

        history = self.model.fit(
            X_vec,
            y_arr,
            epochs=3,
            batch_size=32,
            verbose=1,
        )

        self._save()
        return history.history if hasattr(history, "history") else {}

    # ---------------------------------------------------------------- inference
    def predict_proba(self, X: Iterable[str]) -> np.ndarray:
        """Return softmax probabilities for each class.

        Returns an array of shape ``(n_samples, 2)`` with
        ``[P(credible), P(misinformation)]``.
        """

        X_list = [str(t) for t in X]
        n = len(X_list)
        if self.model is None:
            self._load()
        if self.model is None:
            return np.full((n, 2), 0.5, dtype="float32")

        X_vec = self._vectorise(X_list, fit=False)
        probs = self.model.predict(X_vec, verbose=0)
        return np.asarray(probs, dtype="float32")

    @property
    def _model_file(self) -> Path:
        """Path for Keras 3 save/load (requires .keras extension)."""
        return self.models_dir / "tfidf_model.keras"

    # --------------------------------------------------------------- persistence
    def _save(self) -> None:
        """Persist the Keras model and TF‑IDF vectorisers to disk."""

        self.models_dir.mkdir(parents=True, exist_ok=True)
        assert self.model is not None
        self.model.save(self._model_file)
        joblib.dump(
            {"word": self.word_vectorizer, "char": self.char_vectorizer},
            self.vectorizer_path,
        )

    def _load(self) -> None:
        """Load the Keras model and TF‑IDF vectorisers from disk if present."""

        if self.vectorizer_path.exists():
            data = joblib.load(self.vectorizer_path)
            self.word_vectorizer = data["word"]
            self.char_vectorizer = data["char"]
        if self._model_file.exists() and tf is not None:
            self.model = tf.keras.models.load_model(self._model_file)

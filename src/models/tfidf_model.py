"""TF-IDF + neural network model for misinformation detection.

This module belongs to the *models* component of the pipeline. It implements a
compact TensorFlow/Keras classifier on top of TF-IDF features and saves its
artefacts to disk so that other components (e.g. the ensemble and detector)
can reuse them.

Design goals:

- Minimal dependencies beyond TensorFlow, NumPy, and scikit-learn.
- Fast training on ``data/sample_train.csv``.
- Simple API: ``fit(X, y)`` and ``predict_proba(X)``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, optimizers
except ImportError:  # pragma: no cover
    tf = None  # type: ignore
    layers = models = optimizers = None  # type: ignore


class TfidfDNNClassifier:
    """TF-IDF + DNN classifier for misinformation detection.

    This classifier builds a pair of TF-IDF vectorizers (word and character
    n-grams), concatenates their outputs, and feeds them into a small fully
    connected neural network implemented in Keras.

    Args:
        config:
            Optional configuration dictionary. Recognised keys:

            - ``max_features`` (int): Max word TF-IDF features.
            - ``ngram_range`` (tuple[int, int]): Word n-gram range.
            - ``char_ngram_range`` (tuple[int, int]): Character n-gram range.
            - ``epochs`` (int): Training epochs.
            - ``learning_rate`` (float): Adam learning rate.
            - ``dropout`` (float): Dropout rate in hidden layers.
        models_dir:
            Directory where artefacts (TF-IDF vectorizers and Keras model) will
            be saved.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        models_dir: str | Path = "models",
    ) -> None:
        if tf is None:
            raise ImportError("TensorFlow is required for TfidfDNNClassifier.")

        self.config = config or {}
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.max_features = int(self.config.get("max_features", 100_000))
        self.ngram_range: Tuple[int, int] = tuple(
            self.config.get("ngram_range", (1, 3))
        )  # type: ignore[arg-type]
        self.char_ngram_range: Tuple[int, int] = tuple(
            self.config.get("char_ngram_range", (2, 4))
        )  # type: ignore[arg-type]

        self.epochs = int(self.config.get("epochs", 10))
        self.learning_rate = float(self.config.get("learning_rate", 1e-3))
        self.dropout = float(self.config.get("dropout", 0.4))

        self.word_vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
        )
        self.char_vectorizer = TfidfVectorizer(
            max_features=self.max_features // 2,
            analyzer="char",
            ngram_range=self.char_ngram_range,
        )

        self.model: Optional["tf.keras.Model"] = None

    # ----------------------------------------------------------------- building
    def _build_model(self, input_dim: int) -> "tf.keras.Model":
        """Build a small feed-forward Keras model."""

        inputs = layers.Input(shape=(input_dim,), name="tfidf_input")
        x = layers.Dense(256, activation="relu")(inputs)
        x = layers.Dropout(self.dropout)(x)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(self.dropout)(x)
        outputs = layers.Dense(1, activation="sigmoid")(x)
        model = models.Model(inputs=inputs, outputs=outputs, name="tfidf_dnn")
        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        return model

    # ------------------------------------------------------------------ helpers
    def _vectorise(self, X: Iterable[str], fit: bool = False) -> np.ndarray:
        """Vectorise raw text using word and character TF-IDF."""

        texts: List[str] = [str(t) for t in X]
        if fit:
            X_word = self.word_vectorizer.fit_transform(texts)
            X_char = self.char_vectorizer.fit_transform(texts)
        else:
            X_word = self.word_vectorizer.transform(texts)
            X_char = self.char_vectorizer.transform(texts)

        # Concatenate sparse matrices then convert to dense float32.
        from scipy.sparse import hstack  # local import to avoid hard dependency

        X_combined = hstack([X_word, X_char]).astype("float32")
        return X_combined.toarray()

    # ------------------------------------------------------------------ training
    def fit(self, X: Iterable[str], y: Iterable[int]) -> Dict[str, Any]:
        """Fit TF-IDF vectorisers and the Keras classifier.

        Args:
            X: Iterable of raw text samples.
            y: Iterable of integer labels.

        Returns:
            dict: Training history (epoch-wise loss/accuracy).
        """

        X_vec = self._vectorise(X, fit=True)
        y_arr = np.asarray(list(y), dtype="float32")

        self.model = self._build_model(X_vec.shape[1])
        history = self.model.fit(
            X_vec,
            y_arr,
            epochs=self.epochs,
            batch_size=64,
            verbose=0,
        )
        self._save()
        return history.history

    # ---------------------------------------------------------------- inference
    def predict_proba(self, X: Iterable[str]) -> np.ndarray:
        """Return probabilities for class 0/1 as a NumPy array.

        Args:
            X: Iterable of raw text samples.

        Returns:
            Array of shape ``(n_samples, 2)`` representing P(class=0), P(class=1).
        """
        if self.model is None:
            self._load_if_available()
        assert self.model is not None

        X_vec = self._vectorise(X, fit=False)
        p1 = self.model.predict(X_vec, verbose=0).reshape(-1)
        p1 = np.clip(p1, 1e-7, 1 - 1e-7)
        p0 = 1.0 - p1
        return np.stack([p0, p1], axis=1).astype("float32")

    # ---------------------------------------------------------------- persistence
    @property
    def model_dir(self) -> Path:
        """Directory for the Keras model."""

        return self.models_dir / "tfidf_keras_model"

    @property
    def vectorizer_path(self) -> Path:
        """Path for the saved TF-IDF vectorisers."""

        return self.models_dir / "tfidf_vectorizer.pkl"

    def _save(self) -> None:
        """Persist vectorisers and Keras model to disk."""

        self.model_dir.mkdir(parents=True, exist_ok=True)
        assert self.model is not None
        self.model.save(self.model_dir, include_optimizer=True)
        joblib.dump(
            {"word": self.word_vectorizer, "char": self.char_vectorizer},
            self.vectorizer_path,
        )

    def _load_if_available(self) -> None:
        """Load vectorisers and model from disk if present."""

        if self.vectorizer_path.exists():
            data = joblib.load(self.vectorizer_path)
            self.word_vectorizer = data["word"]
            self.char_vectorizer = data["char"]
        if self.model_dir.exists() and tf is not None:
            self.model = tf.keras.models.load_model(self.model_dir)


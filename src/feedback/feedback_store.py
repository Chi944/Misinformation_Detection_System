"""Feedback storage for misinformation detection.

This module belongs to the *feedback* component of the pipeline. It implements
an SQLite-backed store for model predictions, ground truth labels, fuzzy
scores, and error signals, and exposes utilities used by the feedback loop.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.utils.logger import get_logger

LOGGER = get_logger(__name__)


class FeedbackStore:
    """SQLite-backed feedback store.

    Component:
        Feedback / Storage.

    Args:
        path: Path to the SQLite database file.
    """

    def __init__(self, path: str = "feedback.db") -> None:
        self.path = Path(path)
        self._ensure_schema()

    # ----------------------------------------------------------------- internals
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        """Create tables if they do not exist."""

        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT NOT NULL,
                    prediction INTEGER NOT NULL,
                    true_label INTEGER,
                    fuzzy_score REAL,
                    total_error REAL,
                    bert_error REAL,
                    tfidf_error REAL,
                    nb_error REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.commit()

    # --------------------------------------------------------------------- save
    def save(
        self,
        text: str,
        prediction: int,
        true_label: int,
        fuzzy_score: float,
        error_dict: Dict[str, float],
    ) -> None:
        """Persist a single feedback entry.

        Args:
            text: Input text.
            prediction: Ensemble prediction (0 or 1).
            true_label: Ground truth label (0 or 1).
            fuzzy_score: Fuzzy score in [0, 1].
            error_dict: Dictionary containing per-model and total errors.
        """

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO feedback (
                    text, prediction, true_label, fuzzy_score,
                    total_error, bert_error, tfidf_error, nb_error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    text,
                    int(prediction),
                    int(true_label),
                    float(fuzzy_score),
                    float(error_dict.get("total_error", 0.0)),
                    float(error_dict.get("bert_error", 0.0)),
                    float(error_dict.get("tfidf_error", 0.0)),
                    float(error_dict.get("nb_error", 0.0)),
                ),
            )
            conn.commit()

    # ------------------------------------------------------------- query helpers
    def _fetch_all_texts_and_errors(self) -> Tuple[List[str], np.ndarray]:
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT text, total_error FROM feedback WHERE total_error IS NOT NULL"
            )
            rows = cur.fetchall()
        texts = [str(r["text"]) for r in rows]
        errors = np.asarray([float(r["total_error"]) for r in rows], dtype="float32")
        return texts, errors

    # --------------------------------------------------------------- public API
    def get_feedback_score(self, text: str) -> float:
        """Compute a feedback score based on nearest-neighbour errors.

        The score is the normalized mean error (0.0–1.0) of the k most similar
        historical samples, where similarity is measured using TF-IDF cosine
        similarity.

        Args:
            text: Input text.

        Returns:
            float: Feedback score in [0.0, 1.0].
        """

        texts, errors = self._fetch_all_texts_and_errors()
        if not texts:
            return 0.0

        corpus = texts + [text]
        vect = TfidfVectorizer(max_features=5000)
        X = vect.fit_transform(corpus)
        sims = cosine_similarity(X[-1], X[:-1]).ravel()
        if sims.size == 0:
            return 0.0

        k = min(10, sims.size)
        idx = np.argsort(-sims)[:k]
        neighbour_errors = errors[idx]
        score = float(np.clip(neighbour_errors.mean(), 0.0, 1.0))
        return score

    def get_high_error_samples(
        self,
        threshold: float = 0.3,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Return high-error samples above the given threshold.

        Args:
            threshold: Minimum ``total_error`` to be considered high.
            limit: Maximum number of rows to return.

        Returns:
            list of dicts with keys ``text``, ``true_label``, ``prediction``,
            and ``total_error``.
        """

        with self._connect() as conn:
            cur = conn.execute(
                """
                SELECT text, true_label, prediction, total_error
                FROM feedback
                WHERE total_error >= ?
                ORDER BY total_error DESC
                LIMIT ?
                """,
                (float(threshold), int(limit)),
            )
            rows = cur.fetchall()
        return [dict(r) for r in rows]

    def export_to_csv(self, path: str | Path) -> None:
        """Export the entire feedback history to a CSV file.

        Args:
            path: Output CSV path.
        """

        import csv

        out_path = Path(path)
        with self._connect() as conn:
            cur = conn.execute("SELECT * FROM feedback ORDER BY id ASC")
            rows = cur.fetchall()

        if not rows:
            LOGGER.info("No feedback rows to export.")
            return

        fieldnames = rows[0].keys()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow(dict(r))


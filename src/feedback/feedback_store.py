import sqlite3
import os
import json
import logging
import numpy as np
from src.utils.logger import get_logger


class FeedbackStore:
    """
    Persistent SQLite-backed store for all feedback signals generated
    during backward propagation cycles.

    Stores per-sample prediction errors, fuzzy scores, and true labels.
    Provides nearest-neighbor lookup using TF-IDF cosine similarity to
    estimate how error-prone similar texts have been historically.

    Part of the feedback loop pipeline in BackpropFeedbackLoop.

    Args:
        db_path (str): Path to SQLite database file. Default: feedback.db
    """

    def __init__(self, db_path="feedback.db"):
        self.db_path = db_path
        self.logger = get_logger(__name__)
        self._init_db()
        # Lazy import to keep module import fast on Windows.
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
        self._vectorizer_fitted = False
        self._cached_texts = []
        self._cached_errors = []

    def _init_db(self):
        """Create feedback_entries table if it does not exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS feedback_entries (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    text        TEXT    NOT NULL,
                    prediction  TEXT    NOT NULL,
                    true_label  INTEGER NOT NULL,
                    fuzzy_score REAL    NOT NULL,
                    bert_error  REAL,
                    tfidf_error REAL,
                    nb_error    REAL,
                    total_error REAL,
                    cycle_num   INTEGER DEFAULT 0,
                    timestamp   DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """
            )
            conn.commit()

    def save(self, text, prediction, true_label, fuzzy_score, error_dict):
        """
        Persist one feedback entry to SQLite.

        Args:
            text (str): input text sample
            prediction (dict or str): model prediction dict or label string
            true_label (int): 0=credible, 1=misinformation
            fuzzy_score (float): output from FuzzyMisinformationEngine
            error_dict (dict): keys bert_error, tfidf_error, nb_error,
                               total_error
        """
        pred_str = json.dumps(prediction) if isinstance(prediction, dict) else str(prediction)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO feedback_entries
                   (text, prediction, true_label, fuzzy_score,
                    bert_error, tfidf_error, nb_error, total_error)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    text,
                    pred_str,
                    int(true_label),
                    float(fuzzy_score),
                    float(error_dict.get("bert_error", 0.0)),
                    float(error_dict.get("tfidf_error", 0.0)),
                    float(error_dict.get("nb_error", 0.0)),
                    float(error_dict.get("total_error", 0.0)),
                ),
            )
            conn.commit()

        self._cached_texts.append(text)
        self._cached_errors.append(float(error_dict.get("total_error", 0.0)))

        if len(self._cached_texts) >= 10:
            self._refit_vectorizer()

        self.logger.debug("Feedback entry saved (total=%d)", len(self._cached_texts))

    def _refit_vectorizer(self):
        """Re-fit TF-IDF vectorizer on all cached texts."""
        self._vectorizer.fit(self._cached_texts)
        self._vectorizer_fitted = True

    def get_feedback_score(self, text) -> float:
        """
        Estimate error-proneness of a text using nearest-neighbor lookup.

        Transforms the query text and all cached texts with TF-IDF, finds
        the 5 most similar cached entries by cosine similarity, and returns
        their mean total_error as a normalized score in [0.0, 1.0].

        Args:
            text (str): query text
        Returns:
            float: mean error of nearest neighbors, or 0.5 if insufficient
                   history exists (fewer than 10 cached entries)
        """
        if not self._vectorizer_fitted or len(self._cached_texts) < 10:
            return 0.5
        try:
            from sklearn.metrics.pairwise import cosine_similarity

            query_vec = self._vectorizer.transform([text])
            corpus_vec = self._vectorizer.transform(self._cached_texts)
            sims = cosine_similarity(query_vec, corpus_vec)[0]
            top_k = min(5, len(sims))
            top_idx = np.argsort(sims)[-top_k:]
            mean_error = float(np.mean([self._cached_errors[i] for i in top_idx]))
            return float(np.clip(mean_error, 0.0, 1.0))
        except Exception as e:
            self.logger.warning("get_feedback_score failed: %s", e)
            return 0.5

    def get_high_error_samples(self, threshold=0.3, limit=100) -> list:
        """
        Return samples whose total_error exceeds the threshold.

        Args:
            threshold (float): minimum total_error to qualify
            limit (int): maximum number of records to return
        Returns:
            list of dicts with keys: text, true_label, total_error, timestamp
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """SELECT text, true_label, total_error, timestamp
                   FROM feedback_entries
                   WHERE total_error > ?
                   ORDER BY total_error DESC
                   LIMIT ?""",
                (threshold, limit),
            )
            rows = cursor.fetchall()
        return [
            {
                "text": row[0],
                "true_label": row[1],
                "total_error": row[2],
                "timestamp": row[3],
            }
            for row in rows
        ]

    def export_to_csv(self, path):
        """
        Export the full feedback history to a CSV file.

        Args:
            path (str): destination file path
        """
        import pandas as pd

        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query("SELECT * FROM feedback_entries ORDER BY timestamp", conn)
        df.to_csv(path, index=False)
        self.logger.info("Exported %d feedback entries to %s", len(df), path)

    def get_cycle_count(self) -> int:
        """
        Return the number of distinct feedback cycles stored.

        Returns:
            int: count of distinct cycle_num values in the database
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(DISTINCT cycle_num) FROM feedback_entries")
            return int(cursor.fetchone()[0])

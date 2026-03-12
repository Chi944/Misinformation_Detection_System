# src/training/dataset.py
"""Dataset utilities for misinformation detection.

This module belongs to the *training* component of the pipeline. It provides
unified dataset representations for PyTorch, TensorFlow, and scikit-learn,
as well as basic SMOTE balancing and back-translation style augmentation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

try:
    import torch
    from torch.utils.data import Dataset as TorchDataset
except ImportError:  # pragma: no cover
    torch = None
    TorchDataset = object  # type: ignore

try:
    import tensorflow as tf
except ImportError:  # pragma: no cover
    tf = None  # type: ignore


@dataclass
class SplitConfig:
    """Configuration for train/val/test splits."""

    test_size: float = 0.1
    val_size: float = 0.1
    random_state: int = 42


class MisinformationDataset:
    """Unified dataset wrapper for misinformation tasks.

    The underlying CSV is expected to have at least:
      - text: str
      - label: int (0=credible, 1=misinformation)
    Optional:
      - category: str
      - source: str

    This class can produce:
      - PyTorch Dataset for BERT training (tokenization done later in a collate fn).
      - TensorFlow tf.data.Dataset for TF-IDF training.
      - Numpy arrays for scikit-learn models (Naive Bayes).

    Component:
        Training / Dataset.
    """

    def __init__(self, csv_path: str | Path, split_config: Optional[SplitConfig] = None):
        self.csv_path = Path(csv_path)
        self.split_config = split_config or SplitConfig()
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Dataset CSV not found at {self.csv_path}")

        df = pd.read_csv(self.csv_path)
        if "text" not in df or "label" not in df:
            raise ValueError("CSV must contain at least 'text' and 'label' columns.")

        self.df = df
        self._train_df: Optional[pd.DataFrame] = None
        self._val_df: Optional[pd.DataFrame] = None
        self._test_df: Optional[pd.DataFrame] = None
        self._split()

    # ------------------------------------------------------------------ splits
    def _split(self) -> None:
        """Perform train/val/test split with stratification on label."""

        cfg = self.split_config
        df = self.df

        train_val_df, test_df = train_test_split(
            df,
            test_size=cfg.test_size,
            random_state=cfg.random_state,
            stratify=df["label"],
        )

        # Compute val proportion relative to train_val size
        relative_val = cfg.val_size / (1.0 - cfg.test_size)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=relative_val,
            random_state=cfg.random_state,
            stratify=train_val_df["label"],
        )

        self._train_df = train_df.reset_index(drop=True)
        self._val_df = val_df.reset_index(drop=True)
        self._test_df = test_df.reset_index(drop=True)

    @property
    def train_df(self) -> pd.DataFrame:
        assert self._train_df is not None
        return self._train_df

    @property
    def val_df(self) -> pd.DataFrame:
        assert self._val_df is not None
        return self._val_df

    @property
    def test_df(self) -> pd.DataFrame:
        assert self._test_df is not None
        return self._test_df

    # ----------------------------------------------------------------- sklearn
    def to_sklearn(
        self, split: str = "train"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (texts, labels) numpy arrays for a given split.

        Args:
            split: One of 'train', 'val', 'test'.

        Returns:
            (X, y) where X is an array of texts, y is an array of int labels.
        """
        if split == "train":
            df = self.train_df
        elif split == "val":
            df = self.val_df
        elif split == "test":
            df = self.test_df
        else:
            raise ValueError("split must be one of 'train', 'val', 'test'.")

        X = df["text"].astype(str).to_numpy()
        y = df["label"].astype(int).to_numpy()
        return X, y

    # ----------------------------------------------------------------- SMOTE
    def apply_smote(
        self, X: np.ndarray, y: np.ndarray, random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply SMOTE oversampling to balance classes.

        Args:
            X: Feature matrix or array of texts (will be treated as array-like).
            y: Label array.
            random_state: Random seed.

        Returns:
            (X_res, y_res): Resampled arrays.
        """
        # For text we apply SMOTE on numeric features downstream; here we assume
        # X is numerical (e.g. vectorized). This helper is primarily for that stage.
        smote = SMOTE(random_state=random_state)
        X_res, y_res = smote.fit_resample(X, y)
        return X_res, y_res

    # -------------------------------------------------------------- backtrans
    def augment_backtranslation(
        self, df: Optional[pd.DataFrame] = None, factor: float = 0.2
    ) -> pd.DataFrame:
        """Apply simple paraphrase-style augmentation.

        This is a lightweight stand-in for true EN→FR→EN back-translation.
        For now, it duplicates a fraction of samples and applies trivial
        transformations (e.g. adding a prefix), so the pipeline wiring
        can be tested without external APIs.

        Args:
            df: DataFrame to augment (defaults to train_df).
            factor: Fraction of rows to augment.

        Returns:
            Augmented DataFrame.
        """
        if df is None:
            df = self.train_df

        n_aug = max(1, int(len(df) * factor))
        subset = df.sample(n_aug, random_state=self.split_config.random_state)
        aug = subset.copy()
        aug["text"] = "[PARAPHRASE] " + aug["text"].astype(str)
        return pd.concat([df, aug], ignore_index=True)

    # --------------------------------------------------------------- PyTorch
    def to_pytorch_dataset(self, split: str = "train", tokenizer=None, max_length: int = 512):
        """Return a torch.utils.data.Dataset for a given split.

        Args:
            split: 'train', 'val', or 'test'.
            tokenizer: HuggingFace tokenizer with __call__(text, ...) interface.
            max_length: Max sequence length.

        Returns:
            torch.utils.data.Dataset instance.
        """
        if torch is None:
            raise ImportError("PyTorch not installed.")

        df = {"train": self.train_df, "val": self.val_df, "test": self.test_df}[split]

        class _BertDataset(TorchDataset):
            def __init__(self, frame: pd.DataFrame) -> None:
                self.frame = frame.reset_index(drop=True)

            def __len__(self) -> int:
                return len(self.frame)

            def __getitem__(self, idx: int) -> Dict[str, Any]:
                row = self.frame.iloc[idx]
                text = str(row["text"])
                enc = tokenizer(
                    text,
                    truncation=True,
                    padding="max_length",
                    max_length=max_length,
                    return_tensors="pt",
                )
                item = {
                    "input_ids": enc["input_ids"].squeeze(0),
                    "attention_mask": enc["attention_mask"].squeeze(0),
                    "label": int(row["label"]),
                }
                if "token_type_ids" in enc:
                    item["token_type_ids"] = enc["token_type_ids"].squeeze(0)
                return item

        return _BertDataset(df)

    # ------------------------------------------------------------- TensorFlow
    def to_tensorflow_dataset(
        self, split: str = "train", batch_size: int = 64
    ) -> "tf.data.Dataset":
        """Return a tf.data.Dataset for the given split.

        Args:
            split: 'train', 'val', or 'test'.
            batch_size: Batch size.

        Returns:
            tf.data.Dataset of (text, label).
        """
        if tf is None:
            raise ImportError("TensorFlow not installed.")

        X, y = self.to_sklearn(split)
        ds = tf.data.Dataset.from_tensor_slices(
            (tf.constant(X, dtype=tf.string), tf.constant(y, dtype=tf.int32))
        )
        return ds.batch(batch_size)
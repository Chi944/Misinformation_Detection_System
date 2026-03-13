import os
import json
import csv
import random
import numpy as np
from src.utils.logger import get_logger


class MisinformationDataset:
    """
    Loads, validates, and splits misinformation detection datasets.

    Expects CSV or JSON files with at minimum 'text' and 'label' columns.
    Labels must be 0 (credible) or 1 (misinformation).

    Args:
        data_path (str): path to CSV or JSON data file
        test_size (float): fraction held out for test set. Default 0.2
        val_size (float): fraction of train held out for val. Default 0.1
        random_seed (int): reproducibility seed. Default 42
    """

    REQUIRED_COLUMNS = ["text", "label"]

    def __init__(self, data_path=None, test_size=0.2, val_size=0.1, random_seed=42):
        self.data_path = data_path
        self.test_size = test_size
        self.val_size = val_size
        self.random_seed = random_seed
        self.logger = get_logger(__name__)
        self.df = None  # list of dicts
        self.train = []
        self.val = []
        self.test = []

        if data_path is not None:
            self.load(data_path)

    def load(self, data_path):
        """
        Load dataset from CSV or JSON file.

        Args:
            data_path (str): path to data file (.csv or .json)
        Raises:
            FileNotFoundError: if file does not exist
            ValueError: if required columns are missing
        """
        if not os.path.exists(data_path):
            raise FileNotFoundError("Dataset not found: %s" % data_path)
        ext = os.path.splitext(data_path)[1].lower()
        if ext == ".csv":
            rows = self._load_csv(data_path)
        elif ext == ".json":
            rows = self._load_json(data_path)
        else:
            raise ValueError("Unsupported file type: %s (use .csv or .json)" % ext)
        self._validate(rows)
        self.df = rows
        self.logger.info("Loaded %d samples from %s", len(rows), data_path)
        self._split()

    def _load_csv(self, path):
        """Load rows from CSV file."""
        rows = []
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(dict(row))
        return rows

    def _load_json(self, path):
        """Load rows from JSON file (list of dicts)."""
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "data" in data:
            return data["data"]
        raise ValueError('JSON must be a list of dicts or {"data": [...]}')

    def _validate(self, rows):
        """Check required columns and label values."""
        if not rows:
            raise ValueError("Dataset is empty")
        cols = set(rows[0].keys())
        for col in self.REQUIRED_COLUMNS:
            if col not in cols:
                raise ValueError("Missing required column: %s (found: %s)" % (col, cols))
        bad_labels = [r["label"] for r in rows if str(r["label"]).strip() not in ("0", "1", 0, 1)]
        if bad_labels:
            self.logger.warning("%d rows have unexpected labels", len(bad_labels))

    def _split(self):
        """Split df into train/val/test using stratified sampling."""
        random.seed(self.random_seed)
        credible = [r for r in self.df if str(r["label"]).strip() == "0" or r["label"] == 0]
        misinfo = [r for r in self.df if str(r["label"]).strip() == "1" or r["label"] == 1]
        random.shuffle(credible)
        random.shuffle(misinfo)

        def split_class(rows):
            n = len(rows)
            n_test = max(1, int(n * self.test_size))
            n_val = max(1, int((n - n_test) * self.val_size))
            return (
                rows[n_test + n_val :],
                rows[n_test : n_test + n_val],
                rows[:n_test],
            )

        tr_c, va_c, te_c = split_class(credible)
        tr_m, va_m, te_m = split_class(misinfo)

        self.train = tr_c + tr_m
        self.val = va_c + va_m
        self.test = te_c + te_m
        random.shuffle(self.train)
        random.shuffle(self.val)
        random.shuffle(self.test)
        self.logger.info(
            "Split: train=%d val=%d test=%d", len(self.train), len(self.val), len(self.test)
        )

    def to_sklearn(self, split="train"):
        """
        Return texts and integer labels for sklearn-compatible usage.

        Args:
            split (str): 'train', 'val', or 'test'
        Returns:
            tuple: (texts list, labels list of int)
        """
        rows = getattr(self, split, [])
        texts = [r["text"] for r in rows]
        labels = [int(r["label"]) for r in rows]
        return texts, labels

    def to_torch(self, split="train"):
        """
        Return texts and integer labels for PyTorch DataLoader usage.

        Args:
            split (str): 'train', 'val', or 'test'
        Returns:
            list of (text, label) tuples
        """
        texts, labels = self.to_sklearn(split)
        return list(zip(texts, labels))

    def get_stats(self):
        """
        Return dataset statistics.

        Returns:
            dict: total, credible, misinfo counts per split
        """

        def stats(rows):
            total = len(rows)
            credible = sum(1 for r in rows if int(r["label"]) == 0)
            misinfo = total - credible
            return {
                "total": total,
                "credible": credible,
                "misinfo": misinfo,
                "balance": round(credible / total, 3) if total else 0.0,
            }

        return {
            "train": stats(self.train),
            "val": stats(self.val),
            "test": stats(self.test),
            "all": stats(self.df or []),
        }

    def create_synthetic(self, n_samples=200, seed=42):
        """
        Create a synthetic dataset for smoke tests and CI.

        Generates simple keyword-based samples — not realistic, but
        sufficient to verify model pipelines work end-to-end.

        Args:
            n_samples (int): total samples to generate
            seed (int): random seed
        Returns:
            self: allows chaining
        """
        random.seed(seed)
        credible_phrases = [
            "Scientists publish peer-reviewed study",
            "According to official government data",
            "Researchers at university confirm findings",
            "Clinical trial shows moderate effect",
            "Expert panel reviews evidence carefully",
            "Statistical analysis of census data",
            "Independent audit verifies results",
            "Peer reviewed journal publishes findings",
        ]
        misinfo_phrases = [
            "SHOCKING secret they do not want you to know",
            "Doctors HATE this one weird trick",
            "Government cover-up exposed by insiders",
            "Scientists BAFFLED by miracle cure found",
            "Mainstream media hiding the truth about",
            "They are putting chemicals in the water",
            "Illuminati controls all world governments",
            "Vaccines contain microchips for tracking",
        ]
        rows = []
        for i in range(n_samples // 2):
            phrase = random.choice(credible_phrases)
            rows.append(
                {
                    "text": "%s in study number %d." % (phrase, i),
                    "label": 0,
                    "category": "credible",
                }
            )
        for i in range(n_samples // 2):
            phrase = random.choice(misinfo_phrases)
            rows.append(
                {
                    "text": "%s claim number %d!" % (phrase, i),
                    "label": 1,
                    "category": "misinformation",
                }
            )
        random.shuffle(rows)
        self.df = rows
        self._split()
        self.logger.info("Synthetic dataset created: %d samples", len(rows))
        return self

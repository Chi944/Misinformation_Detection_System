"""Minimal evaluation script for the misinformation detector.

For now this script simply loads the trained models via
``src.detector.MisinformationDetector`` and runs predictions over the test
split of ``data/sample_train.csv``, computing aggregate accuracy. Later phases
will replace this with the full :mod:`src.evaluation.pipeline` orchestration.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.detector import MisinformationDetector
from src.training.dataset import MisinformationDataset


def main() -> None:
    """Run a quick evaluation on the held-out test split."""

    data_path = "data/sample_train.csv"
    ds = MisinformationDataset(data_path)
    X_test, y_test = ds.to_sklearn("test")

    detector = MisinformationDetector(config="config.yaml", fast_mode=True)

    preds = []
    for text in X_test:
        out = detector.predict(str(text))
        label = out["ensemble"]["label"]
        preds.append(1 if label == "misinformation" else 0)

    y_pred = np.asarray(preds, dtype=int)
    acc = float((y_pred == y_test).mean()) if len(y_test) else 0.0

    print(f"[evaluate] Accuracy on test split: {acc:.3f}")


if __name__ == "__main__":
    main()


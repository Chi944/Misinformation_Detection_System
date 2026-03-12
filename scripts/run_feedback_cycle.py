"""Run a single feedback cycle for the misinformation detector.

This script loads the detector, reads a small batch from ``data/sample_train.csv``,
and executes one feedback loop cycle using the provided ground-truth labels.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.detector import MisinformationDetector


def main() -> None:
    """Execute one feedback cycle on a small batch."""

    data_path = Path("data/sample_train.csv")
    if not data_path.exists():
        raise SystemExit(
            "data/sample_train.csv not found. Run scripts/download_sample_data.py first."
        )

    df = pd.read_csv(data_path)
    if "text" not in df or "label" not in df:
        raise SystemExit("sample_train.csv must contain 'text' and 'label' columns.")

    batch = df.sample(min(64, len(df)), random_state=42)
    texts = batch["text"].astype(str).tolist()
    labels = batch["label"].astype(int).tolist()

    detector = MisinformationDetector(config="config.yaml", fast_mode=True)
    loop = detector.feedback_loop

    import asyncio

    summary = asyncio.run(loop.run_cycle(texts, labels))
    print(f"Feedback cycle completed: {summary}")


if __name__ == "__main__":
    main()


"""Train all misinformation detection models and build the ensemble.

This script is a thin wrapper around :class:`src.training.trainer.Trainer`.
It trains the Naive Bayes, TF-IDF DNN, and BERT models on the sample dataset,
builds the ensemble, and writes a JSON summary that can be consumed by CI
accuracy gates.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.training.trainer import Trainer


def main() -> None:
    """Entry point for training script."""

    trainer = Trainer(
        data_path="data/sample_train.csv",
        config_path="config.yaml",
        models_dir="models",
    )
    summary = trainer.train_all()

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / "accuracy_gate_results.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[train_all] Training complete. Metrics written to {out_path}")


if __name__ == "__main__":
    main()


import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.dataset import MisinformationDataset  # noqa: E402
from src.training.trainer import MasterTrainer  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger("train_all")


def parse_args():
    """Parse command line arguments for train_all.py."""
    parser = argparse.ArgumentParser(description="Train all models in the misinformation detector")
    parser.add_argument(
        "--data", default="data/train.csv", help="Path to training data CSV or JSON"
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic data (for CI/testing when real data absent)",
    )
    parser.add_argument(
        "--n-synthetic",
        type=int,
        default=500,
        help="Number of synthetic samples when --synthetic is used",
    )
    parser.add_argument(
        "--skip-gates", action="store_true", help="Skip accuracy gate checks (for debugging only)"
    )
    parser.add_argument(
        "--skip-bert",
        action="store_true",
        help=(
            "Skip BERT training (fast mode). "
            "NB + TF-IDF give ~0.85 accuracy in 5-10 minutes. "
            "Full BERT training takes 20 min on GPU or 2-4 hrs on CPU. "
            "Train BERT on Kaggle (free GPU) if no local GPU available."
        ),
    )
    return parser.parse_args()


def main():
    """
    Main training entry point.

    Loads dataset, trains all 3 models via MasterTrainer, applies
    accuracy gates, and saves model checkpoints.
    """
    args = parse_args()
    logger.info("Starting train_all.py")

    # Load or create dataset
    dataset = MisinformationDataset()
    if args.synthetic:
        logger.info("Using synthetic dataset (%d samples)", args.n_synthetic)
        dataset.create_synthetic(n_samples=args.n_synthetic)
    elif os.path.exists(args.data):
        logger.info("Loading dataset from %s", args.data)
        dataset.load(args.data)
    else:
        logger.warning("Data file not found: %s — using synthetic fallback", args.data)
        dataset.create_synthetic(n_samples=500)

    stats = dataset.get_stats()
    logger.info(
        "Dataset stats: train=%d val=%d test=%d",
        stats["train"]["total"],
        stats["val"]["total"],
        stats["test"]["total"],
    )

    # Train all models
    try:
        trainer = MasterTrainer(config_path=args.config)
        results = trainer.train_all(
            dataset,
            skip_gates=args.skip_gates,
            skip_bert=args.skip_bert,
        )
        logger.info("Training complete. Results:")
        for model_name, metrics in results.items():
            logger.info(
                "  %s -> acc=%.4f f1=%.4f",
                model_name,
                metrics.get("accuracy", 0.0),
                metrics.get("f1", 0.0),
            )
        print("train_all.py complete - all models trained successfully")
    except Exception as e:
        logger.error("Training failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()

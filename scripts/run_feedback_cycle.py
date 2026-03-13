import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.detector import MisinformationDetector  # noqa: E402
from src.training.dataset import MisinformationDataset  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger("run_feedback_cycle")


def parse_args():
    """Parse run_feedback_cycle.py arguments."""
    parser = argparse.ArgumentParser(description="Run one backward propagation feedback cycle")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--n-samples", type=int, default=50)
    return parser.parse_args()


def main():
    """Run one feedback cycle on a batch of samples."""
    args = parse_args()
    logger.info("Starting run_feedback_cycle.py")

    dataset = MisinformationDataset()
    dataset.create_synthetic(n_samples=args.n_samples * 2)
    texts, labels = dataset.to_sklearn("train")
    batch = texts[: args.n_samples]
    lbls = labels[: args.n_samples]

    detector = MisinformationDetector(config=args.config, fast_mode=True)

    if detector.feedback_loop is None:
        logger.error("Feedback loop not initialised")
        sys.exit(1)

    logger.info("Running feedback cycle on %d samples", len(batch))
    try:
        metrics = detector.feedback_loop.run_cycle(batch, true_labels=lbls)
        print("")
        print("=== Feedback Cycle Complete ===")
        print("Cycle metrics: %s" % metrics)
        print("=== Done ===")
    except Exception as e:
        logger.error("Feedback cycle failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()

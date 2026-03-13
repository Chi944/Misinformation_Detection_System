import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.detector import MisinformationDetector  # noqa: E402
from src.training.dataset import MisinformationDataset  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger("evaluate")


def parse_args():
    """Parse evaluate.py arguments."""
    parser = argparse.ArgumentParser(description="Evaluate misinformation detector")
    parser.add_argument("--data", default="data/test.csv")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--no-judge", action="store_true", help="Skip LLM judge evaluation")
    return parser.parse_args()


def main():
    """Run evaluation pipeline and print summary report."""
    args = parse_args()
    logger.info("Starting evaluate.py")

    dataset = MisinformationDataset()
    if args.synthetic:
        logger.info("Using synthetic dataset")
        dataset.create_synthetic(n_samples=200)
    elif os.path.exists(args.data):
        dataset.load(args.data)
    else:
        logger.warning("Data not found: %s - using synthetic fallback", args.data)
        dataset.create_synthetic(n_samples=200)

    detector = MisinformationDetector(config=args.config, fast_mode=False)

    use_judge = not args.no_judge
    report = detector.evaluate(dataset, use_llm_judge=use_judge)

    print("")
    print("=== Evaluation Report ===")
    for model_name, metrics in report.get("model_metrics", {}).items():
        std = metrics.get("standard", {})
        print(
            "%s -> acc=%.4f precision=%.4f recall=%.4f f1=%.4f"
            % (
                model_name.upper(),
                std.get("accuracy", 0.0),
                std.get("precision", 0.0),
                std.get("recall", 0.0),
                std.get("f1", 0.0),
            )
        )
    print("Samples evaluated: %d" % report.get("sample_count", 0))
    report_path = os.path.join("reports", "evaluation_report.json")
    if os.path.exists(report_path):
        print("Full report saved to: %s" % report_path)
    print("=== Done ===")


if __name__ == "__main__":
    main()

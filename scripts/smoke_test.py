import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.logger import get_logger

logger = get_logger("smoke_test")

SAMPLE_TEXTS = [
    "Scientists publish peer-reviewed study on climate change.",
    "SHOCKING secret big pharma does not want you to know!",
    "University researchers confirm vaccine safety in trial.",
    "Government hiding alien contact evidence from the public.",
    "New economic data shows unemployment rate at record low.",
]
EXPECTED_KEYS = [
    "crisp_label",
    "ensemble_probability",
    "fuzzy_score",
    "model_breakdown",
    "model_agreement",
]


def parse_args():
    """Parse smoke test arguments."""
    parser = argparse.ArgumentParser(description="Smoke test for misinformation detector pipeline")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data mode")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    return parser.parse_args()


def check_ollama():
    """
    Check if Ollama is running on localhost:11434.

    Returns:
        bool: True if reachable
    """
    try:
        import requests

        resp = requests.get("http://localhost:11434/api/tags", timeout=3)
        return resp.status_code == 200
    except Exception:
        return False


def run_smoke_test(config_path="config.yaml"):
    """
    Run all smoke tests and return (passed, failed, errors) counts.

    Args:
        config_path (str): path to config.yaml
    Returns:
        tuple: (passed int, failed int, list of error strings)
    """
    passed = 0
    failed = 0
    errors = []

    # 1. Import checks
    logger.info("--- Import checks ---")
    import_checks = [
        (
            "FuzzyMisinformationEngine",
            "from src.fuzzy.fuzzy_engine import FuzzyMisinformationEngine",
        ),
        ("MisinformationDataset", "from src.training.dataset import MisinformationDataset"),
        ("MetricsCalculator", "from src.evaluation.metrics import MetricsCalculator"),
        ("MisinformationDetector", "from src.detector import MisinformationDetector"),
    ]
    for name, stmt in import_checks:
        try:
            exec(stmt)
            logger.info("  PASS import: %s", name)
            passed += 1
        except Exception as e:
            logger.error("  FAIL import: %s - %s", name, e)
            errors.append("import %s: %s" % (name, e))
            failed += 1

    # 2. Fuzzy engine
    logger.info("--- Fuzzy engine ---")
    try:
        from src.fuzzy.fuzzy_engine import FuzzyMisinformationEngine

        engine = FuzzyMisinformationEngine()
        score = engine.compute(
            {
                "source_credibility": 0.8,
                "bert_confidence": 0.9,
                "tfidf_confidence": 0.85,
                "nb_confidence": 0.7,
                "model_agreement": 1.0,
                "feedback_score": 0.5,
            }
        )
        assert 0.5 <= score <= 1.0, "Expected high misinfo score"
        logger.info("  PASS fuzzy engine score=%.4f", score)
        passed += 1
    except Exception as e:
        logger.error("  FAIL fuzzy engine: %s", e)
        errors.append("fuzzy: %s" % e)
        failed += 1

    # 3. Dataset
    logger.info("--- Dataset ---")
    try:
        from src.training.dataset import MisinformationDataset

        ds = MisinformationDataset()
        ds.create_synthetic(n_samples=60)
        texts, labels = ds.to_sklearn("train")
        assert len(texts) > 0
        assert all(label in (0, 1) for label in labels)
        logger.info("  PASS dataset train=%d", len(texts))
        passed += 1
    except Exception as e:
        logger.error("  FAIL dataset: %s", e)
        errors.append("dataset: %s" % e)
        failed += 1

    # 4. Detector predict (fast_mode)
    logger.info("--- Detector predict ---")
    try:
        from src.detector import MisinformationDetector

        detector = MisinformationDetector(config=config_path, fast_mode=True)
        for text in SAMPLE_TEXTS:
            result = detector.predict(text)
            for key in EXPECTED_KEYS:
                assert key in result, "Missing key: %s" % key
            assert result["crisp_label"] in ("credible", "misinformation")
            assert 0.0 <= result["ensemble_probability"] <= 1.0
            assert 0.0 <= result["fuzzy_score"] <= 1.0
            logger.info(
                "  PASS predict: label=%s prob=%.4f fuzzy=%.4f",
                result["crisp_label"],
                result["ensemble_probability"],
                result["fuzzy_score"],
            )
        passed += 1
    except Exception as e:
        logger.error("  FAIL detector predict: %s", e)
        errors.append("predict: %s" % e)
        failed += 1

    # 5. Metrics
    logger.info("--- Metrics ---")
    try:
        from src.evaluation.metrics import MetricsCalculator

        calc = MetricsCalculator()
        y_true = [0, 1, 0, 1, 1]
        y_pred = [0, 1, 0, 0, 1]
        m = calc.compute_standard_metrics(y_true, y_pred)
        assert 0.0 <= m["accuracy"] <= 1.0
        assert 0.0 <= m["f1"] <= 1.0
        logger.info("  PASS metrics acc=%.4f f1=%.4f", m["accuracy"], m["f1"])
        passed += 1
    except Exception as e:
        logger.error("  FAIL metrics: %s", e)
        errors.append("metrics: %s" % e)
        failed += 1

    # 6. Ollama health check
    logger.info("--- Ollama health check ---")
    if check_ollama():
        logger.info("  PASS Ollama running on port 11434")
        passed += 1
    else:
        logger.warning(
            "  SKIP Ollama not running " "(start with: ollama serve && ollama pull llama3)"
        )

    return passed, failed, errors


def main():
    """Run smoke tests and exit 0 on pass, 1 on failure."""
    args = parse_args()
    passed, failed, errors = run_smoke_test(args.config)
    total = passed + failed
    print("")
    print("Smoke test results: %d/%d passed" % (passed, total))
    if errors:
        print("Failures:")
        for e in errors:
            print("  - %s" % e)
    if failed > 0:
        sys.exit(1)
    else:
        print("ALL SMOKE TESTS PASS")
        sys.exit(0)


if __name__ == "__main__":
    main()

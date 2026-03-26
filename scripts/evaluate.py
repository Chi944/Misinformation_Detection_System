import argparse
import os
import sys
from pathlib import Path
from collections import Counter
import json

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
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Evaluate only the first N samples (debugging). Default: all samples.",
    )
    return parser.parse_args()

def _project_root() -> Path:
    # scripts/evaluate.py -> project root
    return Path(__file__).resolve().parents[1]


def _load_saved_weights(detector: MisinformationDetector) -> None:
    """
    Explicitly load all saved model artefacts from ./models.

    This avoids scenarios where lazy loading uses an unexpected CWD or fails
    silently, which can otherwise lead to near-constant 0.5 predictions.
    """
    project_root = _project_root()
    models_dir = project_root / "models"

    # --- BERT
    from src.models.bert_classifier import BERTClassifier

    bert_cfg = (detector.config.get("models", {}) or {}).get("bert", {}) if isinstance(detector.config, dict) else {}
    bert_path = models_dir / "bert_classifier.pt"
    if not bert_path.exists():
        raise FileNotFoundError(f"Missing BERT weights: {bert_path}")

    bert_model = BERTClassifier(config=bert_cfg)
    # Ensure tokenizer exists (used by EnsembleDetector for inference).
    bert_tokenizer = getattr(bert_model, "tokenizer", None)
    bert_model.to(detector.device)
    bert_model.eval()
    bert_model.load(str(bert_path))

    # --- TF-IDF
    from src.models.tfidf_model import TFIDFModel
    import joblib
    import tensorflow as tf

    tfidf_keras_path = models_dir / "tfidf_model.keras"
    tfidf_vec_path = models_dir / "tfidf_vectorizer.joblib"
    if not tfidf_keras_path.exists():
        raise FileNotFoundError(f"Missing TF-IDF Keras model: {tfidf_keras_path}")
    if not tfidf_vec_path.exists():
        raise FileNotFoundError(f"Missing TF-IDF vectorizer: {tfidf_vec_path}")

    tfidf_cfg = (detector.config.get("models", {}) or {}).get("tfidf", {}) if isinstance(detector.config, dict) else {}
    tfidf_model = TFIDFModel(models_dir="models", config=tfidf_cfg)
    tfidf_model.model = tf.keras.models.load_model(str(tfidf_keras_path))
    vecs = joblib.load(str(tfidf_vec_path))
    # TFIDFModel expects dict entries: {"word": <vectorizer>, "char": <vectorizer>}
    tfidf_model.word_vectorizer = vecs.get("word")
    tfidf_model.char_vectorizer = vecs.get("char")

    # --- Naive Bayes
    from src.models.naive_bayes_model import TFNaiveBayesWrapper

    nb_model = TFNaiveBayesWrapper()
    nb_path = models_dir / "naive_bayes.pkl"
    nb_vec_path = models_dir / "nb_vectorizer.pkl"
    if not nb_path.exists():
        raise FileNotFoundError(f"Missing Naive Bayes model: {nb_path}")
    if not nb_vec_path.exists():
        raise FileNotFoundError(f"Missing Naive Bayes vectorizer: {nb_vec_path}")

    nb_model._calibrated_clf = joblib.load(str(nb_path))
    nb_model.vectorizer = joblib.load(str(nb_vec_path))

    # --- Rebuild ensemble so it references our loaded instances.
    from src.models.ensemble_detector import EnsembleDetector

    detector.bert_model = bert_model
    detector.bert_tokenizer = bert_tokenizer
    detector.tfidf_model = tfidf_model
    detector.nb_model = nb_model
    detector.ensemble = EnsembleDetector(
        config=detector.config,
        bert_model=detector.bert_model,
        tfidf_model=detector.tfidf_model,
        nb_model=detector.nb_model,
        bert_tokenizer=detector.bert_tokenizer,
        device=detector.device,
    )


def _unpack_for_eval(detector: MisinformationDetector, dataset: MisinformationDataset):
    # Mirror EvaluationPipeline._unpack_dataset without importing protected members.
    if hasattr(dataset, "df") and isinstance(dataset.df, list) and dataset.df and isinstance(dataset.df[0], dict):
        texts = [r["text"] for r in dataset.df]
        y_true = [int(r["label"]) for r in dataset.df]
        cats = [r.get("category") for r in dataset.df] if "category" in dataset.df[0] else None
        return texts, y_true, cats
    # Fallback: iterate dataset items (works for list/tuple forms)
    texts, y_true, cats = [], [], []
    for item in dataset:
        texts.append(item[0])
        y_true.append(int(item[1]))
        if len(item) >= 3:
            cats.append(item[2])
    return texts, y_true, cats if cats else None


def main():
    """Run evaluation pipeline and print summary report."""
    args = parse_args()
    logger.info("Starting evaluate.py")

    # Make sure relative paths (models/, data/, reports/) resolve correctly.
    root = _project_root()
    os.chdir(root)

    # Load dataset (test split expected at ./data/test.csv).
    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = root / data_path

    # If the provided data file doesn't exist, we fall back to synthetic so
    # this script can still run in CI/limited environments.
    dataset = MisinformationDataset()
    if args.synthetic:
        logger.info("Using synthetic dataset")
        dataset.create_synthetic(n_samples=200)
    elif data_path.exists():
        dataset.load(str(data_path))
    else:
        logger.warning("Data not found: %s - using synthetic fallback", data_path)
        dataset.create_synthetic(n_samples=200)

    detector = MisinformationDetector(config=args.config, fast_mode=False)

    # Explicitly load all saved weight artefacts to avoid lazy-loading/CWD issues.
    _load_saved_weights(detector)

    # Evaluate with the same pipeline components, but with a pre-plot sanity check.
    pipeline = detector.eval_pipeline
    if pipeline is None:
        raise RuntimeError("Evaluation pipeline not initialised")

    use_judge = not args.no_judge
    texts, y_true, categories = _unpack_for_eval(detector, dataset)

    if args.max_samples is not None:
        texts = texts[: args.max_samples]
        y_true = y_true[: args.max_samples]
        if categories is not None:
            categories = categories[: args.max_samples]

    logger.info("Running predictions for %d samples (no-judge=%s)", len(texts), not use_judge)
    all_preds = [detector.predict(t) for t in texts]

    # --- Sanity check: predictions must not collapse to one class.
    # For very small debug subsets it's possible to sample only one class even
    # with a working model, so only hard-fail for sufficiently large runs.
    strict_diversity_check = args.max_samples is None or len(texts) >= 200

    def _check_not_all_same(model_key, labels):
        c = Counter(labels)
        if len(c) < 2:
            msg = (
                f"Prediction collapse detected for {model_key}: counts={dict(c)}. "
                f"This usually means saved weights were not loaded correctly."
            )
            if strict_diversity_check:
                raise RuntimeError(msg)
            logger.warning(msg)

    _check_not_all_same("bert", [p["model_breakdown"]["bert"]["label"] for p in all_preds])
    _check_not_all_same("tfidf", [p["model_breakdown"]["tfidf"]["label"] for p in all_preds])
    _check_not_all_same("naive_bayes", [p["model_breakdown"]["naive_bayes"]["label"] for p in all_preds])
    _check_not_all_same(
        "ensemble",
        [1 if p["crisp_label"] == "misinformation" else 0 for p in all_preds],
    )

    eval_data = pipeline._build_eval_data(all_preds, y_true)

    # Optional LLM judge metrics (may be slow; controlled by --no-judge).
    judge_metrics = {}
    if use_judge and detector.llm_judge is not None:
        try:
            fuzzy_scores = [
                detector.fuzzy_engine.compute(
                    {
                        "source_credibility": 0.5,
                        "bert_confidence": p["model_breakdown"]["bert"]["confidence"],
                        "tfidf_confidence": p["model_breakdown"]["tfidf"]["confidence"],
                        "nb_confidence": p["model_breakdown"]["naive_bayes"]["confidence"],
                        "model_agreement": p["model_agreement"],
                        "feedback_score": 0.5,
                    }
                )
                for p in all_preds
            ]
            preds_for_judge = [
                {
                    "bert": p["model_breakdown"]["bert"]["label"],
                    "tfidf": p["model_breakdown"]["tfidf"]["label"],
                    "naive_bayes": p["model_breakdown"]["naive_bayes"]["label"],
                    "ensemble": p["crisp_label"],
                }
                for p in all_preds
            ]
            judgments = detector.llm_judge.evaluate_batch(list(zip(texts, preds_for_judge, fuzzy_scores)))
            judge_metrics = pipeline.calculator.compute_judge_metrics(judgments, y_true)
            eval_data["judge_metrics"] = judge_metrics
            eval_data["judge_report"] = detector.llm_judge.generate_model_report(judgments)
        except Exception as e:
            logger.warning("LLM judge failed: %s", e)

    model_metrics = {}
    for name in ["bert", "tfidf", "naive_bayes", "ensemble"]:
        data = eval_data["models"].get(name, {})
        model_metrics[name] = pipeline.calculator.compute_all(
            name,
            y_true,
            data.get("y_pred", []),
            data.get("y_prob"),
            categories=categories,
        )

    report = {
        "model_metrics": model_metrics,
        "judge_metrics": judge_metrics,
        "sample_count": len(texts),
    }

    # Save report + generate plots.
    report_path = os.path.join(pipeline.output_dir, "evaluation_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Report saved to %s", report_path)
    pipeline.dashboard.generate(eval_data)

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

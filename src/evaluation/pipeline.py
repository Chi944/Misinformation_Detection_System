import json
import os

from src.evaluation.metrics import MetricsCalculator
from src.utils.logger import get_logger


class EvaluationPipeline:
    """
    Orchestrates full model evaluation: metrics, LLM judge, dashboard.
    Part of MisinformationDetector.

    Args:
        detector: MisinformationDetector instance
        config (dict): loaded config.yaml
    """

    def __init__(self, detector, config):
        self.detector = detector
        self.config = config
        self.calculator = MetricsCalculator()
        self.output_dir = config.get("evaluation", {}).get("output_dir", "reports")
        # Lazy import to avoid slow matplotlib import at module import time.
        from src.evaluation.dashboard import EvaluationDashboard

        self.dashboard = EvaluationDashboard(output_dir=self.output_dir)
        self.logger = get_logger(__name__)
        os.makedirs(self.output_dir, exist_ok=True)

    def evaluate(self, dataset, use_llm_judge=True):
        """
        Run full evaluation pipeline.

        Args:
            dataset: MisinformationDataset or list of (text, label) tuples
            use_llm_judge (bool): whether to call LLM judge
        Returns:
            dict: full evaluation report
        """
        self.logger.info("Starting evaluation (llm_judge=%s)", use_llm_judge)
        texts, y_true, categories = self._unpack_dataset(dataset)
        all_preds = [self.detector.predict(t) for t in texts]
        eval_data = self._build_eval_data(all_preds, y_true)

        judge_metrics = {}
        if use_llm_judge and self.detector.llm_judge is not None:
            try:
                fuzzy_scores = [
                    self.detector.fuzzy_engine.compute(
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
                judgments = self.detector.llm_judge.evaluate_batch(
                    list(zip(texts, preds_for_judge, fuzzy_scores))
                )
                judge_metrics = self.calculator.compute_judge_metrics(judgments, y_true)
                eval_data["judge_metrics"] = judge_metrics
                eval_data["judge_report"] = self.detector.llm_judge.generate_model_report(judgments)
            except Exception as e:
                self.logger.warning("LLM judge failed: %s", e)

        model_metrics = {}
        for name in ["bert", "tfidf", "naive_bayes", "ensemble"]:
            data = eval_data["models"].get(name, {})
            model_metrics[name] = self.calculator.compute_all(
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

        report_path = os.path.join(self.output_dir, "evaluation_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        self.logger.info("Report saved to %s", report_path)

        try:
            self.dashboard.generate(eval_data)
        except Exception as e:
            self.logger.warning("Dashboard failed: %s", e)

        return report

    def _unpack_dataset(self, dataset):
        """Unpack dataset into texts, labels, categories."""
        if hasattr(dataset, "df"):
            df = dataset.df
            if hasattr(df, "columns"):
                cats = df["category"].tolist() if "category" in df.columns else None
                return df["text"].tolist(), [int(x) for x in df["label"].tolist()], cats
            # list of dicts (MisinformationDataset)
            if df and isinstance(df[0], dict):
                texts = [r["text"] for r in df]
                labels = [int(r["label"]) for r in df]
                cats = [r.get("category") for r in df] if df and "category" in df[0] else None
                return texts, labels, cats

        texts, y_true, cats = [], [], []
        for item in dataset:
            if len(item) >= 3:
                texts.append(item[0])
                y_true.append(int(item[1]))
                cats.append(item[2])
            else:
                texts.append(item[0])
                y_true.append(int(item[1]))
        return texts, y_true, cats if cats else None

    def _build_eval_data(self, all_preds, y_true):
        """Build eval_data dict for dashboard and metrics."""
        eval_data = {"models": {}}
        for name in ["bert", "tfidf", "naive_bayes"]:
            eval_data["models"][name] = {
                "y_true": y_true,
                "y_pred": [p["model_breakdown"][name]["label"] for p in all_preds],
                "y_prob": [p["model_breakdown"][name]["confidence"] for p in all_preds],
            }
        eval_data["models"]["ensemble"] = {
            "y_true": y_true,
            "y_pred": [1 if p["crisp_label"] == "misinformation" else 0 for p in all_preds],
            "y_prob": [p["ensemble_probability"] for p in all_preds],
        }
        return eval_data

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
from src.utils.logger import get_logger


class MetricsCalculator:
    """
    Computes all evaluation metrics for each model and the ensemble.
    Part of the EvaluationPipeline.
    """

    def __init__(self):
        self.logger = get_logger(__name__)

    def compute_standard_metrics(self, y_true, y_pred, y_prob=None):
        """
        Compute accuracy, precision, recall, F1, ROC-AUC, PR-AUC.

        Args:
            y_true (list): ground truth labels 0/1
            y_pred (list): predicted labels 0/1
            y_prob (list, optional): predicted probabilities for class 1
        Returns:
            dict: standard classification metrics
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        result = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "roc_auc": 0.5,
            "pr_auc": 0.5,
        }
        if y_prob is not None:
            y_prob_arr = np.array(y_prob)
            try:
                result["roc_auc"] = float(roc_auc_score(y_true, y_prob_arr))
                result["pr_auc"] = float(average_precision_score(y_true, y_prob_arr))
            except Exception as e:
                self.logger.warning("AUC computation failed: %s", e)
        return result

    def compute_ece(self, y_true, y_prob, n_bins=10):
        """
        Compute Expected Calibration Error.

        Args:
            y_true (list): ground truth labels
            y_prob (list): predicted probabilities for class 1
            n_bins (int): number of calibration bins
        Returns:
            float: ECE score, lower is better
        """
        y_true = np.array(y_true)
        y_prob = np.array(y_prob)
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        n = len(y_true)
        for i in range(n_bins):
            mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
            if mask.sum() == 0:
                continue
            acc = float(y_true[mask].mean())
            conf = float(y_prob[mask].mean())
            ece += (mask.sum() / n) * abs(acc - conf)
        return float(ece)

    def compute_fuzzy_metrics(self, fuzzy_scores, y_true, tolerance=0.15):
        """
        Compute fuzzy-specific accuracy metrics.

        Args:
            fuzzy_scores (list): fuzzy engine outputs per sample
            y_true (list): ground truth labels
            tolerance (float): acceptable deviation threshold
        Returns:
            dict: mean_fuzzy_score, fuzzy_accuracy, fuzzy_std
        """
        scores = np.array(fuzzy_scores)
        labels = np.array(y_true, dtype=float)
        within = np.abs(scores - labels) <= tolerance
        return {
            "mean_fuzzy_score": float(scores.mean()),
            "fuzzy_std": float(scores.std()),
            "fuzzy_accuracy": float(within.mean()),
        }

    def compute_judge_metrics(self, judgments, y_true):
        """
        Compute LLM judge agreement statistics.

        Args:
            judgments (list): judgment dicts from LLMJudge
            y_true (list): ground truth labels
        Returns:
            dict: agreement_rate, mean_confidence, disagreement_cases
        """
        if not judgments:
            return {}
        verdicts = [1 if j.get("independent_verdict") == "misinformation" else 0 for j in judgments]
        agreement = [v == int(t) for v, t in zip(verdicts, y_true)]
        disagreements = [
            {
                "index": i,
                "judge": verdicts[i],
                "true": int(y_true[i]),
            }
            for i, a in enumerate(agreement)
            if not a
        ]
        confidences = [float(j.get("judge_confidence", 0.0)) for j in judgments]
        return {
            "agreement_rate": float(np.mean(agreement)),
            "mean_confidence": float(np.mean(confidences)),
            "disagreement_count": len(disagreements),
            "disagreement_cases": disagreements[:10],
            "verdict_distribution": {
                "credible": verdicts.count(0),
                "misinformation": verdicts.count(1),
            },
        }

    def compute_all(
        self,
        model_name,
        y_true,
        y_pred,
        y_prob=None,
        fuzzy_scores=None,
        judgments=None,
        categories=None,
    ):
        """
        Compute the full metrics suite for one model.

        Args:
            model_name (str): label for logging
            y_true (list): ground truth labels
            y_pred (list): predicted labels
            y_prob (list, optional): predicted probabilities
            fuzzy_scores (list, optional): fuzzy engine outputs
            judgments (list, optional): LLM judge dicts
            categories (list, optional): per-sample category strings
        Returns:
            dict: all computed metrics
        """
        self.logger.info("Computing metrics for %s", model_name)
        result = {}
        result["standard"] = self.compute_standard_metrics(y_true, y_pred, y_prob)
        if y_prob is not None:
            result["ece"] = self.compute_ece(y_true, y_prob)
        if fuzzy_scores is not None:
            result["fuzzy"] = self.compute_fuzzy_metrics(fuzzy_scores, y_true)
        if judgments is not None:
            result["llm_judge"] = self.compute_judge_metrics(judgments, y_true)
        if categories is not None:
            result["per_category"] = self._per_category_breakdown(y_true, y_pred, categories)
        return result

    def _per_category_breakdown(self, y_true, y_pred, categories):
        """
        Compute accuracy and F1 per category.

        Args:
            y_true (list): ground truth labels
            y_pred (list): predicted labels
            categories (list): category string per sample
        Returns:
            dict: category -> accuracy, f1, count
        """
        breakdown = {}
        for cat in sorted(set(categories)):
            idx = [i for i, c in enumerate(categories) if c == cat]
            if not idx:
                continue
            yt = [y_true[i] for i in idx]
            yp = [y_pred[i] for i in idx]
            breakdown[cat] = {
                "count": len(idx),
                "accuracy": float(accuracy_score(yt, yp)),
                "f1": float(f1_score(yt, yp, zero_division=0)),
            }
        return breakdown

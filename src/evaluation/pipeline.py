"""Evaluation pipeline orchestration.

This module belongs to the *evaluation* component of the pipeline. It
coordinates metric computation, optional LLM-based judging, and dashboard
generation, and writes a JSON report to the reports directory.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

from src.evaluation.llm_judge import LLMJudge
from src.evaluation import dashboard, metrics as eval_metrics
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)


class EvaluationPipeline:
    """End-to-end evaluation pipeline for the detector.

    Component:
        Evaluation / Pipeline.
    """

    def __init__(
        self,
        detector,
        judge: Optional[LLMJudge] = None,
        output_dir: str | Path = "reports",
    ) -> None:
        self.detector = detector
        self.judge = judge
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(
        self,
        texts: Sequence[str],
        labels: Sequence[int],
        use_llm_judge: bool = True,
    ) -> Dict[str, Any]:
        """Evaluate all models and optionally call the LLM judge.

        Args:
            texts: Input texts.
            labels: Ground-truth labels (0/1).
            use_llm_judge: Whether to call the LLM judge.

        Returns:
            Dict with metrics, fuzzy stats, and optional LLM judge summary.
        """

        y_true = np.asarray(list(labels), dtype=int)
        probs_by_model: Dict[str, List[List[float]]] = {
            "bert": [],
            "tfidf": [],
            "naive_bayes": [],
            "ensemble": [],
        }
        fuzzy_scores: List[float] = []

        for text in texts:
            out = self.detector.predict(text)
            mb = out["model_breakdown"]
            probs_by_model["bert"].append(
                [1.0 - mb["bert"]["confidence"], mb["bert"]["confidence"]]
            )
            probs_by_model["tfidf"].append(
                [1.0 - mb["tfidf"]["confidence"], mb["tfidf"]["confidence"]]
            )
            probs_by_model["naive_bayes"].append(
                [1.0 - mb["naive_bayes"]["confidence"], mb["naive_bayes"]["confidence"]]
            )
            probs_by_model["ensemble"].append(
                [1.0 - out["ensemble_probability"], out["ensemble_probability"]]
            )
            fuzzy_scores.append(float(out["fuzzy_score"]))

        probs_by_model_np = {k: np.asarray(v, dtype="float32") for k, v in probs_by_model.items()}
        fuzzy_arr = np.asarray(fuzzy_scores, dtype="float32")

        metrics_report = eval_metrics.classification_report(y_true, probs_by_model_np)
        fuzzy_report = eval_metrics.fuzzy_metrics(y_true, fuzzy_arr)

        llm_report: Dict[str, Any] = {}
        if use_llm_judge and self.judge is not None:
            dataset = []
            for i, text in enumerate(texts):
                dataset.append(
                    {
                        "text": text,
                        "predictions": {
                            "bert": probs_by_model_np["bert"][i].tolist(),
                            "tfidf": probs_by_model_np["tfidf"][i].tolist(),
                            "naive_bayes": probs_by_model_np["naive_bayes"][i].tolist(),
                            "ensemble": probs_by_model_np["ensemble"][i].tolist(),
                        },
                        "fuzzy_score": float(fuzzy_arr[i]),
                        "cycle_metrics": {},
                    }
                )
            import asyncio

            judgments = asyncio.run(self.judge.evaluate_batch(dataset))
            llm_report = eval_metrics.llm_judge_metrics(judgments)
        else:
            judgments = []

        # Render dashboard
        dashboard.generate_dashboard(y_true, probs_by_model_np, fuzzy_arr, self.output_dir / "evaluation_dashboard.png")

        report = {
            "metrics": metrics_report,
            "fuzzy": fuzzy_report,
            "llm_judge": llm_report,
        }
        out_path = self.output_dir / "evaluation_report.json"
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

        LOGGER.info("Evaluation report written to %s", out_path)
        return report


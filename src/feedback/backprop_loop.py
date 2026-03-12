"""Backward propagation feedback loop orchestration.

This module belongs to the *feedback* component of the pipeline. It coordinates
detector predictions, fuzzy scoring, optional LLM judge calls, model updates,
and Git automation in a multi-step cycle.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from src.detector import MisinformationDetector
from src.evaluation.llm_judge import LLMJudge
from src.feedback.feedback_store import FeedbackStore
from src.feedback.online_trainer import OnlineTrainer
from src.utils.git_manager import GitManager
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)


@dataclass
class CycleMetrics:
    """Metrics captured for a single feedback cycle."""

    ensemble_f1: float
    bert_f1: float
    tfidf_f1: float
    nb_f1: float
    cycle_num: int


class BackpropFeedbackLoop:
    """Feedback loop controller implementing the 9-step cycle.

    Component:
        Feedback / Controller.

    Args:
        detector: Master misinformation detector instance.
        store: FeedbackStore for persisting samples.
        trainer: OnlineTrainer used for incremental updates.
        judge: LLMJudge instance.
        git_manager: GitManager for committing results.
    """

    def __init__(
        self,
        detector: MisinformationDetector,
        store: FeedbackStore,
        trainer: OnlineTrainer,
        judge: Optional[LLMJudge] = None,
        git_manager: Optional[GitManager] = None,
    ) -> None:
        self.detector = detector
        self.store = store
        self.trainer = trainer
        self.judge = judge
        self.git_manager = git_manager or GitManager()

        self.cycle_count: int = 0
        self.metrics_history: List[CycleMetrics] = []

    # ----------------------------------------------------------------- utilities
    @staticmethod
    def _f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        from sklearn.metrics import f1_score

        if y_true.size == 0:
            return 0.0
        return float(f1_score(y_true, y_pred, zero_division=0))

    # ------------------------------------------------------------------- run loop
    async def run_cycle(
        self,
        texts: Sequence[str],
        true_labels: Optional[Sequence[int]] = None,
    ) -> Dict[str, Any]:
        """Run a complete feedback cycle.

        Args:
            texts: Input texts for this cycle.
            true_labels: Optional ground-truth labels (0/1). If omitted, the
                LLM judge is used to infer labels.

        Returns:
            dict: Summary metrics for the cycle.
        """

        self.cycle_count += 1
        cycle_num = self.cycle_count
        LOGGER.info("Starting feedback cycle %d with %d samples.", cycle_num, len(texts))

        # STEP 1 — FORWARD PASS ------------------------------------------------
        predictions: List[Dict[str, Any]] = []
        for text in texts:
            det_out = self.detector.predict(text)
            ensemble = det_out["ensemble"]
            fuzzy = det_out["fuzzy"]

            fb_score = self.store.get_feedback_score(text)

            predictions.append(
                {
                    "text": text,
                    "detector": det_out,
                    "ensemble_prob": float(ensemble["probability_misinformation"]),
                    "ensemble_label": 1 if ensemble["label"] == "misinformation" else 0,
                    "bert_conf": float(det_out["models"]["bert"][1]),
                    "tfidf_conf": float(det_out["models"]["tfidf"][1]),
                    "nb_conf": float(det_out["models"]["naive_bayes"][1]),
                    "model_agreement": float(ensemble["agreement"]),
                    "fuzzy_score": float(fuzzy["score"]),
                    "feedback_score": float(fb_score),
                }
            )

        # STEP 2 — GET GROUND TRUTH -------------------------------------------
        if true_labels is not None:
            labels = np.asarray(list(true_labels), dtype=int)
            judge_conf = np.ones_like(labels, dtype="float32")
        else:
            if self.judge is None:
                raise RuntimeError("LLMJudge is required when true_labels are not provided.")
            dataset = [
                {
                    "text": p["text"],
                    "predictions": p["detector"],
                    "fuzzy_score": p["fuzzy_score"],
                    "cycle_metrics": {"cycle_num": cycle_num},
                }
                for p in predictions
            ]
            judgments = await self.judge.evaluate_batch(dataset)
            labels = np.asarray(
                [1 if j["independent_verdict"] == "misinformation" else 0 for j in judgments],
                dtype=int,
            )
            judge_conf = np.asarray(
                [float(j.get("judge_confidence", 1.0)) for j in judgments], dtype="float32"
            )

        # STEP 3 — COMPUTE ERROR SIGNALS --------------------------------------
        bert_errors = []
        tfidf_errors = []
        nb_errors = []
        total_errors = []
        high_error_flags = []
        high_error_samples: List[Dict[str, Any]] = []

        for i, p in enumerate(predictions):
            true = float(labels[i])
            jc = float(judge_conf[i])

            bert_error = abs(p["bert_conf"] - true) * jc
            tfidf_error = abs(p["tfidf_conf"] - true) * jc
            nb_error = abs(p["nb_conf"] - true) * jc
            total_error = 0.5 * bert_error + 0.3 * tfidf_error + 0.2 * nb_error

            bert_errors.append(bert_error)
            tfidf_errors.append(tfidf_error)
            nb_errors.append(nb_error)
            total_errors.append(total_error)

            high = total_error > 0.25
            high_error_flags.append(high)
            if high:
                high_error_samples.append(
                    {
                        "text": p["text"],
                        "true_label": int(true),
                        "inputs": p.get("bert_inputs"),  # optional, for future use
                    }
                )

        bert_errors_arr = np.asarray(bert_errors, dtype="float32")
        tfidf_errors_arr = np.asarray(tfidf_errors, dtype="float32")
        nb_errors_arr = np.asarray(nb_errors, dtype="float32")
        total_errors_arr = np.asarray(total_errors, dtype="float32")
        high_error_flags_arr = np.asarray(high_error_flags, dtype=bool)

        # STEP 4 — BACKWARD PASS (only on high_error samples) ------------------
        self.trainer.update_bert(high_error_samples)
        self.trainer.update_tfidf(high_error_samples)
        self.trainer.update_naive_bayes(high_error_samples)

        # STEP 5 — UPDATE FUZZY THRESHOLDS ------------------------------------
        is_wrong = (labels != np.asarray([p["ensemble_label"] for p in predictions]))
        fuzzy_scores = np.asarray([p["fuzzy_score"] for p in predictions], dtype="float32")
        wrong_idx = np.where(is_wrong & (np.abs(fuzzy_scores - labels.astype("float32")) > 0.3))[0]
        if wrong_idx.size > 0:
            mean_error = float(np.mean(np.abs(fuzzy_scores[wrong_idx] - labels[wrong_idx])))
            delta = 0.02 * np.sign(mean_error - 0.5)
            engine = self.detector.fuzzy
            new_thr = float(
                np.clip(engine.threshold_suspicious + delta, 0.35, 0.60)
            )
            engine.threshold_suspicious = new_thr
            LOGGER.info(
                "Adjusted fuzzy suspicious threshold to %.3f based on mean_error=%.3f",
                new_thr,
                mean_error,
            )

        # STEP 6 — RECALIBRATE ENSEMBLE WEIGHTS -------------------------------
        from sklearn.metrics import f1_score

        y_true = labels
        y_pred_ens = np.asarray([p["ensemble_label"] for p in predictions], dtype=int)
        # For base models, use 0.5 threshold on confidences.
        y_pred_bert = (np.asarray([p["bert_conf"] for p in predictions]) >= 0.5).astype(int)
        y_pred_tfidf = (np.asarray([p["tfidf_conf"] for p in predictions]) >= 0.5).astype(int)
        y_pred_nb = (np.asarray([p["nb_conf"] for p in predictions]) >= 0.5).astype(int)

        bert_f1 = float(f1_score(y_true, y_pred_bert, zero_division=0))
        tfidf_f1 = float(f1_score(y_true, y_pred_tfidf, zero_division=0))
        nb_f1 = float(f1_score(y_true, y_pred_nb, zero_division=0))
        ens_f1 = float(f1_score(y_true, y_pred_ens, zero_division=0))

        batch_metrics = {
            "bert": {"f1": bert_f1},
            "tfidf": {"f1": tfidf_f1},
            "naive_bayes": {"f1": nb_f1},
            "ensemble": {"f1": ens_f1},
        }
        if hasattr(self.detector.ensemble, "recalibrate_weights"):
            self.detector.ensemble.recalibrate_weights(batch_metrics)

        # STEP 7 — PERSIST ALL FEEDBACK ---------------------------------------
        for i, p in enumerate(predictions):
            self.store.save(
                text=p["text"],
                prediction=int(p["ensemble_label"]),
                true_label=int(labels[i]),
                fuzzy_score=float(p["fuzzy_score"]),
                error_dict={
                    "bert_error": float(bert_errors_arr[i]),
                    "tfidf_error": float(tfidf_errors_arr[i]),
                    "nb_error": float(nb_errors_arr[i]),
                    "total_error": float(total_errors_arr[i]),
                },
            )

        # STEP 8 — IMPROVEMENT TREND CHECK ------------------------------------
        metrics = CycleMetrics(
            ensemble_f1=ens_f1,
            bert_f1=bert_f1,
            tfidf_f1=tfidf_f1,
            nb_f1=nb_f1,
            cycle_num=cycle_num,
        )
        self.metrics_history.append(metrics)

        if len(self.metrics_history) >= 3:
            last_three = self.metrics_history[-3:]
            if all(m.ensemble_f1 < 0.75 for m in last_three):
                LOGGER.critical(
                    "Ensemble F1 below 0.75 for last 3 cycles. Marking retrain required."
                )
                flag_path = Path("RETRAIN_REQUIRED.flag")
                payload = {
                    "cycle": cycle_num,
                    "recent_ensemble_f1": [m.ensemble_f1 for m in last_three],
                }
                flag_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        # STEP 9 — GIT COMMIT --------------------------------------------------
        if self.git_manager is not None:
            self.git_manager.commit_cycle_results(
                cycle_num,
                {
                    "ensemble_f1": ens_f1,
                    "bert_f1": bert_f1,
                    "tfidf_f1": tfidf_f1,
                    "nb_f1": nb_f1,
                },
            )

        summary = {
            "cycle_num": cycle_num,
            "ensemble_f1": ens_f1,
            "bert_f1": bert_f1,
            "tfidf_f1": tfidf_f1,
            "nb_f1": nb_f1,
            "num_samples": len(texts),
            "num_high_error": int(high_error_flags_arr.sum()),
        }
        LOGGER.info("Completed feedback cycle %d: %s", cycle_num, summary)
        return summary


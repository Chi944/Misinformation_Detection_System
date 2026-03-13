import logging
import numpy as np
from pathlib import Path
from src.utils.logger import get_logger


class BackpropFeedbackLoop:
    """
    Orchestrates the full backward propagation feedback cycle.

    After each forward pass through all 3 models, error signals flow
    back to incrementally update BERT, TF-IDF, and Naive Bayes weights,
    adjust fuzzy thresholds, and recalibrate ensemble weights.

    Full cycle (9 steps):
        text_batch -> forward pass -> ground truth -> error signals
        -> backward pass -> fuzzy update -> weight recalibration
        -> persist -> trend check -> git commit

    Part of the MisinformationDetector pipeline.

    Args:
        detector: MisinformationDetector master instance
        config (dict): loaded config.yaml as dict
    """

    def __init__(self, detector, config):
        self.detector = detector
        self.config = config
        # Lazy imports to keep module import fast.
        from src.feedback.feedback_store import FeedbackStore
        from src.feedback.online_trainer import OnlineTrainer

        self.store = FeedbackStore()
        self.online = OnlineTrainer(
            bert_model=detector.bert_model,
            bert_tokenizer=detector.bert_tokenizer,
            tfidf_model=detector.tfidf_model,
            tfidf_vectorizer=detector.tfidf_vectorizer,
            nb_model=detector.nb_model,
            nb_vectorizer=detector.nb_vectorizer,
            device=detector.device,
        )
        self.cycle_count = 0
        self.metrics_history = []
        self.logger = get_logger(__name__)

        fb_cfg = config.get("feedback", {})
        self.high_error_threshold = float(fb_cfg.get("high_error_threshold", 0.25))
        self.min_f1 = float(fb_cfg.get("min_f1_threshold", 0.75))
        self.fail_limit = int(fb_cfg.get("consecutive_fail_limit", 3))

    def run_cycle(self, text_batch: list, true_labels: list = None) -> dict:
        """
        Execute one complete forward + feedback backward cycle.

        Args:
            text_batch (list): list of text strings to process
            true_labels (list, optional): int labels 0/1. If None, the
                LLM judge determines ground truth automatically.
        Returns:
            dict: per-model and ensemble metrics for this cycle
        """
        self.logger.info("=== Feedback Cycle #%d START ===", self.cycle_count + 1)

        # ── STEP 1: FORWARD PASS ────────────────────────────────────────
        predictions = []
        fuzzy_scores = []
        for text in text_batch:
            pred = self.detector.predict(text)
            predictions.append(pred)
            fb_score = self.store.get_feedback_score(text)
            fscore = self.detector.fuzzy_engine.compute(
                {
                    "source_credibility": 0.5,
                    "bert_confidence": pred["model_breakdown"]["bert"]["confidence"],
                    "tfidf_confidence": pred["model_breakdown"]["tfidf"]["confidence"],
                    "nb_confidence": pred["model_breakdown"]["naive_bayes"]["confidence"],
                    "model_agreement": pred["model_agreement"],
                    "feedback_score": fb_score,
                }
            )
            fuzzy_scores.append(fscore)

        # ── STEP 2: GROUND TRUTH ────────────────────────────────────────
        judge_confidences = [1.0] * len(text_batch)
        if true_labels is None:
            try:
                judgments = self.detector.llm_judge.evaluate_batch(
                    list(zip(text_batch, predictions, fuzzy_scores))
                )
                true_labels = [
                    1 if j.get("independent_verdict") == "misinformation" else 0 for j in judgments
                ]
                judge_confidences = [float(j.get("judge_confidence", 1.0)) for j in judgments]
            except Exception as e:
                self.logger.warning("LLM judge failed, defaulting labels to 0: %s", e)
                true_labels = [0] * len(text_batch)

        # ── STEP 3: COMPUTE ERROR SIGNALS ───────────────────────────────
        errors = self._compute_errors(predictions, true_labels, judge_confidences)
        high_error_idx = [
            i for i, e in enumerate(errors) if e["total_error"] > self.high_error_threshold
        ]
        self.logger.info("High-error samples: %d/%d", len(high_error_idx), len(text_batch))

        # ── STEP 4: BACKWARD PASS ───────────────────────────────────────
        if high_error_idx:
            err_batch = [(text_batch[i], true_labels[i]) for i in high_error_idx]
            self.online.update_bert(err_batch)
            self.online.update_tfidf(err_batch)
            self.online.update_naive_bayes(err_batch)

        # ── STEP 5: UPDATE FUZZY THRESHOLDS ─────────────────────────────
        self._update_fuzzy_thresholds(errors, true_labels, fuzzy_scores)

        # ── STEP 6: RECALIBRATE ENSEMBLE WEIGHTS ────────────────────────
        cycle_metrics = self._compute_cycle_metrics(predictions, true_labels)
        self.detector.ensemble.recalibrate_weights(cycle_metrics)

        # ── STEP 7: PERSIST FEEDBACK ─────────────────────────────────────
        for i, text in enumerate(text_batch):
            self.store.save(text, predictions[i], true_labels[i], fuzzy_scores[i], errors[i])

        # ── STEP 8: TREND CHECK ──────────────────────────────────────────
        self.metrics_history.append(cycle_metrics)
        self._check_improvement_trend()

        # ── STEP 9: GIT COMMIT ───────────────────────────────────────────
        try:
            self.detector.git_manager.commit_cycle_results(self.cycle_count + 1, cycle_metrics)
        except Exception as e:
            self.logger.warning("Git commit failed (non-fatal): %s", e)

        self.cycle_count += 1
        ens_f1 = cycle_metrics.get("ensemble", {}).get("f1", 0.0)
        self.logger.info(
            "=== Feedback Cycle #%d COMPLETE - Ensemble F1: %.4f ===", self.cycle_count, ens_f1
        )
        return cycle_metrics

    def _compute_errors(self, predictions, true_labels, judge_confidences) -> list:
        """
        Compute per-sample error signals for all 3 models.

        Args:
            predictions (list): list of predict() result dicts
            true_labels (list): list of int ground truth labels
            judge_confidences (list): per-sample LLM judge confidence
        Returns:
            list of dicts: bert_error, tfidf_error, nb_error,
                           total_error, is_wrong per sample
        """
        errors = []
        for pred, label, conf in zip(predictions, true_labels, judge_confidences):
            label = float(label)
            b = abs(pred["model_breakdown"]["bert"]["confidence"] - label) * conf
            t = abs(pred["model_breakdown"]["tfidf"]["confidence"] - label) * conf
            n = abs(pred["model_breakdown"]["naive_bayes"]["confidence"] - label) * conf
            total = b * 0.5 + t * 0.3 + n * 0.2
            pred_lbl = 1 if pred["crisp_label"] == "misinformation" else 0
            errors.append(
                {
                    "bert_error": b,
                    "tfidf_error": t,
                    "nb_error": n,
                    "total_error": total,
                    "is_wrong": pred_lbl != int(label),
                }
            )
        return errors

    def _update_fuzzy_thresholds(self, errors, true_labels, fuzzy_scores):
        """
        Gradient-free hill climbing update of fuzzy threshold_suspicious.

        When the fuzzy engine is confidently wrong, nudge the suspicious
        threshold by +/- 0.02 toward reducing that error.

        Args:
            errors (list): error dicts from _compute_errors
            true_labels (list): int ground truth labels
            fuzzy_scores (list): float fuzzy scores for each sample
        """
        wrong_fuzzy = [
            fuzzy_scores[i]
            for i in range(len(errors))
            if errors[i]["is_wrong"] and abs(fuzzy_scores[i] - true_labels[i]) > 0.3
        ]
        if not wrong_fuzzy:
            return
        mean_err = float(np.mean(wrong_fuzzy))
        delta = 0.02 * np.sign(mean_err - 0.5)
        old_thr = self.detector.fuzzy_engine.threshold_suspicious
        new_thr = float(np.clip(old_thr + delta, 0.35, 0.60))
        self.detector.fuzzy_engine.threshold_suspicious = new_thr
        self.logger.info(
            "Fuzzy threshold_suspicious: %.3f -> %.3f (delta=%+.3f)", old_thr, new_thr, delta
        )

    def _compute_cycle_metrics(self, predictions, true_labels) -> dict:
        """
        Compute per-model and ensemble F1 + accuracy for this batch.

        Args:
            predictions (list): list of predict() result dicts
            true_labels (list): int ground truth labels
        Returns:
            dict: per-model metrics plus ensemble and current weights
        """
        from sklearn.metrics import accuracy_score, f1_score

        y_true = [int(l) for l in true_labels]
        metrics = {}
        for name in ["bert", "tfidf", "naive_bayes"]:
            y_pred = [p["model_breakdown"][name]["label"] for p in predictions]
            metrics[name] = {
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
                "accuracy": float(accuracy_score(y_true, y_pred)),
            }
        ens_pred = [1 if p["crisp_label"] == "misinformation" else 0 for p in predictions]
        metrics["ensemble"] = {
            "f1": float(f1_score(y_true, ens_pred, zero_division=0)),
            "accuracy": float(accuracy_score(y_true, ens_pred)),
        }
        metrics["weights"] = dict(self.detector.ensemble.weights)
        return metrics

    def _check_improvement_trend(self):
        """
        Alert and write flag file if F1 has been below threshold for
        consecutive_fail_limit cycles in a row.

        Writes RETRAIN_REQUIRED.flag to project root when triggered.
        """
        if len(self.metrics_history) < self.fail_limit:
            return
        recent_f1 = [
            m.get("ensemble", {}).get("f1", 1.0) for m in self.metrics_history[-self.fail_limit :]
        ]
        if not all(f < self.min_f1 for f in recent_f1):
            return
        self.logger.critical(
            "ACCURACY ALERT: Ensemble F1 below %.2f for %d consecutive " "cycles. Recent F1s: %s",
            self.min_f1,
            self.fail_limit,
            [("%.4f" % f) for f in recent_f1],
        )
        flag = Path("RETRAIN_REQUIRED.flag")
        flag.write_text(
            "Full retrain required at cycle %d\n"
            "Recent F1 scores: %s\n" % (self.cycle_count, [("%.4f" % f) for f in recent_f1])
        )
        self.logger.critical("Wrote %s", flag)

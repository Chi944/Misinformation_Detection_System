"""Master misinformation detector interface.

This class wires together the three base models, the ensemble, fuzzy engine,
feedback components, and evaluation pipeline to provide a unified API.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import yaml

from src.evaluation.llm_judge import LLMJudge
from src.evaluation.pipeline import EvaluationPipeline
from src.feedback.backprop_loop import BackpropFeedbackLoop
from src.feedback.feedback_store import FeedbackStore
from src.feedback.online_trainer import OnlineTrainer
from src.fuzzy.fuzzy_engine import FuzzyMisinformationEngine
from src.models.bert_classifier import BERTMisinformationClassifier
from src.models.ensemble_detector import EnsembleDetector, EnsembleWeights
from src.models.naive_bayes_model import TFNaiveBayesWrapper
from src.models.tfidf_model import TfidfDNNClassifier
from src.training.dataset import MisinformationDataset
from src.utils.git_manager import GitManager
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)


class MisinformationDetector:
    """High-level detector orchestrating all core components.

    Component:
        Detector / Master.

    Args:
        config: Path to ``config.yaml``.
        fast_mode: If True, use lighter settings for CI/smoke tests.
    """

    def __init__(self, config: str = "config.yaml", fast_mode: bool = False) -> None:
        self.config_path = Path(config)
        self.fast_mode = fast_mode

        with self.config_path.open("r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        # Suppress TF logs
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

        model_cfg = self.cfg.get("models", {})

        # Base models -----------------------------------------------------------
        nb_cfg = model_cfg.get("naive_bayes", {})
        tfidf_cfg = model_cfg.get("tfidf", {})
        bert_cfg = model_cfg.get("bert", {})

        self.nb = TFNaiveBayesWrapper(config=nb_cfg, models_dir="models")
        self.tfidf = TfidfDNNClassifier(config=tfidf_cfg, models_dir="models")

        self.bert = None
        if not self.fast_mode:
            self.bert = BERTMisinformationClassifier(
                checkpoint=bert_cfg.get("checkpoint", "bert-base-uncased"),
                dropout=bert_cfg.get("dropout", 0.3),
            )

        weights = EnsembleWeights(
            bert=float(bert_cfg.get("weight", 0.5)),
            tfidf=float(tfidf_cfg.get("weight", 0.3)),
            naive_bayes=float(nb_cfg.get("weight", 0.2)),
        )
        self.ensemble = EnsembleDetector(
            bert_model=self.bert if self.bert is not None else BERTMisinformationClassifier(),  # type: ignore[arg-type]
            tfidf_model=self.tfidf,
            nb_model=self.nb,
            weights=weights,
        )

        # Fuzzy engine ---------------------------------------------------------
        self.fuzzy = FuzzyMisinformationEngine()
        fuzzy_cfg = self.cfg.get("fuzzy", {})
        self.fuzzy.threshold_suspicious = float(
            fuzzy_cfg.get("threshold_suspicious", self.fuzzy.threshold_suspicious)
        )
        self.fuzzy.threshold_misinformation = float(
            fuzzy_cfg.get("threshold_misinformation", self.fuzzy.threshold_misinformation)
        )

        # Feedback components --------------------------------------------------
        self.feedback_store = FeedbackStore(path="feedback.db")
        self.online_trainer = OnlineTrainer(
            bert_model=self.bert,
            tfidf_model=self.tfidf,
            nb_model=self.nb,
        )
        self.git_manager = GitManager()

        # LLM judge and evaluation pipeline -----------------------------------
        llm_cfg = self.cfg.get("llm_judge", {})
        self.llm_judge = LLMJudge(
            host=llm_cfg.get("host", "http://localhost:11434"),
            model=llm_cfg.get("model", "llama3"),
            max_retries=int(llm_cfg.get("max_retries", 3)),
            backoff_base=float(llm_cfg.get("backoff_base", 2)),
        )
        self.eval_pipeline = EvaluationPipeline(
            detector=self,
            judge=self.llm_judge,
            output_dir=self.cfg.get("evaluation", {}).get("output_dir", "reports"),
        )

        self.feedback_loop = BackpropFeedbackLoop(
            detector=self,
            store=self.feedback_store,
            trainer=self.online_trainer,
            judge=self.llm_judge,
            git_manager=self.git_manager,
        )

    # ----------------------------------------------------------------- predict
    def predict(self, text: str) -> Dict[str, Any]:
        """Run a prediction on a single text.

        Returns:
            Dict with crisp label, ensemble probability, fuzzy score, per-model
            breakdown, ensemble weights, and model agreement.
        """

        ens = self.ensemble.predict(text)

        bert_conf = float(ens["bert_proba"][1])
        tfidf_conf = float(ens["tfidf_proba"][1])
        nb_conf = float(ens["naive_bayes_proba"][1])

        fuzzy_inputs = {
            "source_credibility": 0.5,
            "bert_confidence": bert_conf,
            "tfidf_confidence": tfidf_conf,
            "nb_confidence": nb_conf,
            "model_agreement": float(ens["agreement"]),
            "feedback_score": 0.5,
        }
        misinfo_score = self.fuzzy.compute(fuzzy_inputs)

        weights = {
            "bert": float(self.ensemble.weights.bert),
            "tfidf": float(self.ensemble.weights.tfidf),
            "naive_bayes": float(self.ensemble.weights.naive_bayes),
        }

        model_breakdown = {
            "bert": {"label": int(bert_conf >= 0.5), "confidence": bert_conf},
            "tfidf": {"label": int(tfidf_conf >= 0.5), "confidence": tfidf_conf},
            "naive_bayes": {"label": int(nb_conf >= 0.5), "confidence": nb_conf},
        }

        return {
            "crisp_label": ens["crisp_label"],
            "ensemble_probability": float(ens["ensemble_probability"]),
            "fuzzy_score": float(misinfo_score),
            "model_breakdown": model_breakdown,
            "ensemble_weights": weights,
            "model_agreement": float(ens["agreement"]),
            # Backwards-compatible fields used elsewhere:
            "models": {
                "bert": ens["bert_proba"],
                "tfidf": ens["tfidf_proba"],
                "naive_bayes": ens["naive_bayes_proba"],
            },
            "ensemble": {
                "probability_misinformation": ens["ensemble_probability"],
                "label": ens["crisp_label"],
                "agreement": ens["agreement"],
            },
            "fuzzy": {
                "score": misinfo_score,
                "inputs": fuzzy_inputs,
            },
        }

    # ---------------------------------------------------------------- evaluate
    def evaluate(
        self,
        texts: Sequence[str],
        labels: Sequence[int],
        use_llm_judge: bool = True,
    ) -> Dict[str, Any]:
        """Run the full evaluation pipeline.

        Args:
            texts: Input texts.
            labels: Ground-truth labels.
            use_llm_judge: Whether to include the LLM judge.

        Returns:
            Evaluation report dictionary.
        """

        return self.eval_pipeline.evaluate(texts, labels, use_llm_judge=use_llm_judge)

    def evaluate_quick(
        self,
        texts: Sequence[str],
        labels: Sequence[int],
    ) -> Dict[str, Any]:
        """Fast evaluation without the LLM judge (for CI/smoke tests)."""

        return self.eval_pipeline.evaluate(texts, labels, use_llm_judge=False)


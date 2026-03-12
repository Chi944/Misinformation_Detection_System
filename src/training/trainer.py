# src/training/trainer.py
"""Training orchestration for misinformation models.

This module belongs to the *training* component of the pipeline. It trains
the Naive Bayes, TF-IDF DNN, and BERT models, computes validation metrics and
thresholds, and enforces accuracy gates.

It assumes:
- MisinformationDataset in src.training.dataset
- TFNaiveBayesWrapper in src.models.naive_bayes_model
- TfidfDNNClassifier in src.models.tfidf_model
- BERTMisinformationClassifier + BERTTrainer in src.models.bert_classifier
- EnsembleDetector in src.models.ensemble_detector
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import yaml
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_curve

from src.models.bert_classifier import BERTMisinformationClassifier, BERTTrainer
from src.models.ensemble_detector import EnsembleDetector, EnsembleWeights
from src.models.naive_bayes_model import TFNaiveBayesWrapper
from src.models.tfidf_model import TfidfDNNClassifier
from src.training.dataset import MisinformationDataset
from src.utils.logger import get_logger

try:
    import torch
    from torch.utils.data import DataLoader
    from transformers import BertTokenizerFast
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    DataLoader = object  # type: ignore
    BertTokenizerFast = object  # type: ignore

LOGGER = get_logger(__name__)


class AccuracyGateError(Exception):
    """Raised when a model fails to meet accuracy gate thresholds."""


@dataclass
class GateThresholds:
    """Accuracy gate thresholds for a model."""

    accuracy: float
    precision: float
    f1: float


GATES = {
    "bert": GateThresholds(accuracy=0.78, precision=0.76, f1=0.77),
    "tfidf": GateThresholds(accuracy=0.76, precision=0.75, f1=0.75),
    "naive_bayes": GateThresholds(accuracy=0.75, precision=0.75, f1=0.74),
    "ensemble": GateThresholds(accuracy=0.82, precision=0.80, f1=0.81),
}


def _compute_metrics(y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """Compute accuracy, precision, recall, f1 for binary probs."""
    y_hat = (y_proba[:, 1] >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_hat)),
        "precision": float(precision_score(y_true, y_hat, zero_division=0)),
        "recall": float(recall_score(y_true, y_hat, zero_division=0)),
        "f1": float(f1_score(y_true, y_hat, zero_division=0)),
    }


def _youden_j_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Compute decision threshold using Youden's J statistic."""
    fpr, tpr, thresholds = roc_curve(y_true, y_proba[:, 1])
    j = tpr - fpr
    idx = int(np.argmax(j))
    return float(thresholds[idx])


class Trainer:
    """Unified trainer for NB, TF-IDF DNN, and BERT.

    This class owns:
      - a MisinformationDataset
      - instances of TFNaiveBayesWrapper, TfidfDNNClassifier, BERT trainer
      - an EnsembleDetector built from the three models

    It exposes `train_all()` which trains all models, computes metrics and
    thresholds, enforces accuracy gates, and returns a summary dict.
    """

    def __init__(
        self,
        data_path: str | Path = "data/sample_train.csv",
        config_path: str | Path = "config.yaml",
        models_dir: str | Path = "models",
    ) -> None:
        self.data_path = Path(data_path)
        self.config_path = Path(config_path)
        self.models_dir = Path(models_dir)

        with self.config_path.open("r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.dataset = MisinformationDataset(self.data_path)

        self.nb = TFNaiveBayesWrapper(
            config=self.config.get("models", {}).get("naive_bayes", {}),
            models_dir=self.models_dir,
        )
        self.tfidf = TfidfDNNClassifier(
            config=self.config.get("models", {}).get("tfidf", {}),
            models_dir=self.models_dir,
        )

        self.bert_model = None
        self.bert_trainer = None
        if torch is not None:
            bert_cfg = self.config.get("models", {}).get("bert", {})
            self.bert_model = BERTMisinformationClassifier(
                checkpoint=bert_cfg.get("checkpoint", "bert-base-uncased"),
                dropout=bert_cfg.get("dropout", 0.3),
            )
            tokenizer = BertTokenizerFast.from_pretrained(
                bert_cfg.get("checkpoint", "bert-base-uncased")
            )
            self.bert_trainer = BERTTrainer(
                model=self.bert_model,
                tokenizer=tokenizer,
                models_dir=self.models_dir,
                freeze_layers=bert_cfg.get("freeze_layers", 6),
                learning_rate=bert_cfg.get("learning_rate", 2e-5),
                epochs=bert_cfg.get("epochs", 5),
            )

        self.ensemble: EnsembleDetector | None = None

    # ----------------------------------------------------------------- NB
    def _train_naive_bayes(self) -> Dict[str, Any]:
        LOGGER.info("Training Naive Bayes...")
        X_train, y_train = self.dataset.to_sklearn("train")
        X_val, y_val = self.dataset.to_sklearn("val")

        # For NB we let the wrapper handle vectorization; SMOTE will be applied
        # in feature space downstream, so we keep this simple here.
        self.nb.fit(X_train, y_train)
        val_proba = self.nb.predict_proba_np(X_val)
        thr = _youden_j_threshold(y_val, val_proba)
        metrics = _compute_metrics(y_val, val_proba, threshold=thr)
        self._check_gates("naive_bayes", metrics)
        return {"metrics": metrics, "threshold": thr}

    # --------------------------------------------------------------- TF-IDF
    def _train_tfidf(self) -> Dict[str, Any]:
        LOGGER.info("Training TF-IDF DNN...")
        X_train, y_train = self.dataset.to_sklearn("train")
        X_val, y_val = self.dataset.to_sklearn("val")

        # For simplicity we do not apply SMOTE here, but it can be added by
        # vectorizing first then calling dataset.apply_smote().
        hist = self.tfidf.fit(X_train, y_train)
        val_proba = self.tfidf.predict_proba(X_val)
        thr = _youden_j_threshold(y_val, val_proba)
        metrics = _compute_metrics(y_val, val_proba, threshold=thr)
        self._check_gates("tfidf", metrics)
        return {"metrics": metrics, "threshold": thr, "history": hist}

    # ---------------------------------------------------------------- BERT
    def _train_bert(self) -> Dict[str, Any]:
        if self.bert_trainer is None or self.bert_model is None:
            raise RuntimeError("PyTorch / BERT not available in this environment.")

        LOGGER.info("Training BERT...")
        bert_cfg = self.config.get("models", {}).get("bert", {})
        max_len = bert_cfg.get("max_length", 512)
        batch_size = bert_cfg.get("batch_size", 16)

        train_ds = self.dataset.to_pytorch_dataset("train", tokenizer=self.bert_trainer.tokenizer, max_length=max_len)
        val_ds = self.dataset.to_pytorch_dataset("val", tokenizer=self.bert_trainer.tokenizer, max_length=max_len)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        best_metrics = self.bert_trainer.train(train_loader, val_loader)

        # For now, approximate probabilities via model output on val set.
        self.bert_model.eval()
        all_probs = []
        all_labels = []
        device = self.bert_trainer.device
        with torch.no_grad():
            for batch in val_loader:
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                tti = batch.get("token_type_ids")
                if tti is not None:
                    tti = tti.to(device)
                labels = batch["label"].to(device)
                probs = self.bert_model(ids, attention_mask=mask, token_type_ids=tti)
                all_probs.append(probs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        val_proba = np.concatenate(all_probs, axis=0)
        y_val = np.concatenate(all_labels, axis=0)
        thr = _youden_j_threshold(y_val, val_proba)
        metrics = _compute_metrics(y_val, val_proba, threshold=thr)
        self._check_gates("bert", metrics)
        return {"metrics": metrics, "threshold": thr, "best_metrics": best_metrics}

    # -------------------------------------------------------------- Ensemble
    def _build_ensemble(self) -> Dict[str, Any]:
        LOGGER.info("Building ensemble...")
        if self.bert_model is None:
            raise RuntimeError("BERT model not trained/available for ensemble.")

        self.ensemble = EnsembleDetector(
            bert_model=self.bert_model,
            tfidf_model=self.tfidf,
            nb_model=self.nb,
            weights=EnsembleWeights(),
        )

        # Evaluate ensemble on validation set
        X_val, y_val = self.dataset.to_sklearn("val")
        proba_list = []
        for text in X_val:
            out = self.ensemble.predict(str(text))
            # Convert crisp_label to probability proxy; we already store ensemble_probability.
            # For metrics, we need probabilities for class 1:
            p1 = out["ensemble_probability"] if out["crisp_label"] == "misinformation" else 1.0 - out["ensemble_probability"]
            proba_list.append([1.0 - p1, p1])
        val_proba = np.asarray(proba_list, dtype="float32")
        thr = _youden_j_threshold(y_val, val_proba)
        metrics = _compute_metrics(y_val, val_proba, threshold=thr)
        self._check_gates("ensemble", metrics)
        return {"metrics": metrics, "threshold": thr}

    # --------------------------------------------------------------- gates
    def _check_gates(self, model_name: str, metrics: Dict[str, float]) -> None:
        """Enforce accuracy gates for a given model."""
        gate = GATES[model_name]
        if (
            metrics["accuracy"] < gate.accuracy
            or metrics["precision"] < gate.precision
            or metrics["f1"] < gate.f1
        ):
            raise AccuracyGateError(
                f"{model_name} failed accuracy gate: "
                f"acc={metrics['accuracy']:.3f} (>= {gate.accuracy}), "
                f"prec={metrics['precision']:.3f} (>= {gate.precision}), "
                f"f1={metrics['f1']:.3f} (>= {gate.f1})"
            )

    # --------------------------------------------------------------- public
    def train_all(self) -> Dict[str, Any]:
        """Train NB, TF-IDF, BERT, and ensemble; enforce accuracy gates.

        Returns:
            dict: Nested metrics and thresholds for all models.
        """
        summary: Dict[str, Any] = {}

        nb_result = self._train_naive_bayes()
        summary["naive_bayes"] = nb_result

        tfidf_result = self._train_tfidf()
        summary["tfidf"] = tfidf_result

        bert_result = self._train_bert()
        summary["bert"] = bert_result

        ensemble_result = self._build_ensemble()
        summary["ensemble"] = ensemble_result

        return summary
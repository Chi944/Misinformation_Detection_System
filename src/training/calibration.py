# src/training/calibration.py
"""Calibration utilities for model outputs.

This module belongs to the *training* component of the pipeline. It provides
helpers for Platt scaling, temperature scaling, and isotonic calibration for
the ensemble.

The interfaces are minimal so that the Trainer can plug them in after
training base models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore


@dataclass
class CalibrationResult:
    """Container for calibration artefacts."""

    info: Dict[str, Any]


class CalibrationWrapper:
    """Calibration helper for Naive Bayes, BERT, and ensemble.

    Component:
        Training / Calibration.
    """

    # -------------------------- Naive Bayes: Platt scaling ------------------
    @staticmethod
    def platt_scaling(
        base_clf,
        X_val,
        y_val,
    ) -> Tuple[Any, CalibrationResult]:
        """Apply Platt scaling (sigmoid) to a base classifier.

        Args:
            base_clf: Fitted scikit-learn classifier with predict_proba.
            X_val: Validation features.
            y_val: Validation labels.

        Returns:
            (calibrated_clf, CalibrationResult)
        """
        calib = CalibratedClassifierCV(base_clf, method="sigmoid", cv="prefit")
        calib.fit(X_val, y_val)
        return calib, CalibrationResult(info={"method": "platt_sigmoid"})

    # ----------------------------- BERT: temperature scaling -----------------
    @staticmethod
    def _nll_with_temperature(logits: torch.Tensor, labels: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        """Negative log-likelihood under temperature-scaled logits."""
        scaled_logits = logits / T
        log_probs = torch.log_softmax(scaled_logits, dim=1)
        loss = torch.nn.functional.nll_loss(log_probs, labels)
        return loss

    @staticmethod
    def temperature_scaling(
        model: torch.nn.Module,
        val_loader,
        device: torch.device | None = None,
    ) -> Tuple[float, CalibrationResult]:
        """Learn a single temperature parameter for BERT on validation data.

        Args:
            model: Trained BERT classifier.
            val_loader: DataLoader with validation batches.
            device: Torch device.

        Returns:
            (temperature, CalibrationResult)
        """
        if torch is None:
            raise ImportError("PyTorch not available for temperature scaling.")

        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()

        all_logits = []
        all_labels = []
        with torch.no_grad():
            for batch in val_loader:
                ids = batch["input_ids"].to(device)  # type: ignore[arg-type]
                mask = batch["attention_mask"].to(device)  # type: ignore[arg-type]
                tti = batch.get("token_type_ids")
                if tti is not None:
                    tti = tti.to(device)  # type: ignore[assignment]
                labels = batch["label"].to(device)  # type: ignore[arg-type]
                # Get logits before softmax: we can invert BERTMisinformationClassifier
                # by using the classifier head directly if needed; here we re-logit
                # from probabilities as a proxy.
                probs = model(ids, attention_mask=mask, token_type_ids=tti)
                logits = torch.log(probs + 1e-12)
                all_logits.append(logits)
                all_labels.append(labels)

        logits_cat = torch.cat(all_logits, dim=0)
        labels_cat = torch.cat(all_labels, dim=0)

        T = torch.nn.Parameter(torch.ones(1, device=device))

        optimizer = torch.optim.LBFGS([T], lr=0.01, max_iter=50)

        def _closure():
            optimizer.zero_grad()
            loss = CalibrationWrapper._nll_with_temperature(logits_cat, labels_cat, T)
            loss.backward()
            return loss

        optimizer.step(_closure)
        temperature = float(T.detach().cpu().item())
        return temperature, CalibrationResult(info={"method": "temperature", "T": temperature})

    # --------------------------- Ensemble: isotonic regression ---------------
    @staticmethod
    def isotonic_regression(
        ensemble_probs: np.ndarray,
        y_val: np.ndarray,
    ) -> Tuple[IsotonicRegression, CalibrationResult]:
        """Fit isotonic regression on ensemble probabilities.

        Args:
            ensemble_probs: Array of shape (n_samples, 2) or (n_samples,) for
                the positive class.
            y_val: Binary labels.

        Returns:
            (isotonic_model, CalibrationResult)
        """
        if ensemble_probs.ndim == 2:
            p1 = ensemble_probs[:, 1]
        else:
            p1 = ensemble_probs

        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(p1, y_val.astype(float))
        return iso, CalibrationResult(info={"method": "isotonic"})
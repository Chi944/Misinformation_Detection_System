"""Evaluation metrics for misinformation detection.

This module belongs to the *evaluation* component of the pipeline. It computes
classification, calibration, fuzzy, and LLM-judge metrics for the three base
models and the ensemble.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from sklearn import metrics as skm


def _classification_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
) -> Dict[str, float]:
    """Compute core classification metrics for binary predictions.

    Args:
        y_true: Array of shape (n,) with 0/1 labels.
        y_proba: Array of shape (n, 2) with class probabilities.

    Returns:
        Dict with accuracy, precision, recall, F1, ROC-AUC, PR-AUC.
    """

    if y_proba.ndim == 2:
        p1 = y_proba[:, 1]
    else:
        p1 = y_proba
    y_pred = (p1 >= 0.5).astype(int)

    return {
        "accuracy": float(skm.accuracy_score(y_true, y_pred)),
        "precision": float(skm.precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(skm.recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(skm.f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(skm.roc_auc_score(y_true, p1)) if len(np.unique(y_true)) > 1 else 0.0,
        "pr_auc": float(skm.average_precision_score(y_true, p1)),
    }


def _expected_calibration_error(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Compute Expected Calibration Error (ECE) for binary probabilities."""

    if y_proba.ndim == 2:
        p1 = y_proba[:, 1]
    else:
        p1 = y_proba
    y_true = y_true.astype(int)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(p1, bins) - 1
    ece = 0.0
    for b in range(n_bins):
        mask = bin_ids == b
        if not np.any(mask):
            continue
        conf = p1[mask].mean()
        acc = y_true[mask].mean()
        ece += np.abs(conf - acc) * (mask.sum() / len(y_true))
    return float(ece)


def classification_report(
    y_true: np.ndarray,
    probs_by_model: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    """Compute full metric set for each model.

    Args:
        y_true: Ground-truth labels (0/1).
        probs_by_model: Mapping of model name -> probability array.

    Returns:
        Dict mapping model name -> metrics dict.
    """

    report: Dict[str, Any] = {}
    for name, proba in probs_by_model.items():
        base = _classification_metrics(y_true, proba)
        base["ece"] = _expected_calibration_error(y_true, proba)
        report[name] = base
    return report


def fuzzy_metrics(
    y_true: np.ndarray,
    fuzzy_scores: np.ndarray,
) -> Dict[str, float]:
    """Compute fuzzy-specific metrics.

    Args:
        y_true: Binary labels.
        fuzzy_scores: Fuzzy misinfo scores in [0, 1].

    Returns:
        Dict with mean score and accuracy-within-tolerance.
    """

    if y_true.size == 0:
        return {"mean_score": 0.0, "fuzzy_accuracy": 0.0}

    # Interpret labels as 0.0 or 1.0 and measure closeness within ±0.15.
    diffs = np.abs(fuzzy_scores - y_true.astype("float32"))
    fuzzy_acc = float((diffs <= 0.15).mean())
    return {
        "mean_score": float(fuzzy_scores.mean()),
        "fuzzy_accuracy": fuzzy_acc,
    }


def llm_judge_metrics(all_judgments: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate LLM-judge statistics over all samples.

    Args:
        all_judgments: Iterable of judgment dicts from LLMJudge.

    Returns:
        Dict with agreement stats, confidence distribution, and flag counts.
    """

    judgments = list(all_judgments)
    if not judgments:
        return {}

    confidences = np.asarray([float(j.get("judge_confidence", 0.0)) for j in judgments])
    flags_flat: List[str] = []
    for j in judgments:
        flags_flat.extend(list(j.get("flags", [])))

    return {
        "count": len(judgments),
        "mean_confidence": float(confidences.mean()),
        "std_confidence": float(confidences.std()),
        "flags_total": len(flags_flat),
    }


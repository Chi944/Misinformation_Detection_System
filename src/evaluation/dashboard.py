"""Dashboard generation for evaluation results.

This module belongs to the *evaluation* component of the pipeline. It renders a
multi-panel PNG summarising classification, calibration, and agreement metrics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics as skm


def generate_dashboard(
    y_true: np.ndarray,
    probs_by_model: Dict[str, np.ndarray],
    fuzzy_scores: np.ndarray,
    output_path: str | Path = "reports/evaluation_dashboard.png",
) -> None:
    """Generate a dashboard PNG with confusion matrices and ROC curves.

    Args:
        y_true: Ground-truth labels.
        probs_by_model: Mapping model name -> probability array.
        fuzzy_scores: Fuzzy misinfo scores.
        output_path: Destination PNG path.
    """

    models = list(probs_by_model.keys())
    n_models = len(models)
    if n_models == 0:
        return

    fig, axes = plt.subplots(2, max(2, n_models), figsize=(4 * max(2, n_models), 8))

    # Top row: confusion matrices
    for idx, name in enumerate(models):
        ax = axes[0, idx] if n_models > 1 else axes[0]
        proba = probs_by_model[name]
        p1 = proba[:, 1] if proba.ndim == 2 else proba
        y_pred = (p1 >= 0.5).astype(int)
        cm = skm.confusion_matrix(y_true, y_pred, labels=[0, 1])
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.set_title(f"{name} confusion")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, str(v), ha="center", va="center", color="black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Bottom row: ROC curves and fuzzy score histogram
    roc_ax = axes[1, 0]
    for name in models:
        proba = probs_by_model[name]
        p1 = proba[:, 1] if proba.ndim == 2 else proba
        if len(np.unique(y_true)) < 2:
            continue
        fpr, tpr, _ = skm.roc_curve(y_true, p1)
        roc_ax.plot(fpr, tpr, label=name)
    roc_ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    roc_ax.set_title("ROC curves")
    roc_ax.set_xlabel("FPR")
    roc_ax.set_ylabel("TPR")
    roc_ax.legend()

    fuzz_ax = axes[1, 1] if max(2, n_models) > 1 else axes[1]
    fuzz_ax.hist(fuzzy_scores, bins=20, range=(0, 1), alpha=0.7)
    fuzz_ax.set_title("Fuzzy score distribution")
    fuzz_ax.set_xlabel("misinfo_score")
    fuzz_ax.set_ylabel("count")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


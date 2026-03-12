"""Tests for the backward propagation feedback loop.

These tests ensure that a minimal feedback cycle can run and that the
RETRAIN_REQUIRED flag is created when ensemble F1 stays low.
"""

from __future__ import annotations

from pathlib import Path

import asyncio
import numpy as np

from src.detector import MisinformationDetector
from src.feedback.backprop_loop import BackpropFeedbackLoop
from src.feedback.feedback_store import FeedbackStore
from src.feedback.online_trainer import OnlineTrainer


def _make_loop(tmp_path: Path) -> BackpropFeedbackLoop:
    detector = MisinformationDetector(config="config.yaml", fast_mode=True)
    store = FeedbackStore(path=str(tmp_path / "feedback.db"))
    trainer = OnlineTrainer(
        bert_model=None,
        tfidf_model=detector.tfidf,
        nb_model=detector.nb,
    )
    loop = BackpropFeedbackLoop(
        detector=detector,
        store=store,
        trainer=trainer,
        judge=None,
        git_manager=None,
    )
    return loop


def test_feedback_cycle_runs_and_returns_summary(tmp_path):
    """run_cycle should execute without error and return summary dict."""

    loop = _make_loop(tmp_path)

    texts = ["example one", "example two"]
    labels = [0, 1]

    summary = asyncio.run(loop.run_cycle(texts, labels))

    assert "cycle_num" in summary
    assert summary["num_samples"] == len(texts)


def test_retrain_flag_created_after_low_f1(tmp_path, monkeypatch):
    """RETRAIN_REQUIRED.flag should be written after 3 low-F1 cycles."""

    loop = _make_loop(tmp_path)

    # Force ensemble F1 to be low by monkeypatching internal helper, if needed.
    monkeypatch.setattr(loop, "_f1_score", lambda y_true, y_pred: 0.5)

    texts = ["low f1 sample"] * 3
    labels = [1, 1, 1]

    for _ in range(3):
        asyncio.run(loop.run_cycle(texts, labels))

    flag_path = Path("RETRAIN_REQUIRED.flag")
    assert flag_path.exists()


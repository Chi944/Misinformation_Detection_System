"""Accuracy gate tests for all models.

These tests assume that `scripts/train_all.py` has been run and that
`reports/accuracy_gate_results.json` is available. They assert that each model
meets relaxed CI thresholds (>= 0.75 accuracy).
"""

from __future__ import annotations

import json
from pathlib import Path


def test_accuracy_gates_from_report():
    """All models should meet the minimum accuracy thresholds in CI."""

    report_path = Path("reports/accuracy_gate_results.json")
    assert report_path.exists(), "Run scripts/train_all.py before this test."

    data = json.loads(report_path.read_text(encoding="utf-8"))

    for name in ("bert", "tfidf", "naive_bayes", "ensemble"):
        section = data.get(name) or {}
        metrics = section.get("metrics") or {}
        acc = float(metrics.get("accuracy", 0.0))
        assert acc >= 0.75, f"{name} accuracy below 0.75: {acc}"


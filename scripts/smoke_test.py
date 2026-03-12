"""Smoke test for the misinformation detector.

This script exercises the core components on a small set of examples and is
designed to complete in under 60 seconds on CPU-only environments.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import requests

from src.detector import MisinformationDetector
from src.feedback.backprop_loop import BackpropFeedbackLoop


SAMPLE_TEXTS = [
    "Scientists at MIT publish peer-reviewed study on climate change in Nature journal.",
    "BREAKING: Government secretly adding microchips to drinking water supply.",
    "New study suggests moderate exercise linked to improved mental health outcomes.",
    "Celebrity claims vaccines caused her child's superpowers according to insider.",
    "Federal Reserve raises interest rates by 25 basis points as expected.",
]

SAMPLE_LABELS = [0, 1, 0, 1, 0]


def check_ollama(host: str = "http://localhost:11434") -> None:
    """Check that Ollama is reachable.

    This is a soft check: on failure we print a warning but do not exit with
    an error to keep the smoke test lightweight.
    """

    url = f"{host.rstrip('/')}/api/tags"
    try:
        resp = requests.get(url, timeout=2)
        resp.raise_for_status()
        print(f"Ollama reachable at {url}")
    except Exception:
        print(
            f"WARNING: Ollama not reachable at {url}. "
            "Run `ollama serve` and `ollama pull llama3` if you want LLM judging."
        )


def main() -> int:
    """Run the full smoke test sequence."""

    print("Running smoke test...")
    check_ollama()

    detector = MisinformationDetector(config="config.yaml", fast_mode=True)

    # 1. Per-model predictions
    for text in SAMPLE_TEXTS:
        out = detector.predict(text)
        mb = out["model_breakdown"]
        for name in ("bert", "tfidf", "naive_bayes"):
            conf = mb[name]["confidence"]
            assert 0.0 <= conf <= 1.0, f"{name} confidence out of range"

    print("Per-model prediction checks passed.")

    # 2. Ensemble checks
    for text in SAMPLE_TEXTS:
        out = detector.predict(text)
        prob = out["ensemble_probability"]
        assert out["crisp_label"] in {"credible", "misinformation"}
        assert 0.0 <= prob <= 1.0
    print("Ensemble checks passed.")

    # 3. Fuzzy engine checks
    from src.fuzzy.fuzzy_engine import FuzzyMisinformationEngine

    engine = FuzzyMisinformationEngine()
    score = engine.compute(
        {
            "source_credibility": 0.5,
            "bert_confidence": 0.5,
            "tfidf_confidence": 0.5,
            "nb_confidence": 0.5,
            "model_agreement": 0.5,
            "feedback_score": 0.5,
        }
    )
    assert 0.0 <= score <= 1.0
    print("Fuzzy engine checks passed.")

    # 4. Feedback loop on first 3 samples (no LLM judge, true labels provided)
    loop: BackpropFeedbackLoop = detector.feedback_loop
    import asyncio

    _ = asyncio.run(loop.run_cycle(SAMPLE_TEXTS[:3], SAMPLE_LABELS[:3]))
    print("Feedback loop check passed.")

    # 5. Quick evaluation on all samples
    report = detector.evaluate_quick(SAMPLE_TEXTS, SAMPLE_LABELS)
    assert "metrics" in report or "fuzzy" in report
    print("Quick evaluation check passed.")

    # 6. Flag file
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    flag_path = reports_dir / "smoke_test_passed.flag"
    flag_path.write_text(
        f"Smoke test passed at {datetime.utcnow().isoformat()}Z\n", encoding="utf-8"
    )
    print(f"Smoke test completed successfully. Flag written to {flag_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


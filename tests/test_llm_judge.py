"""Skeleton tests for the LLMJudge integration.

The real tests for LLMJudge depend on an Ollama server running locally, which
is not guaranteed in CI. For now we only validate that the class can be
imported without side effects that raise at import time.
"""

import pytest


@pytest.mark.xfail(reason="Requires local Ollama server; exercised in later phases.")
def test_llm_judge_import_and_init():
    """Instantiate LLMJudge in a best-effort way.

    This is marked xfail because `_verify_ollama_and_model` may raise if the
    local environment is not configured.
    """

    from src.evaluation.llm_judge import LLMJudge

    judge = LLMJudge()
    assert judge is not None


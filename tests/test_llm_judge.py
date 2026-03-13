import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.evaluation.llm_judge import LLMJudge
from src.utils.logger import get_logger


def _judge():
    j = LLMJudge.__new__(LLMJudge)
    j.logger = get_logger("test_judge")
    return j


def test_import():
    assert LLMJudge is not None


def test_fallback_has_all_keys():
    for k in LLMJudge.REQUIRED_KEYS:
        assert k in LLMJudge.FALLBACK_VERDICT, "missing: %s" % k


def test_parse_valid_json():
    j = _judge()
    r = j._parse_response('{"independent_verdict":"credible","judge_confidence":0.9}')
    assert r["independent_verdict"] == "credible"
    assert r["judge_confidence"] == 0.9
    for k in LLMJudge.REQUIRED_KEYS:
        assert k in r


def test_parse_invalid_json_returns_fallback():
    j = _judge()
    r = j._parse_response("this is not json at all !!!")
    assert r["independent_verdict"] == "uncertain"
    assert "parse_error" in r["flags"]


def test_parse_strips_prefix():
    j = _judge()
    r = j._parse_response(
        "Some preamble text here:\\n"
        '{"independent_verdict":"misinformation","judge_confidence":0.8}'
    )
    assert r["independent_verdict"] == "misinformation"


def test_report_empty_input():
    j = _judge()
    assert j.generate_model_report([]) == {}


def test_report_structure():
    j = _judge()
    r = j.generate_model_report([dict(LLMJudge.FALLBACK_VERDICT)] * 2)
    assert "overall" in r
    assert "bert" in r
    assert "ensemble" in r


def test_batch_with_ollama():
    try:
        judge = LLMJudge(host="http://localhost:11434")
    except RuntimeError:
        pytest.skip("Ollama not running")
    preds = {
        "bert": "credible",
        "tfidf": "credible",
        "naive_bayes": "credible",
        "ensemble": "credible",
    }
    results = judge.evaluate_batch([("Text.", preds, 0.2)])
    assert len(results) == 1
    assert "independent_verdict" in results[0]

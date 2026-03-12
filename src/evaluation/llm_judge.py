"""LLM-as-judge evaluation utilities using a local Ollama model.

This module belongs to the *evaluation* component of the pipeline. Instead of
calling a hosted Anthropic API, it talks to a local Ollama instance running a
model such as ``llama3``.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
from typing import Any, Dict, List, Optional

import requests


class LLMJudge:
    """LLM-based judge for misinformation detection using Ollama.

    This class keeps the same public async method signatures that other parts
    of the codebase expect, but internally it sends prompts to a local Ollama
    server instead of a remote Anthropic endpoint.

    The Ollama service is expected to be available at ``host`` (default
    ``http://localhost:11434``) and to have the configured model (default
    ``llama3``) pulled and ready. See the README for instructions.

    Args:
        host: Base URL of the Ollama server.
        model: Model name to use (e.g. ``llama3``).
        max_retries: Maximum number of retries on network / parsing errors.
        backoff_base: Base for exponential backoff between retries.
    """

    SYSTEM_PROMPT: str = (
        "You are an expert fact-checker and AI evaluation judge specializing in "
        "misinformation detection. You will receive: a text sample, predictions from "
        "3 models (BERT PyTorch, TF-IDF TensorFlow, Naive Bayes), the weighted "
        "ensemble prediction, a fuzzy confidence score (0.0-1.0), and feedback loop "
        "cycle metrics.\n\n"
        "Your task: independently assess credible OR misinformation. Evaluate each "
        "model as CORRECT, INCORRECT, or UNCERTAIN. Evaluate the ensemble. Identify "
        "best and worst model. Assess fuzzy calibration. Note if feedback loop is "
        "improving or degrading accuracy.\n\n"
        "Return ONLY valid JSON with these exact keys:\n"
        "independent_verdict (credible|misinformation),\n"
        "judge_confidence (float),\n"
        "bert_judgment (CORRECT|INCORRECT|UNCERTAIN),\n"
        "tfidf_judgment (CORRECT|INCORRECT|UNCERTAIN),\n"
        "naive_bayes_judgment (CORRECT|INCORRECT|UNCERTAIN),\n"
        "ensemble_judgment (CORRECT|INCORRECT|UNCERTAIN),\n"
        "best_model (bert|tfidf|naive_bayes|ensemble),\n"
        "worst_model (bert|tfidf|naive_bayes),\n"
        "fuzzy_calibration (well_calibrated|overconfident|underconfident),\n"
        "suggested_fuzzy_score (float),\n"
        "feedback_trend (improving|degrading|stable|insufficient_data),\n"
        "justification (1-2 sentence string),\n"
        "flags (array of strings)."
    )

    def __init__(
        self,
        host: Optional[str] = None,
        model: Optional[str] = None,
        max_retries: int = 3,
        backoff_base: float = 2.0,
    ) -> None:
        self.host = host or os.getenv("LLM_JUDGE_HOST", "http://localhost:11434")
        self.model = model or os.getenv("LLM_JUDGE_MODEL", "llama3")
        self.max_retries = max_retries
        self.backoff_base = backoff_base

        self._verify_ollama_and_model()

    # ------------------------------------------------------------------ utils
    def _verify_ollama_and_model(self) -> None:
        """Verify that Ollama is reachable and the model is available.

        Raises:
            RuntimeError: If Ollama is not reachable or the model is missing.
        """

        tags_url = f"{self.host.rstrip('/')}/api/tags"
        try:
            resp = requests.get(tags_url, timeout=3)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                f"Ollama not reachable at {tags_url}. "
                "Ensure Ollama is running and try: `ollama pull llama3`."
            ) from exc

        models = {m.get("name", "") for m in data.get("models", [])}
        if not any(name.startswith(self.model) for name in models):
            raise RuntimeError(
                f"Ollama is running but model '{self.model}' is not available. "
                f"Install it with: `ollama pull {self.model}`."
            )

    async def _call_with_retry(self, prompt: str) -> str:
        """Call Ollama with retries and exponential backoff.

        Args:
            prompt: Prompt string to send.

        Returns:
            str: The raw response text from the model.
        """

        async def _once() -> str:
            url = f"{self.host.rstrip('/')}/api/generate"
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
            }

            def _do_request() -> str:
                r = requests.post(url, json=payload, timeout=60)
                r.raise_for_status()
                body = r.json()
                # Ollama returns a single JSON with a 'response' field.
                return str(body.get("response", ""))

            return await asyncio.to_thread(_do_request)

        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                return await _once()
            except Exception as exc:  # pragma: no cover - network dependent
                last_exc = exc
                wait = self.backoff_base**attempt + random.random()
                await asyncio.sleep(wait)
        # If we get here, all retries failed
        raise RuntimeError(f"Ollama call failed after {self.max_retries} attempts") from last_exc

    @staticmethod
    def _fallback_judgment() -> Dict[str, Any]:
        """Return a conservative fallback judgment when parsing fails."""

        return {
            "independent_verdict": "credible",
            "judge_confidence": 0.0,
            "bert_judgment": "UNCERTAIN",
            "tfidf_judgment": "UNCERTAIN",
            "naive_bayes_judgment": "UNCERTAIN",
            "ensemble_judgment": "UNCERTAIN",
            "best_model": "ensemble",
            "worst_model": "naive_bayes",
            "fuzzy_calibration": "stable",
            "suggested_fuzzy_score": 0.5,
            "feedback_trend": "insufficient_data",
            "justification": "LLM judge could not produce a reliable structured response.",
            "flags": [],
        }

    @staticmethod
    def _extract_json_block(text: str) -> Dict[str, Any]:
        """Extract and parse the first JSON object found in text.

        Args:
            text: Raw text returned by the model.

        Returns:
            dict: Parsed JSON object.
        """

        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError("No JSON object found in model output")
        return json.loads(match.group(0))

    # ----------------------------------------------------------------- public
    async def evaluate_single(
        self,
        text: str,
        predictions: Dict[str, Any],
        fuzzy_score: float,
        cycle_metrics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Evaluate a single sample with the LLM judge.

        Args:
            text: Input text.
            predictions: Dictionary containing per-model and ensemble predictions.
            fuzzy_score: Fuzzy confidence score in [0, 1].
            cycle_metrics: Optional metrics for the current feedback cycle.

        Returns:
            dict: Judgment dictionary matching the JSON schema in SYSTEM_PROMPT.
        """

        cycle_metrics = cycle_metrics or {}
        prompt = (
            f"{self.SYSTEM_PROMPT}\n\n"
            f"TEXT:\n{text}\n\n"
            f"PREDICTIONS (JSON):\n{json.dumps(predictions)}\n\n"
            f"FUZZY_SCORE: {fuzzy_score}\n\n"
            f"CYCLE_METRICS (JSON):\n{json.dumps(cycle_metrics)}\n\n"
            "Return ONLY the JSON object as described."
        )

        try:
            raw = await self._call_with_retry(prompt)
            parsed = self._extract_json_block(raw)
            # Basic sanity: ensure required keys, else fallback
            required = {
                "independent_verdict",
                "judge_confidence",
                "bert_judgment",
                "tfidf_judgment",
                "naive_bayes_judgment",
                "ensemble_judgment",
                "best_model",
                "worst_model",
                "fuzzy_calibration",
                "suggested_fuzzy_score",
                "feedback_trend",
                "justification",
                "flags",
            }
            if not required.issubset(parsed.keys()):
                return self._fallback_judgment()
            return parsed
        except Exception:
            return self._fallback_judgment()

    async def evaluate_batch(
        self,
        dataset: List[Dict[str, Any]],
        batch_size: int = 10,
    ) -> List[Dict[str, Any]]:
        """Evaluate a batch of samples concurrently.

        Args:
            dataset: Iterable of items containing keys
                ``text``, ``predictions``, ``fuzzy_score``, and optional
                ``cycle_metrics``.
            batch_size: Maximum number of concurrent requests.

        Returns:
            list of dict: One judgment dictionary per input item.
        """

        results: List[Dict[str, Any]] = []
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i : i + batch_size]
            tasks = [
                self.evaluate_single(
                    item["text"],
                    item["predictions"],
                    item["fuzzy_score"],
                    item.get("cycle_metrics"),
                )
                for item in batch
            ]
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)
        return results



import json
import requests
from src.utils.logger import get_logger


class LLMJudge:
    """
    Local LLM-as-judge using Ollama. No API key required.
    Start Ollama: ollama serve
    Pull model:   ollama pull llama3
    """

    REQUIRED_KEYS = [
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
    ]

    FALLBACK_VERDICT = {
        "independent_verdict": "uncertain",
        "judge_confidence": 0.0,
        "bert_judgment": "UNCERTAIN",
        "tfidf_judgment": "UNCERTAIN",
        "naive_bayes_judgment": "UNCERTAIN",
        "ensemble_judgment": "UNCERTAIN",
        "best_model": "ensemble",
        "worst_model": "naive_bayes",
        "fuzzy_calibration": "well_calibrated",
        "suggested_fuzzy_score": 0.5,
        "feedback_trend": "insufficient_data",
        "justification": "Judge unavailable or parse error.",
        "flags": ["parse_error"],
    }

    def __init__(self, model="llama3", host="http://localhost:11434"):
        self.model = model
        self.host = host.rstrip("/")
        self.logger = get_logger(__name__)
        self._verify_connection()

    def _verify_connection(self):
        """Verify Ollama is running and model is available."""
        try:
            resp = requests.get("%s/api/tags" % self.host, timeout=5)
            resp.raise_for_status()
        except requests.ConnectionError:
            raise RuntimeError("Ollama not running. Start with: ollama serve")
        except Exception as e:
            raise RuntimeError("Ollama health check failed: %s" % e)

        available = [m.get("name", "") for m in resp.json().get("models", [])]
        if not any(self.model in name for name in available):
            raise RuntimeError(
                "Model '%s' not found. Run: ollama pull %s\n"
                "Available: %s" % (self.model, self.model, available)
            )
        self.logger.info("LLMJudge ready - model=%s host=%s", self.model, self.host)

    def _build_prompt(self, text, predictions, fuzzy_score, cycle_metrics=None):
        """Build evaluation prompt for the judge model."""
        metrics_str = str(cycle_metrics) if cycle_metrics else "N/A"
        return (
            "You are an expert fact-checker and AI evaluation judge.\n\n"
            "Evaluate this text and the model predictions below.\n"
            "Return ONLY a valid JSON object, no markdown, no extra text.\n\n"
            'TEXT:\n"""\n%s\n"""\n\n'
            "PREDICTIONS:\n"
            "  BERT: %s\n"
            "  TF-IDF: %s\n"
            "  Naive Bayes: %s\n"
            "  Ensemble: %s\n"
            "  Fuzzy Score: %.4f\n"
            "  Cycle metrics: %s\n\n"
            "Return ONLY this JSON:\n"
            "{\n"
            '  "independent_verdict": "credible or misinformation",\n'
            '  "judge_confidence": 0.0,\n'
            '  "bert_judgment": "CORRECT or INCORRECT or UNCERTAIN",\n'
            '  "tfidf_judgment": "CORRECT or INCORRECT or UNCERTAIN",\n'
            '  "naive_bayes_judgment": "CORRECT or INCORRECT or UNCERTAIN",\n'
            '  "ensemble_judgment": "CORRECT or INCORRECT or UNCERTAIN",\n'
            '  "best_model": "bert or tfidf or naive_bayes or ensemble",\n'
            '  "worst_model": "bert or tfidf or naive_bayes",\n'
            '  "fuzzy_calibration": "well_calibrated or overconfident or underconfident",\n'
            '  "suggested_fuzzy_score": 0.0,\n'
            '  "feedback_trend": "improving or degrading or stable or insufficient_data",\n'
            '  "justification": "1-2 sentence explanation",\n'
            '  "flags": []\n'
            "}"
        ) % (
            text,
            predictions.get("bert", "unknown"),
            predictions.get("tfidf", "unknown"),
            predictions.get("naive_bayes", "unknown"),
            predictions.get("ensemble", "unknown"),
            float(fuzzy_score),
            metrics_str,
        )

    def _parse_response(self, raw):
        """Extract and parse JSON from model response."""
        raw = raw.strip()
        if "{" in raw and "}" in raw:
            raw = raw[raw.index("{") : raw.rindex("}") + 1]
        try:
            parsed = json.loads(raw)
            for key in self.REQUIRED_KEYS:
                if key not in parsed:
                    parsed[key] = self.FALLBACK_VERDICT[key]
            return parsed
        except Exception as e:
            self.logger.warning("JSON parse failed: %s", e)
            fallback = dict(self.FALLBACK_VERDICT)
            fallback["justification"] = "Parse error: %s" % str(e)
            return fallback

    def evaluate_single(self, text, predictions, fuzzy_score, cycle_metrics=None):
        """
        Evaluate one sample with the local Ollama judge.

        Args:
            text (str): text to evaluate
            predictions (dict): bert/tfidf/naive_bayes/ensemble keys
            fuzzy_score (float): fuzzy engine output
            cycle_metrics (dict, optional): recent cycle metrics
        Returns:
            dict: judgment with all REQUIRED_KEYS
        """
        prompt = self._build_prompt(text, predictions, fuzzy_score, cycle_metrics)
        try:
            resp = requests.post(
                "%s/api/generate" % self.host,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.1, "num_predict": 400},
                },
                timeout=120,
            )
            resp.raise_for_status()
            raw = resp.json().get("response", "").strip()
            return self._parse_response(raw)
        except Exception as e:
            self.logger.warning("evaluate_single failed: %s - returning fallback", e)
            fallback = dict(self.FALLBACK_VERDICT)
            fallback["justification"] = "Judge error: %s" % str(e)
            return fallback

    def evaluate_batch(self, dataset, batch_size=10):
        """
        Evaluate a batch of samples sequentially.

        Args:
            dataset (list): list of (text, predictions, fuzzy_score) tuples
            batch_size (int): kept for API compatibility
        Returns:
            list: judgment dicts, one per sample
        """
        results = []
        for i, item in enumerate(dataset):
            try:
                if len(item) == 3:
                    text, predictions, fuzzy_score = item
                    cycle_metrics = None
                else:
                    text, predictions, fuzzy_score, cycle_metrics = item
                result = self.evaluate_single(text, predictions, fuzzy_score, cycle_metrics)
            except Exception as e:
                self.logger.warning("Batch item %d failed: %s", i, e)
                result = dict(self.FALLBACK_VERDICT)
                result["justification"] = "Batch item error: %s" % str(e)
            results.append(result)
        return results

    def generate_model_report(self, all_judgments):
        """
        Aggregate judgment statistics across all evaluations.

        Args:
            all_judgments (list): judgment dicts from evaluate_batch
        Returns:
            dict: per-model stats and overall summary
        """
        if not all_judgments:
            return {}
        report = {}
        for key in ["bert_judgment", "tfidf_judgment", "naive_bayes_judgment", "ensemble_judgment"]:
            name = key.replace("_judgment", "")
            judgments = [j.get(key, "UNCERTAIN") for j in all_judgments]
            total = len(judgments)
            correct = judgments.count("CORRECT")
            report[name] = {
                "correct": correct,
                "incorrect": judgments.count("INCORRECT"),
                "uncertain": judgments.count("UNCERTAIN"),
                "total": total,
                "agreement_rate": round(correct / total, 4) if total else 0.0,
            }
        all_flags = []
        for j in all_judgments:
            all_flags.extend(j.get("flags", []))
        flag_freq = {}
        for flag in all_flags:
            flag_freq[flag] = flag_freq.get(flag, 0) + 1
        confidences = [float(j.get("judge_confidence", 0.0)) for j in all_judgments]
        report["overall"] = {
            "total_evaluated": len(all_judgments),
            "mean_confidence": (
                round(sum(confidences) / len(confidences), 4) if confidences else 0.0
            ),
            "flag_frequency": flag_freq,
            "best_model_votes": {
                m: sum(1 for j in all_judgments if j.get("best_model") == m)
                for m in ["bert", "tfidf", "naive_bayes", "ensemble"]
            },
        }
        return report

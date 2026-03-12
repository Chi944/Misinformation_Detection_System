"""Evaluation pipeline orchestration.

This module belongs to the *evaluation* component of the pipeline. It will
later coordinate metric computation, LLM-based judging, and dashboard creation.
"""

from typing import Any, Dict


class EvaluationPipeline:
    """Stub evaluation pipeline."""

    def __init__(self) -> None:
        raise NotImplementedError("EvaluationPipeline not implemented yet.")

    def evaluate(self, dataset: Any, use_llm_judge: bool = True) -> Dict[str, Any]:
        """Evaluate models on a dataset.

        Args:
            dataset: Dataset to evaluate on.
            use_llm_judge: Whether to call the LLM judge.

        Returns:
            dict: Placeholder results dictionary.
        """

        raise NotImplementedError("EvaluationPipeline.evaluate not implemented yet.")


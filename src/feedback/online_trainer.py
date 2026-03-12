"""Online training utilities for feedback-driven updates.

This module belongs to the *feedback* component of the pipeline. It provides
lightweight update functions for BERT, TF-IDF, and Naive Bayes models that are
invoked by the backward propagation feedback loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from src.utils.logger import get_logger

LOGGER = get_logger(__name__)


@dataclass
class EWCState:
    """Elastic Weight Consolidation state for BERT.

    Component:
        Feedback / OnlineTrainer.

    Attributes:
        params: Flattened parameter vector from the reference model.
        fisher: Approximated Fisher information (same shape as params).
    """

    params: np.ndarray
    fisher: np.ndarray


class OnlineTrainer:
    """Online trainer for incremental model updates.

    This class owns references to the base models and encapsulates how they
    are updated on high-error samples during feedback cycles.

    Args:
        bert_model: Trained BERT classifier (or ``None`` in fast mode).
        tfidf_model: Trained TF-IDF DNN classifier.
        nb_model: Trained Naive Bayes wrapper.
        ewc_state: Optional pre-computed EWC state for BERT.
    """

    def __init__(
        self,
        bert_model: Any | None,
        tfidf_model: Any,
        nb_model: Any,
        ewc_state: EWCState | None = None,
    ) -> None:
        self.bert_model = bert_model
        self.tfidf_model = tfidf_model
        self.nb_model = nb_model
        self.ewc_state = ewc_state

    # ----------------------------------------------------------------- BERT EWC
    def update_bert(
        self,
        error_batch: Sequence[Dict[str, Any]],
        lr: float = 5e-6,
        steps: int = 3,
    ) -> None:
        """Mini backward pass on high-error samples for BERT.

        Args:
            error_batch: Sequence of dicts containing at least ``inputs`` and
                ``true_label`` usable by the caller to prepare BERT batches.
            lr: Learning rate for the update (default 5e-6).
            steps: Number of gradient steps per call (default 3).
        """

        if self.bert_model is None or not error_batch:
            return

        import torch

        self.bert_model.train()
        device = next(self.bert_model.parameters()).device
        optimizer = torch.optim.AdamW(self.bert_model.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()

        # Very lightweight loop: treat each sample individually.
        for step, sample in enumerate(error_batch):
            if step >= steps:
                break
            inputs = sample["inputs"]
            labels = torch.tensor([int(sample["true_label"])], device=device)

            optimizer.zero_grad()
            outputs = self.bert_model(
                input_ids=inputs["input_ids"].to(device),
                attention_mask=inputs["attention_mask"].to(device),
                token_type_ids=inputs.get("token_type_ids"),
            )
            loss = criterion(outputs, labels)

            # EWC penalty towards stored parameters if available.
            if self.ewc_state is not None:
                params = torch.cat([p.view(-1) for p in self.bert_model.parameters()])
                ref = torch.from_numpy(self.ewc_state.params).to(device)
                fisher = torch.from_numpy(self.ewc_state.fisher).to(device)
                ewc_penalty = torch.sum(fisher * (params - ref) ** 2)
                loss = loss + 0.5 * ewc_penalty

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.bert_model.parameters(), max_norm=1.0)
            optimizer.step()

        self.bert_model.eval()
        LOGGER.info("Online BERT update completed on %d samples.", min(len(error_batch), steps))

    # ------------------------------------------------------------ TF-IDF update
    def update_tfidf(
        self,
        error_batch: Sequence[Dict[str, Any]],
        epochs: int = 2,
    ) -> None:
        """Incrementally update the TF-IDF DNN on high-error samples.

        Args:
            error_batch: Sequence of dicts containing ``text`` and ``true_label``.
            epochs: Number of fit epochs on the error batch.
        """

        if not error_batch:
            return

        texts = [str(s["text"]) for s in error_batch]
        labels = np.asarray([int(s["true_label"]) for s in error_batch], dtype="int32")

        # Reuse existing vectoriser and model via the public fit interface.
        # We call fit with a small number of epochs to avoid overfitting.
        orig_epochs = getattr(self.tfidf_model, "epochs", epochs)
        self.tfidf_model.epochs = epochs
        self.tfidf_model.fit(texts, labels)
        self.tfidf_model.epochs = orig_epochs

        LOGGER.info("Online TF-IDF update completed on %d samples.", len(error_batch))

    # --------------------------------------------------------- Naive Bayes perf
    def update_naive_bayes(self, error_batch: Sequence[Dict[str, Any]]) -> None:
        """Update the Naive Bayes model using partial_fit.

        Args:
            error_batch: Sequence of dicts containing ``text`` and ``true_label``.
        """

        if not error_batch:
            return

        texts = [str(s["text"]) for s in error_batch]
        labels = np.asarray([int(s["true_label"]) for s in error_batch], dtype="int32")

        # Vectorisation is handled inside the wrapper; we simply call fit/partial.
        try:
            # If wrapper exposes partial_fit, use it for true online updates.
            if hasattr(self.nb_model, "pipeline") and hasattr(
                self.nb_model.pipeline.named_steps.get("clf"), "partial_fit"
            ):
                # Use predict_proba to force vectoriser initialisation if needed.
                _ = self.nb_model.predict_proba_np(texts)
                clf = self.nb_model.pipeline.named_steps["clf"]
                vect = self.nb_model.pipeline.named_steps["vect"]
                X = vect.transform(texts)
                clf.partial_fit(X, labels, classes=np.array([0, 1], dtype="int32"))
            else:
                self.nb_model.fit(texts, labels)
        except Exception:  # pragma: no cover - defensive
            # Fall back to full fit if partial update fails.
            self.nb_model.fit(texts, labels)

        LOGGER.info("Online Naive Bayes update completed on %d samples.", len(error_batch))


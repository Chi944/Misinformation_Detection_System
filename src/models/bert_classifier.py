"""BERT-based misinformation classifier and trainer.

This module provides a thin wrapper around a Hugging Face BERT model plus a
trainer class that can operate on small datasets (e.g. ``data/sample_train.csv``)
without requiring complex infrastructure. The implementation is intentionally
minimal but fully functional so that the rest of the pipeline can rely on it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, BertModel, get_linear_schedule_with_warmup


class BERTPoolerClassifier(nn.Module):
    """BERT encoder + pooler + linear head (common external / LIAR training layout).

    State dict keys match ``bert.*``, ``classifier.*`` as saved from many Kaggle scripts.
    """

    def __init__(self, checkpoint: str = "bert-base-uncased", dropout: float = 0.3) -> None:
        super().__init__()
        self.bert = BertModel.from_pretrained(checkpoint)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(768, 2)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        x = self.dropout(out.pooler_output)
        return self.classifier(x)


class BERTPoolerDetectorWrapper(nn.Module):
    """Adapts :class:`BERTPoolerClassifier` for code that expects ``wrapper.model(...)``."""

    def __init__(self, pooler: BERTPoolerClassifier) -> None:
        super().__init__()
        self.add_module("model", pooler)


class BERTMisinformationClassifier(nn.Module):
    """Wrapper around ``BertForSequenceClassification`` for binary labels.

    The forward pass returns probabilities for the two classes
    ``[P(credible), P(misinformation)]``.

    Args:
        checkpoint: Hugging Face model identifier, e.g. ``"bert-base-uncased"``.
        dropout: Dropout probability applied to the classification head.
    """

    def __init__(self, checkpoint: str = "bert-base-uncased", dropout: float = 0.3) -> None:
        super().__init__()
        self.model = BertForSequenceClassification.from_pretrained(
            checkpoint,
            num_labels=2,
            problem_type="single_label_classification",
        )
        # Optional extra dropout before the classifier head.
        if dropout > 0.0:
            self.model.dropout = nn.Dropout(dropout)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run a forward pass and return probabilities.

        Args:
            input_ids: Tensor of token IDs.
            attention_mask: Optional attention mask.
            token_type_ids: Optional segment IDs.

        Returns:
            Tensor of shape ``(batch_size, 2)`` with class probabilities.
        """

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        logits = outputs.logits
        return torch.softmax(logits, dim=-1)

    @torch.no_grad()
    def predict_proba(self, batch: Dict[str, torch.Tensor], device: torch.device) -> np.ndarray:
        """Convenience method to obtain probabilities for a batch.

        Args:
            batch: Mapping containing ``input_ids``, ``attention_mask``, and
                optional ``token_type_ids``.
            device: Torch device.

        Returns:
            NumPy array of shape ``(batch_size, 2)``.
        """

        self.eval()
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        tti = batch.get("token_type_ids")
        if tti is not None:
            tti = tti.to(device)
        probs = self(ids, attention_mask=mask, token_type_ids=tti)
        return probs.detach().cpu().numpy()


@dataclass
class BERTTrainingConfig:
    """Configuration for BERT training."""

    freeze_layers: int = 6
    learning_rate: float = 2e-5
    epochs: int = 5
    models_dir: Path = Path("models")


class BERTTrainer:
    """Trainer for :class:`BERTMisinformationClassifier`.

    The trainer implements a standard supervised fine‑tuning loop with the
    following characteristics:

    - AdamW optimiser with a learning rate of ``2e-5`` by default.
    - Linear warm‑up scheduler over 10% of the total training steps.
    - Five training epochs by default.
    - Mini‑batches of tokenised inputs delivered via a ``DataLoader``.
    - Mixed‑precision training using ``torch.cuda.amp.autocast`` and a
      ``GradScaler`` when a CUDA device is available.
    - Gradient clipping with ``max_norm=1.0`` to stabilise training.
    - Freezing of the bottom encoder layers of BERT to reduce the number of
      trainable parameters (six layers by default).
    - Early stopping on validation F1 score with a patience of two epochs.

    The model checkpoint is saved to ``models/bert_classifier.pt`` whenever a
    new best validation F1 score is observed.
    """

    def __init__(
        self,
        model: BERTMisinformationClassifier,
        tokenizer: Any,
        models_dir: str | Path = "models",
        freeze_layers: int = 6,
        learning_rate: float = 2e-5,
        epochs: int = 3,
        class_weights: Optional[torch.Tensor] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = BERTTrainingConfig(
            freeze_layers=freeze_layers,
            learning_rate=learning_rate,
            epochs=epochs,
            models_dir=Path(models_dir),
        )
        self.config.models_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self._freeze_encoder_layers(self.config.freeze_layers)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate)
        self.criterion = (
            nn.CrossEntropyLoss(weight=class_weights.to(self.device))
            if class_weights is not None
            else nn.CrossEntropyLoss()
        )
        self.scaler = GradScaler(enabled=self.device.type == "cuda")

    # ----------------------------------------------------------------- internals
    def _freeze_encoder_layers(self, n_layers: int) -> None:
        """Freeze the first ``n_layers`` encoder layers to speed up training."""

        encoder = self.model.model.bert.encoder
        for layer in list(encoder.layer)[:n_layers]:
            for param in layer.parameters():
                param.requires_grad = False

    # ------------------------------------------------------------------ training
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> Dict[str, float]:
        """Train the BERT classifier and return the best validation metrics.

        The training loop uses mixed precision (when CUDA is available),
        linear warm‑up scheduling, gradient clipping, and early stopping on
        validation F1 score.

        Args:
            train_loader: DataLoader over training batches.
            val_loader: DataLoader over validation batches.

        Returns:
            dict: Dictionary containing the best validation loss, accuracy and
            F1 score observed during training.
        """

        total_steps = len(train_loader) * self.config.epochs
        warmup_steps = max(1, int(0.1 * total_steps))
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        best_val_loss = float("inf")
        best_val_acc = 0.0
        best_val_f1 = 0.0
        patience = 2
        epochs_without_improvement = 0

        for epoch in range(self.config.epochs):
            self.model.train()
            for batch in train_loader:
                self.optimizer.zero_grad(set_to_none=True)
                ids = batch["input_ids"].to(self.device)
                mask = batch["attention_mask"].to(self.device)
                tti = batch.get("token_type_ids")
                if tti is not None:
                    tti = tti.to(self.device)
                labels = batch["label"].to(self.device)

                with autocast(enabled=self.device.type == "cuda"):
                    outputs = self.model(
                        input_ids=ids,
                        attention_mask=mask,
                        token_type_ids=tti,
                    )
                    loss = self.criterion(outputs, labels)

                self.scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                scheduler.step()

            # Validation and early stopping
            val_loss, val_acc, val_f1 = self._evaluate(val_loader)
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_val_loss = val_loss
                best_val_acc = val_acc
                epochs_without_improvement = 0
                self._save_best()
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    break

        return {
            "val_loss": best_val_loss,
            "val_accuracy": best_val_acc,
            "val_f1": best_val_f1,
        }

    def _evaluate(self, loader: DataLoader) -> Tuple[float, float, float]:
        """Evaluate the model on a validation loader.

        Args:
            loader: DataLoader providing validation batches.

        Returns:
            Tuple of ``(loss, accuracy, f1)`` computed over the full set.
        """

        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_examples = 0
        all_labels: List[int] = []
        all_preds: List[int] = []

        with torch.no_grad():
            for batch in loader:
                ids = batch["input_ids"].to(self.device)
                mask = batch["attention_mask"].to(self.device)
                tti = batch.get("token_type_ids")
                if tti is not None:
                    tti = tti.to(self.device)
                labels = batch["label"].to(self.device)

                outputs = self.model(
                    input_ids=ids,
                    attention_mask=mask,
                    token_type_ids=tti,
                )
                loss = self.criterion(outputs, labels)
                total_loss += float(loss.item()) * labels.size(0)
                preds = outputs.argmax(dim=1)

                total_correct += int((preds == labels).sum().item())
                total_examples += labels.size(0)
                all_labels.extend(labels.cpu().tolist())
                all_preds.extend(preds.cpu().tolist())

        avg_loss = total_loss / max(1, total_examples)
        accuracy = total_correct / max(1, total_examples)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        return avg_loss, float(accuracy), float(f1)

    # ---------------------------------------------------------------- persistence
    @property
    def checkpoint_path(self) -> Path:
        """Path for the best BERT model checkpoint."""

        return self.config.models_dir / "bert_classifier.pt"

    def _save_best(self) -> None:
        """Save the current model weights as the best checkpoint."""

        torch.save(self.model.state_dict(), self.checkpoint_path)

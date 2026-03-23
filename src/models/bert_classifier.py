"""BERT-based misinformation classifier (pooler + linear head, Colab/LIAR layout)."""

import os

import numpy as np
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

from src.utils.logger import get_logger


class BERTClassifier(nn.Module):
    """BERT-based misinformation classifier."""

    def __init__(self, config=None):
        super().__init__()
        self.config = config or {}
        self.logger = get_logger(__name__)
        self.device = "cpu"
        self._ready = False

        self.bert = None
        self.dropout = None
        self.classifier = None
        self.tokenizer = None

        self._try_load()

    def _try_load(self):
        """Load BERT model and tokenizer from disk."""
        weights_path = os.path.join("models", "bert_classifier.pt")
        train_fresh = bool(self.config.get("train_fresh", False))

        try:
            self.bert = BertModel.from_pretrained("bert-base-uncased")
            self.dropout = nn.Dropout(0.3)
            self.classifier = nn.Linear(768, 2)
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

            if not train_fresh and os.path.exists(weights_path):
                state = torch.load(weights_path, map_location=self.device)
                self.load_state_dict(state)
                self.logger.info("BERT loaded from %s", weights_path)
            elif train_fresh:
                self.logger.info("BERT train_fresh=True: not loading disk weights")
            else:
                self.logger.warning("BERT weights not found at %s", weights_path)

            self.eval()
            self._ready = True

        except Exception as e:
            self.logger.warning("BERT load failed: %s", e)
            self._ready = False

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(self.dropout(out.pooler_output))

    def load_weights(self, path=None):
        """Load state dict from ``models/bert_classifier.pt`` or given path."""
        weights_path = path or os.path.join("models", "bert_classifier.pt")
        if not os.path.exists(weights_path):
            self.logger.warning("load_weights: missing %s", weights_path)
            return
        try:
            dev = next(self.parameters()).device
        except StopIteration:
            dev = torch.device("cpu")
        state = torch.load(weights_path, map_location=dev)
        self.load_state_dict(state, strict=True)
        self.eval()
        self._ready = True
        self.logger.info("BERT weights loaded from %s", weights_path)

    def predict(self, texts):
        """
        Return P(misinfo) for each text.

        Args:
            texts: list of strings
        Returns:
            numpy array of floats between 0 and 1
        """
        if not self._ready or self.tokenizer is None:
            self.logger.warning("BERT not ready - returning neutral 0.5")
            return np.full(len(list(texts)), 0.5)

        texts = [str(t) for t in texts]

        try:
            enc = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            dev = next(self.parameters()).device
            ids = enc["input_ids"].to(dev)
            mask = enc["attention_mask"].to(dev)

            with torch.no_grad():
                logits = self.forward(ids, mask)
                probs = torch.softmax(logits, dim=1)
                return probs[:, 1].cpu().numpy()

        except Exception as e:
            self.logger.warning("BERT predict failed: %s", e)
            return np.full(len(texts), 0.5)

    def predict_proba(self, texts):
        """Return (n, 2) array of [P(credible), P(misinfo)]."""
        p_misinfo = self.predict(texts)
        return np.column_stack([1 - p_misinfo, p_misinfo])

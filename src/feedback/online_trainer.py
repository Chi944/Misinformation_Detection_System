import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW

from src.utils.logger import get_logger


class OnlineTrainer:
    """
    Handles incremental model updates during feedback cycles.

    Updates each of the 3 models on high-error samples only, using
    techniques that prevent catastrophic forgetting:
    - BERT: mini backward pass with EWC penalty and very low LR
    - TF-IDF: Keras fit on error batch only
    - Naive Bayes: sklearn partial_fit for true online learning

    Part of the backward propagation feedback loop pipeline.

    Args:
        bert_model: BERTClassifier (or compatible nn.Module with forward(ids, mask))
        bert_tokenizer: HuggingFace tokenizer for BERT
        tfidf_model: TFIDFModel instance
        tfidf_vectorizer: fitted TfidfVectorizer
        nb_model: fitted MultinomialNB (or CalibratedClassifierCV)
        nb_vectorizer: fitted CountVectorizer for Naive Bayes
        device (str): 'cuda' or 'cpu'
    """

    def __init__(
        self,
        bert_model,
        bert_tokenizer,
        tfidf_model,
        tfidf_vectorizer,
        nb_model,
        nb_vectorizer,
        device="cpu",
    ):
        self.bert_model = bert_model
        self.bert_tokenizer = bert_tokenizer
        self.tfidf_model = tfidf_model
        self.tfidf_vectorizer = tfidf_vectorizer
        self.nb_model = nb_model
        self.nb_vectorizer = nb_vectorizer
        self.device = device
        self.logger = get_logger(__name__)

        # Store initial BERT parameter values for EWC
        self._ewc_params = {
            name: param.clone().detach()
            for name, param in bert_model.named_parameters()
            if param.requires_grad
        }
        # Initialise Fisher information as ones (uniform importance)
        self._ewc_fisher = {
            name: torch.ones_like(param)
            for name, param in bert_model.named_parameters()
            if param.requires_grad
        }

    def _ewc_penalty(self, model) -> torch.Tensor:
        """
        Elastic Weight Consolidation penalty to prevent catastrophic
        forgetting during incremental BERT updates.

        Penalises large deviations from stored parameter values,
        weighted by estimated Fisher information.

        Args:
            model: the BERT model being updated
        Returns:
            torch.Tensor: scalar EWC loss term
        """
        penalty = torch.tensor(0.0, device=self.device)
        for name, param in model.named_parameters():
            if name in self._ewc_params:
                stored = self._ewc_params[name].to(self.device)
                fisher = self._ewc_fisher[name].to(self.device)
                penalty = penalty + (fisher * (param - stored).pow(2)).sum()
        return 0.5 * penalty

    def update_bert(self, error_batch, lr=5e-6, steps=3):
        """
        Mini backward pass on high-error samples to correct BERT weights.

        Uses a very low learning rate and few gradient steps to avoid
        catastrophic forgetting. EWC penalty protects important weights.

        Args:
            error_batch (list): list of (text, true_label) tuples
            lr (float): learning rate, default 5e-6 (vs training 2e-5)
            steps (int): gradient steps per call, default 3
        """
        if not error_batch:
            return
        self.bert_model.train()
        optimizer = AdamW([p for p in self.bert_model.parameters() if p.requires_grad], lr=lr)
        criterion = nn.CrossEntropyLoss()
        try:
            for _step in range(steps):
                for text, label in error_batch:
                    enc = self.bert_tokenizer(
                        text,
                        return_tensors="pt",
                        max_length=512,
                        truncation=True,
                        padding=True,
                    )
                    enc = {k: v.to(self.device) for k, v in enc.items()}
                    label_t = torch.tensor([int(label)], device=self.device)
                    logits = self.bert_model(
                        enc["input_ids"], enc["attention_mask"]
                    )
                    ce_loss = criterion(logits, label_t)
                    ewc_loss = self._ewc_penalty(self.bert_model)
                    loss = ce_loss + 0.1 * ewc_loss
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.bert_model.parameters(), max_norm=1.0)
                    optimizer.step()
            self.logger.info("BERT updated on %d samples, %d steps", len(error_batch), steps)
        except Exception as e:
            self.logger.warning("BERT update failed: %s", e)
        finally:
            self.bert_model.eval()

    def update_tfidf(self, error_batch, epochs=2):
        """
        Incremental TF-IDF model update on high-error samples.

        Calls Keras model.fit() on the error batch only.
        Fewer epochs and smaller batches prevent overfitting.

        Args:
            error_batch (list): list of (text, true_label) tuples
            epochs (int): Keras training epochs, default 2
        """
        if not error_batch:
            return
        try:
            texts, labels = zip(*error_batch)
            X = self.tfidf_model.transform_features(list(texts))
            y = np.array(labels, dtype=np.int32)
            self.tfidf_model.model.fit(
                X,
                y,
                epochs=epochs,
                batch_size=max(4, len(error_batch)),
                verbose=0,
            )
            self.logger.info("TF-IDF updated on %d samples, %d epochs", len(error_batch), epochs)
        except Exception as e:
            self.logger.warning("TF-IDF update failed: %s", e)

    def update_naive_bayes(self, error_batch):
        """
        True online learning for Naive Bayes via sklearn partial_fit.

        No full retraining needed — partial_fit updates the class
        priors and likelihoods incrementally.

        Args:
            error_batch (list): list of (text, true_label) tuples
        """
        if not error_batch:
            return
        try:
            texts, labels = zip(*error_batch)
            X = self.nb_vectorizer.transform(list(texts))
            y = np.array(labels, dtype=np.int32)
            self.nb_model.partial_fit(X, y, classes=[0, 1])
            self.logger.info("Naive Bayes partial_fit on %d samples", len(error_batch))
        except Exception as e:
            self.logger.warning("Naive Bayes update failed: %s", e)

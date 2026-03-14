import os

import numpy as np

from src.utils.logger import get_logger


class AccuracyGateError(Exception):
    """Raised when a model fails to meet accuracy gate thresholds."""


class MasterTrainer:
    """
    Master trainer for all 3 models + ensemble.

    Handles:
      - training BERT (PyTorch) model
      - training TF-IDF (TensorFlow) model
      - training Naive Bayes (sklearn) model
      - enforcing accuracy gates

    Args:
        config_path (str): path to config.yaml
    """

    def __init__(self, config_path="config.yaml"):
        self.config_path = config_path
        self.logger = get_logger(__name__)
        self.config = self._load_config(config_path)

    def _load_config(self, path):
        import yaml

        with open(path, "r") as f:
            return yaml.safe_load(f)

    def _gate_check(self, model_name, metrics):
        """
        Enforce accuracy gates for each model.

        Accuracy gates are:
          bert:     acc>=0.78, precision>=0.76, f1>=0.77
          tfidf:    acc>=0.76, precision>=0.75, f1>=0.75
          nb:       acc>=0.75, precision>=0.75, f1>=0.74
          ensemble: acc>=0.82, precision>=0.80, f1>=0.81
        """
        gates = {
            "bert": {"accuracy": 0.78, "precision": 0.76, "f1": 0.77},
            "tfidf": {"accuracy": 0.76, "precision": 0.75, "f1": 0.75},
            "nb": {"accuracy": 0.75, "precision": 0.75, "f1": 0.74},
            "naive_bayes": {"accuracy": 0.75, "precision": 0.75, "f1": 0.74},
            "ensemble": {"accuracy": 0.82, "precision": 0.80, "f1": 0.81},
        }
        if model_name not in gates:
            return
        req = gates[model_name]
        for k, thr in req.items():
            if float(metrics.get(k, 0.0)) < float(thr):
                raise AccuracyGateError(
                    "%s failed gate %s: %.4f < %.4f"
                    % (model_name, k, float(metrics.get(k, 0.0)), float(thr))
                )

    def train_all(self, dataset, skip_gates=False, skip_bert=False):
        """
        Train all three models on the provided dataset.

        Args:
            dataset: MisinformationDataset instance
            skip_gates (bool): if True skip accuracy gate checks
            skip_bert (bool): if True skip BERT training (faster on CPU)
        Returns:
            dict: per-model training metrics
        """
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        results = {}

        train_texts, train_labels = dataset.to_sklearn("train")
        val_texts, val_labels = dataset.to_sklearn("val")
        train_texts = [str(t) for t in train_texts]
        val_texts = [str(t) for t in val_texts]

        self.logger.info(
            "Training on %d samples, validating on %d",
            len(train_texts),
            len(val_texts),
        )

        # --- Train Naive Bayes ---
        try:
            self.logger.info("Training Naive Bayes...")
            from src.models.naive_bayes_model import TFNaiveBayesWrapper

            nb_cfg = self.config.get("models", {}).get("naive_bayes", {})
            nb = TFNaiveBayesWrapper(
                models_dir="models",
                max_features=nb_cfg.get("max_features", 50000),
                ngram_range=tuple(nb_cfg.get("ngram_range", [1, 2])),
                alpha=nb_cfg.get("alpha", 1.0),
            )
            nb.fit(train_texts, train_labels)
            self.logger.info("Naive Bayes trained")
            proba = nb._predict_proba_numpy(val_texts)
            pred_labels = [1 if float(p[1]) >= 0.5 else 0 for p in proba]
            from sklearn.metrics import accuracy_score, f1_score, precision_score

            acc = accuracy_score(val_labels, pred_labels)
            f1 = f1_score(val_labels, pred_labels, zero_division=0)
            prec = precision_score(val_labels, pred_labels, zero_division=0)
            results["naive_bayes"] = {"accuracy": acc, "f1": f1, "precision": prec}
            results["nb"] = results["naive_bayes"]
            self.logger.info("NB val: acc=%.4f f1=%.4f", acc, f1)
            if not skip_gates:
                self._gate_check("nb", results["nb"])
        except Exception as e:
            self.logger.error("Naive Bayes training failed: %s", e)
            results["naive_bayes"] = results["nb"] = {
                "accuracy": 0.0,
                "f1": 0.0,
                "precision": 0.0,
            }

        # --- Train TF-IDF ---
        try:
            self.logger.info("Training TF-IDF DNN...")
            from src.models.tfidf_model import TFIDFModel

            tf_cfg = self.config.get("models", {}).get("tfidf", {})
            # Cap max_features to avoid OOM on large datasets (e.g. 80k samples)
            max_feat = min(tf_cfg.get("max_features", 100000), 25000)
            tfidf = TFIDFModel(
                models_dir="models",
                max_features=max_feat,
                ngram_range=tuple(tf_cfg.get("ngram_range", [1, 3])),
                char_ngram_range=tuple(tf_cfg.get("char_ngram_range", [2, 4])),
                learning_rate=tf_cfg.get("learning_rate", 1.0e-3),
                epochs=tf_cfg.get("epochs", 20),
            )
            # Use subset if very large to avoid memory issues
            n_train = len(train_texts)
            if n_train > 25000:
                import random
                idx = list(range(n_train))
                random.seed(42)
                random.shuffle(idx)
                use_idx = idx[:25000]
                tfidf.fit([train_texts[i] for i in use_idx], [train_labels[i] for i in use_idx])
            else:
                tfidf.fit(train_texts, train_labels)
            self.logger.info("TF-IDF trained")
            preds = tfidf.predict_proba(val_texts)
            if preds is not None and len(preds) > 0:
                pred_labels = [1 if float(p[1]) >= 0.5 else 0 for p in preds]
                from sklearn.metrics import accuracy_score, f1_score, precision_score

                acc = accuracy_score(val_labels[: len(pred_labels)], pred_labels)
                f1 = f1_score(val_labels[: len(pred_labels)], pred_labels, zero_division=0)
                prec = precision_score(
                    val_labels[: len(pred_labels)], pred_labels, zero_division=0
                )
            else:
                acc = f1 = prec = 0.0
            results["tfidf"] = {"accuracy": acc, "f1": f1, "precision": prec}
            self.logger.info("TF-IDF val: acc=%.4f f1=%.4f", acc, f1)
            if not skip_gates:
                self._gate_check("tfidf", results["tfidf"])
        except Exception as e:
            self.logger.error("TF-IDF training failed: %s", e)
            results["tfidf"] = {"accuracy": 0.0, "f1": 0.0, "precision": 0.0}

        # --- Train BERT ---
        if skip_bert:
            results["bert"] = {"accuracy": 0.0, "f1": 0.0, "precision": 0.0}
            self.logger.info("BERT skipped (--skip-bert)")
        else:
            try:
                self.logger.info("Training BERT...")
                import torch
                from transformers import BertTokenizer

                from src.models.bert_classifier import BERTMisinformationClassifier

                cfg = self.config.get("models", {}).get("bert", {})
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                checkpoint = cfg.get("checkpoint", "bert-base-uncased")
                model = BERTMisinformationClassifier(
                    checkpoint=checkpoint,
                    dropout=cfg.get("dropout", 0.3),
                ).to(device)
                tokenizer = BertTokenizer.from_pretrained(checkpoint)
                optimizer = torch.optim.AdamW(
                    model.parameters(), lr=cfg.get("learning_rate", 2e-5)
                )
                max_len = min(cfg.get("max_length", 512), 128)
                epochs = self.config.get("training", {}).get("epochs", 3)
                if "epochs" in cfg:
                    epochs = cfg["epochs"]
                batch_sz = cfg.get("batch_size", 16)
                batch_sz = min(batch_sz, 32)

                model.train()
                for epoch in range(epochs):
                    import random

                    combined = list(zip(train_texts, train_labels))
                    random.shuffle(combined)
                    total_loss = 0.0
                    n_batches = 0
                    n_use = min(len(combined), 8000)
                    for i in range(0, n_use, batch_sz):
                        batch = combined[i : i + batch_sz]
                        bt, bl = zip(*batch)
                        enc = tokenizer(
                            list(bt),
                            padding=True,
                            truncation=True,
                            max_length=max_len,
                            return_tensors="pt",
                        )
                        ids = enc["input_ids"].to(device)
                        mask = enc["attention_mask"].to(device)
                        lbls = torch.tensor(list(bl), dtype=torch.long).to(device)
                        optimizer.zero_grad()
                        logits = model.model(input_ids=ids, attention_mask=mask).logits
                        loss = torch.nn.functional.cross_entropy(logits, lbls)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        total_loss += loss.item()
                        n_batches += 1
                    avg_loss = total_loss / max(n_batches, 1)
                    self.logger.info("BERT epoch %d/%d loss=%.4f", epoch + 1, epochs, avg_loss)

                # Save BERT for detector
                os.makedirs("models", exist_ok=True)
                torch.save(model.model.state_dict(), "models/bert_classifier.pt")
                self.logger.info("BERT checkpoint saved to models/bert_classifier.pt")

                # Evaluate BERT on val set
                model.eval()
                all_preds = []
                with torch.no_grad():
                    val_sub = val_texts[: min(len(val_texts), 2000)]
                    for i in range(0, len(val_sub), batch_sz):
                        bt = val_sub[i : i + batch_sz]
                        enc = tokenizer(
                            list(bt),
                            padding=True,
                            truncation=True,
                            max_length=max_len,
                            return_tensors="pt",
                        )
                        ids = enc["input_ids"].to(device)
                        mask = enc["attention_mask"].to(device)
                        out = model.model(input_ids=ids, attention_mask=mask).logits
                        preds_batch = torch.argmax(out, dim=1).cpu().tolist()
                        all_preds.extend(preds_batch)
                val_subset = val_labels[: len(all_preds)]
                from sklearn.metrics import accuracy_score, f1_score, precision_score

                acc = accuracy_score(val_subset, all_preds)
                f1 = f1_score(val_subset, all_preds, zero_division=0)
                prec = precision_score(val_subset, all_preds, zero_division=0)
                results["bert"] = {"accuracy": acc, "f1": f1, "precision": prec}
                self.logger.info("BERT val: acc=%.4f f1=%.4f", acc, f1)
                if not skip_gates:
                    self._gate_check("bert", results["bert"])
            except Exception as e:
                self.logger.error("BERT training failed: %s", e)
                results["bert"] = {"accuracy": 0.0, "f1": 0.0, "precision": 0.0}

        # --- Ensemble (weighted average of available models) ---
        bert_acc = results.get("bert", {}).get("accuracy", 0.0)
        tf_acc = results.get("tfidf", {}).get("accuracy", 0.0)
        nb_acc = results.get("naive_bayes", results.get("nb", {})).get("accuracy", 0.0)
        ens_acc = bert_acc * 0.5 + tf_acc * 0.3 + nb_acc * 0.2
        bert_f1 = results.get("bert", {}).get("f1", 0.0)
        tf_f1 = results.get("tfidf", {}).get("f1", 0.0)
        nb_f1 = results.get("naive_bayes", results.get("nb", {})).get("f1", 0.0)
        ens_f1 = bert_f1 * 0.5 + tf_f1 * 0.3 + nb_f1 * 0.2
        results["ensemble"] = {"accuracy": ens_acc, "f1": ens_f1, "precision": 0.0}
        self.logger.info("Ensemble val: acc=%.4f f1=%.4f", ens_acc, ens_f1)
        if not skip_gates:
            self._gate_check("ensemble", results["ensemble"])

        return results

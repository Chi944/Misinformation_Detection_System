import os

import yaml

from src.utils.logger import get_logger
from src.utils.domain_credibility import DomainCredibility
from src.utils.explainability import Explainability


class MisinformationDetector:
    """
    Master class for the full misinformation detection pipeline.
    Coordinates all 3 models, ensemble, fuzzy engine, feedback loop,
    LLM judge, and evaluation pipeline.

    Args:
        config (str or dict): path to config.yaml or pre-loaded dict
        fast_mode (bool): skip heavy model downloads for CI/smoke tests
    """

    def __init__(self, config="config.yaml", fast_mode=False):
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        self.logger = get_logger(__name__)
        self.fast_mode = fast_mode

        if isinstance(config, dict):
            self.config = config
        else:
            with open(config, "r") as f:
                self.config = yaml.safe_load(f)

        self.device = self._setup_device()
        self.bert_model = None
        self.bert_tokenizer = None
        self.tfidf_model = None
        self.tfidf_vectorizer = None
        self.nb_model = None
        self.nb_vectorizer = None
        self.ensemble = None
        self.fuzzy_engine = None
        self.feedback_loop = None
        self.llm_judge = None
        self.eval_pipeline = None
        self.git_manager = None

        # Phase 17 enhancements
        self.domain_credibility = DomainCredibility()
        self.explainability = Explainability()

        self._init_fuzzy()

        if self.fast_mode:
            self.logger.info("fast_mode=True: skipping model/LLM/eval init")
        else:
            self._init_models()
            self._init_ensemble()
            self._init_llm_judge()
            self._init_feedback_loop()
            self._init_evaluation()
            self._init_git_manager()

        self.logger.info(
            "MisinformationDetector ready (fast_mode=%s device=%s)", fast_mode, self.device
        )

    def _setup_device(self):
        """Auto-detect GPU for PyTorch and TensorFlow."""
        if self.fast_mode:
            self.logger.info("fast_mode=True: skipping GPU detection")
            return "cpu"
        try:
            import torch

            if torch.cuda.is_available():
                self.logger.info("PyTorch CUDA available")
                return "cuda"
        except ImportError:
            pass
        try:
            import tensorflow as tf

            gpus = tf.config.list_physical_devices("GPU")
            if gpus:
                self.logger.info("TensorFlow GPU: %s", gpus)
        except ImportError:
            pass
        self.logger.info("Using CPU")
        return "cpu"

    def _init_models(self):
        """Initialise all 3 individual models with lazy imports."""
        cfg = self.config.get("models", {})

        if self.fast_mode:
            self.logger.info("fast_mode=True: skipping BERT init")
        else:
            try:
                import torch

                from src.models.bert_classifier import BERTClassifier

                bert_cfg = dict(cfg.get("bert", {}))
                self.bert_model = BERTClassifier(config=bert_cfg)
                self.bert_model.to(self.device)
                self.bert_model.eval()
                self.bert_tokenizer = getattr(self.bert_model, "tokenizer", None)
                self.logger.info("BERT initialised")
            except Exception as e:
                self.logger.warning("BERT init failed: %s", e)

        try:
            from src.models.tfidf_model import TFIDFModel

            tf_cfg = cfg.get("tfidf", {})
            self.tfidf_model = TFIDFModel(
                models_dir="models",
                max_features=tf_cfg.get("max_features", 100000),
                ngram_range=tuple(tf_cfg.get("ngram_range", [1, 3])),
                char_ngram_range=tuple(tf_cfg.get("char_ngram_range", [2, 4])),
                learning_rate=tf_cfg.get("learning_rate", 1.0e-3),
                epochs=tf_cfg.get("epochs", 20),
            )
            self.tfidf_vectorizer = getattr(self.tfidf_model, "word_vectorizer", None)
            self.logger.info("TF-IDF initialised")
        except Exception as e:
            self.logger.warning("TF-IDF init failed: %s", e)

        try:
            from src.models.naive_bayes_model import TFNaiveBayesWrapper

            self.nb_model = TFNaiveBayesWrapper()
            self.nb_vectorizer = getattr(self.nb_model, "vectorizer", None)
            self.logger.info("Naive Bayes initialised")
        except Exception as e:
            self.logger.warning("NB init failed: %s", e)

    def _init_ensemble(self):
        """Initialise ensemble with all 3 models."""
        try:
            from src.models.ensemble_detector import EnsembleDetector

            self.ensemble = EnsembleDetector(
                config=self.config,
                bert_model=self.bert_model,
                tfidf_model=self.tfidf_model,
                nb_model=self.nb_model,
                bert_tokenizer=self.bert_tokenizer,
                device=self.device,
            )
            self.logger.info("Ensemble initialised")
        except Exception as e:
            self.logger.warning("Ensemble init failed: %s", e)

    def _init_fuzzy(self):
        """Initialise fuzzy inference engine."""
        try:
            from src.fuzzy.fuzzy_engine import FuzzyMisinformationEngine

            self.fuzzy_engine = FuzzyMisinformationEngine()
            self.logger.info("Fuzzy engine initialised")
        except Exception as e:
            self.logger.warning("Fuzzy init failed: %s", e)

    def _init_llm_judge(self):
        """Initialise local Ollama LLM judge."""
        try:
            from src.evaluation.llm_judge import LLMJudge

            judge_cfg = self.config.get("llm_judge", {})
            self.llm_judge = LLMJudge(
                model=judge_cfg.get("model", "llama3"),
                host=judge_cfg.get("host", "http://localhost:11434"),
            )
            self.logger.info("LLM Judge initialised")
        except Exception as e:
            self.logger.warning("LLM Judge init failed (Ollama may not be running): %s", e)

    def _init_feedback_loop(self):
        """Initialise backward propagation feedback loop."""
        try:
            from src.feedback.backprop_loop import BackpropFeedbackLoop

            self.feedback_loop = BackpropFeedbackLoop(self, self.config)
            self.logger.info("Feedback loop initialised")
        except Exception as e:
            self.logger.warning("Feedback loop init failed: %s", e)

    def _init_evaluation(self):
        """Initialise evaluation pipeline."""
        try:
            from src.evaluation.pipeline import EvaluationPipeline

            self.eval_pipeline = EvaluationPipeline(self, self.config)
            self.logger.info("Evaluation pipeline initialised")
        except Exception as e:
            self.logger.warning("Eval pipeline init failed: %s", e)

    def _init_git_manager(self):
        """Initialise git automation manager."""
        try:
            from src.utils.git_manager import GitManager

            self.git_manager = GitManager()
            self.logger.info("Git manager initialised")
        except Exception as e:
            self.logger.warning("Git manager init failed: %s", e)

    def predict(self, text, url=None, explain=False):
        """
        Run full prediction on one text sample.

        Args:
            text (str): input text to classify
            url (str, optional): source URL; used to adjust probability based
                on domain credibility.
            explain (bool): if True, include word-level explainability output.
        Returns:
            dict: crisp_label, ensemble_probability, fuzzy_score,
                  model_breakdown, ensemble_weights, model_agreement,
                  optionally `source_credibility` and `explanation`.
        """
        if self.ensemble is None:
            result = self._fallback_predict(text)
        else:
            result = self.ensemble.predict(text)

        # Phase 17.4: domain credibility adjustment
        try:
            original_prob = float(result.get("ensemble_probability", 0.5))

            # Only adjust if we have a URL or we can find a domain in the text.
            domain_hint = url if url else self.domain_credibility.extract_domain(text)
            if url is not None or domain_hint is not None:
                score = self.domain_credibility.get_score(url or domain_hint)
                domain = (
                    self.domain_credibility.extract_domain(url)
                    if url is not None
                    else domain_hint
                )
                adjusted_prob = self.domain_credibility.adjust_probability(
                    original_prob, url or domain_hint
                )
                adjusted_flag = abs(float(adjusted_prob) - float(original_prob)) > 1e-9

                result["ensemble_probability"] = float(adjusted_prob)
                result["crisp_label"] = (
                    "misinformation" if result["ensemble_probability"] >= 0.5 else "credible"
                )

                crisp_int = 1 if result["crisp_label"] == "misinformation" else 0
                labels = [
                    int(result["model_breakdown"]["bert"]["label"]),
                    int(result["model_breakdown"]["tfidf"]["label"]),
                    int(result["model_breakdown"]["naive_bayes"]["label"]),
                ]
                result["model_agreement"] = float(sum(l == crisp_int for l in labels) / len(labels))

                result["source_credibility"] = {
                    "domain": domain,
                    "score": float(score),
                    "label": self.domain_credibility.get_label(score),
                    "adjusted": bool(adjusted_flag),
                }
        except Exception as e:  # pragma: no cover - defensive
            self.logger.warning("Source credibility adjustment failed: %s", e)

        # Optional phase 17.5: explainability
        if explain:
            try:
                result["explanation"] = self.explainability.explain(
                    text,
                    tfidf_model=self.tfidf_model,
                    nb_model=self.nb_model,
                )
            except Exception as e:  # pragma: no cover - defensive
                self.logger.warning("Explainability failed: %s", e)
                result["explanation"] = {
                    "method": "none",
                    "top_words": [],
                    "explanation": "Explanation not available.",
                }

        if self.fuzzy_engine is not None:
            result["fuzzy_score"] = self.fuzzy_engine.compute(
                {
                    "source_credibility": 0.5,
                    "bert_confidence": result["model_breakdown"]["bert"]["confidence"],
                    "tfidf_confidence": result["model_breakdown"]["tfidf"]["confidence"],
                    "nb_confidence": result["model_breakdown"]["naive_bayes"]["confidence"],
                    "model_agreement": result["model_agreement"],
                    "feedback_score": 0.5,
                }
            )
        else:
            result["fuzzy_score"] = 0.5

        return result

    def _fallback_predict(self, text):
        """Return neutral prediction when ensemble is not loaded."""
        self.logger.warning("Ensemble not loaded - returning neutral")
        return {
            "crisp_label": "credible",
            "ensemble_probability": 0.5,
            "fuzzy_score": 0.5,
            "model_breakdown": {
                "bert": {"label": 0, "confidence": 0.5},
                "tfidf": {"label": 0, "confidence": 0.5},
                "naive_bayes": {"label": 0, "confidence": 0.5},
            },
            "ensemble_weights": {"bert": 0.50, "tfidf": 0.30, "naive_bayes": 0.20},
            "model_agreement": 1.0,
        }

    def evaluate(self, dataset, use_llm_judge=True):
        """
        Run full evaluation pipeline.

        Args:
            dataset: dataset or list of (text, label) tuples
            use_llm_judge (bool): whether to call LLM judge
        Returns:
            dict: evaluation report
        """
        if self.eval_pipeline is None:
            self.logger.warning("Eval pipeline not initialised")
            return {}
        return self.eval_pipeline.evaluate(dataset, use_llm_judge)

    def evaluate_quick(self, texts, labels):
        """
        Fast evaluation without LLM judge. Used by CI and smoke tests.

        Args:
            texts (list): text samples
            labels (list): ground truth labels
        Returns:
            dict: evaluation report without LLM judge
        """
        dataset = list(zip(texts, labels))
        if self.eval_pipeline is not None:
            return self.eval_pipeline.evaluate(dataset, use_llm_judge=False)
        return {"ensemble": {"accuracy": 0.0, "f1": 0.0}}

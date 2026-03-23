import numpy as np

from src.utils.logger import get_logger


class Explainability:
    """
    Extract top words driving TF-IDF and Naive Bayes predictions.

    The output is structured to be human-friendly and also to provide a
    machine-readable summary of `misinfo_indicators` for UI/reporting.
    """

    def __init__(self):
        self.logger = get_logger(__name__)

    def _empty_explanation(self, reason=""):
        return {
            "method": "none",
            "top_words": [],
            "explanation": "Explanation not available: %s" % reason,
        }

    def _format_explanation(self, top_words):
        if not top_words:
            return "No significant words found"
        words = [w["word"] for w in top_words[:3]]
        return "Key words: %s" % ", ".join(words)

    def _format_nb_explanation(self, misinfo_words, credible_words):
        parts = []
        if misinfo_words:
            words = [w["word"] for w in misinfo_words]
            parts.append("Misinfo indicators: %s" % ", ".join(words))
        if credible_words:
            words = [w["word"] for w in credible_words]
            parts.append("Credible indicators: %s" % ", ".join(words))
        return ". ".join(parts) if parts else "No significant words found"

    def explain_tfidf(self, text, tfidf_model, top_n=5):
        """Return top words that drove the TF-IDF prediction."""
        try:
            if tfidf_model is None or getattr(tfidf_model, "word_vectorizer", None) is None:
                return self._empty_explanation("TF-IDF not loaded")

            texts = [str(text)]
            word_vec = tfidf_model.word_vectorizer
            word_feats = word_vec.transform(texts)

            feature_names = np.array(word_vec.get_feature_names_out())
            scores = word_feats.toarray()[0]

            top_idx = np.argsort(scores)[::-1][: top_n * 3]
            top_idx = [i for i in top_idx if scores[i] > 0][:top_n]

            top_words = [
                {"word": feature_names[i], "score": round(float(scores[i]), 4)}
                for i in top_idx
            ]

            return {
                "method": "tfidf",
                "top_words": top_words,
                "explanation": self._format_explanation(top_words),
            }
        except Exception as e:  # pragma: no cover - defensive
            self.logger.warning("TF-IDF explain failed: %s", e)
            return self._empty_explanation(str(e))

    def explain_naive_bayes(self, text, nb_model, top_n=5):
        """Return top words that drove the Naive Bayes prediction."""
        try:
            if nb_model is None or getattr(nb_model, "vectorizer", None) is None:
                return self._empty_explanation("NB not loaded")

            # Ensure model is loaded
            if getattr(nb_model, "_calibrated_clf", None) is None and hasattr(nb_model, "_load"):
                try:
                    nb_model._load()
                except Exception:
                    pass

            calibrated = getattr(nb_model, "_calibrated_clf", None)
            if calibrated is None:
                return self._empty_explanation("NB calibrated classifier not loaded")

            base_estimator = (
                getattr(calibrated, "estimator_", None)
                or getattr(calibrated, "estimator", None)
                or getattr(calibrated, "base_estimator", None)
            )
            if base_estimator is None or not hasattr(base_estimator, "feature_log_prob_"):
                return self._empty_explanation("NB has no feature_log_prob_")

            vectorizer = nb_model.vectorizer
            feature_names = np.array(vectorizer.get_feature_names_out())
            texts = [str(text)]
            X = vectorizer.transform(texts).toarray()[0]

            # diff = class1 - class0 log prob per feature
            log_probs = base_estimator.feature_log_prob_
            if log_probs.shape[0] >= 2:
                diff = log_probs[1] - log_probs[0]
            else:
                diff = log_probs[0]

            contributions = X * diff
            top_misinfo_idx = np.argsort(contributions)[::-1][:top_n]
            top_credible_idx = np.argsort(contributions)[:top_n]

            top_misinfo = [
                {"word": feature_names[i], "contribution": round(float(contributions[i]), 4)}
                for i in top_misinfo_idx
                if X[i] > 0
            ]
            top_credible = [
                {
                    "word": feature_names[i],
                    "contribution": round(float(abs(contributions[i])), 4),
                }
                for i in top_credible_idx
                if X[i] > 0 and contributions[i] < 0
            ]

            return {
                "method": "naive_bayes",
                "misinfo_words": top_misinfo[:top_n],
                "credible_words": top_credible[:top_n],
                "explanation": self._format_nb_explanation(top_misinfo[:3], top_credible[:3]),
            }
        except Exception as e:  # pragma: no cover - defensive
            self.logger.warning("NB explain failed: %s", e)
            return self._empty_explanation(str(e))

    def explain(self, text, tfidf_model=None, nb_model=None, top_n=5):
        """
        Return combined explainability output from all available models.
        """
        result = {
            "text_length": len(str(text).split()),
            "models": {},
        }

        if tfidf_model is not None:
            result["models"]["tfidf"] = self.explain_tfidf(text, tfidf_model, top_n)

        if nb_model is not None:
            result["models"]["naive_bayes"] = self.explain_naive_bayes(text, nb_model, top_n)

        all_misinfo_words = set()
        all_credible_words = set()

        for _, expl in result["models"].items():
            for w in expl.get("top_words", []):
                all_misinfo_words.add(w["word"])
            for w in expl.get("misinfo_words", []):
                all_misinfo_words.add(w["word"])
            for w in expl.get("credible_words", []):
                all_credible_words.add(w["word"])

        result["summary"] = {
            "misinfo_indicators": list(all_misinfo_words)[:top_n],
            "credible_indicators": list(all_credible_words)[:top_n],
        }

        return result


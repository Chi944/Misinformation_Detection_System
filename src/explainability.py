"""
Explainability module for misinformation detection models.
Provides LIME explanations and attention-based saliency mapping.
"""

from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lime.lime_text import LimeTextExplainer

from src.config import RESULTS_DIR, LABEL_NAMES


class ModelExplainer:
    """Explainability tools for misinformation detection models."""

    def __init__(self, class_names: List[str] = None):
        self.class_names = class_names or list(LABEL_NAMES.values())
        self.lime_explainer = LimeTextExplainer(
            class_names=self.class_names, split_expression=r"\W+", bow=True
        )

    def explain_with_lime(
        self, text: str, predict_fn, num_features: int = 10, num_samples: int = 500
    ) -> Dict[str, Any]:
        """
        Generate LIME explanation for a prediction.

        Args:
            text: Input text to explain
            predict_fn: Prediction function that returns probabilities
            num_features: Number of top features to include
            num_samples: Number of perturbed samples for LIME

        Returns:
            Dictionary with explanation details
        """
        explanation = self.lime_explainer.explain_instance(
            text, predict_fn, num_features=num_features, num_samples=num_samples
        )

        # Get prediction
        probs = predict_fn([text])[0]
        predicted_class = np.argmax(probs)

        # Extract feature weights
        feature_weights = explanation.as_list(label=predicted_class)

        return {
            "text": text,
            "predicted_class": self.class_names[predicted_class],
            "confidence": float(probs[predicted_class]),
            "probabilities": {self.class_names[i]: float(p) for i, p in enumerate(probs)},
            "feature_weights": feature_weights,
            "explanation": explanation,
        }

    def visualize_lime_explanation(
        self, explanation_result: Dict[str, Any], save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Visualize LIME explanation as a bar chart.

        Args:
            explanation_result: Result from explain_with_lime
            save_path: Path to save the figure

        Returns:
            Matplotlib figure
        """
        feature_weights = explanation_result["feature_weights"]

        # Separate positive and negative weights
        words = [fw[0] for fw in feature_weights]
        weights = [fw[1] for fw in feature_weights]

        colors = ["green" if w > 0 else "red" for w in weights]

        fig, ax = plt.subplots(figsize=(10, 6))

        y_pos = np.arange(len(words))
        ax.barh(y_pos, weights, color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(words)
        ax.invert_yaxis()
        ax.set_xlabel("Feature Importance")
        ax.set_title(
            f"LIME Explanation\nPredicted: {explanation_result['predicted_class']} "
            f"(Confidence: {explanation_result['confidence']:.2%})"
        )

        # Add legend
        ax.axvline(x=0, color="black", linestyle="-", linewidth=0.5)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved LIME visualization to {save_path}")

        return fig

    def highlight_text_lime(self, explanation_result: Dict[str, Any]) -> str:
        """
        Create HTML highlighted text based on LIME explanation.

        Args:
            explanation_result: Result from explain_with_lime

        Returns:
            HTML string with highlighted text
        """
        text = explanation_result["text"]
        feature_weights = dict(explanation_result["feature_weights"])

        words = text.split()
        highlighted_words = []

        for word in words:
            clean_word = "".join(c for c in word.lower() if c.isalnum())

            if clean_word in feature_weights:
                weight = feature_weights[clean_word]
                if weight > 0:
                    color = f"rgba(255, 0, 0, {min(abs(weight) * 2, 0.8)})"
                else:
                    color = f"rgba(0, 128, 0, {min(abs(weight) * 2, 0.8)})"
                highlighted_words.append(f'<span style="background-color: {color}">{word}</span>')
            else:
                highlighted_words.append(word)

        return " ".join(highlighted_words)

    def explain_attention(self, bert_model, text: str, top_k: int = 20) -> Dict[str, Any]:
        """
        Get attention-based explanation from BERT model.

        Args:
            bert_model: Trained BertClassifier instance
            text: Input text to explain
            top_k: Number of top attended tokens to highlight

        Returns:
            Dictionary with attention explanation
        """
        attention_info = bert_model.get_attention_weights(text)

        tokens = attention_info["tokens"]
        cls_attention = attention_info["cls_attention"]

        # Filter out special tokens and padding
        valid_indices = []
        valid_tokens = []
        valid_attention = []

        for i, (token, attn) in enumerate(zip(tokens, cls_attention)):
            if token not in ["[CLS]", "[SEP]", "[PAD]"] and not token.startswith("##"):
                valid_indices.append(i)
                valid_tokens.append(token)
                valid_attention.append(attn)

        # Normalize attention
        valid_attention = np.array(valid_attention)
        if valid_attention.sum() > 0:
            valid_attention = valid_attention / valid_attention.sum()

        # Get top-k attended tokens
        top_indices = np.argsort(valid_attention)[-top_k:][::-1]
        top_tokens = [(valid_tokens[i], float(valid_attention[i])) for i in top_indices]

        return {
            "text": text,
            "tokens": valid_tokens,
            "attention_scores": valid_attention.tolist(),
            "top_attended_tokens": top_tokens,
        }

    def visualize_attention(
        self, attention_result: Dict[str, Any], save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Visualize attention scores.

        Args:
            attention_result: Result from explain_attention
            save_path: Path to save the figure

        Returns:
            Matplotlib figure
        """
        top_tokens = attention_result["top_attended_tokens"]

        words = [t[0] for t in top_tokens[:15]]  # Show top 15
        scores = [t[1] for t in top_tokens[:15]]

        fig, ax = plt.subplots(figsize=(10, 6))

        y_pos = np.arange(len(words))
        ax.barh(y_pos, scores, color="steelblue")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(words)
        ax.invert_yaxis()
        ax.set_xlabel("Attention Score")
        ax.set_title("Top Attended Tokens (BERT Attention)")

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved attention visualization to {save_path}")

        return fig

    def get_feature_importance_tfidf(
        self, model, top_n: int = 20
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Get feature importance from TF-IDF + Logistic Regression model.

        Args:
            model: TfidfLogisticClassifier instance
            top_n: Number of top features per class

        Returns:
            Dictionary with feature importance per class
        """
        fi = model.get_feature_importance(top_n=top_n)

        return {
            "misinformation_indicators": list(
                zip(fi["misinformation_features"].tolist(), fi["misinformation_scores"].tolist())
            ),
            "credible_indicators": list(
                zip(fi["credible_features"].tolist(), fi["credible_scores"].tolist())
            ),
        }

    def visualize_feature_importance(
        self, feature_importance: Dict[str, Any], save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Visualize TF-IDF feature importance.

        Args:
            feature_importance: Result from get_feature_importance_tfidf
            save_path: Path to save the figure

        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Misinformation indicators
        misinfo = feature_importance["misinformation_indicators"][:10]
        words_m = [f[0] for f in misinfo]
        scores_m = [f[1] for f in misinfo]

        axes[0].barh(range(len(words_m)), scores_m, color="red", alpha=0.7)
        axes[0].set_yticks(range(len(words_m)))
        axes[0].set_yticklabels(words_m)
        axes[0].invert_yaxis()
        axes[0].set_xlabel("Coefficient")
        axes[0].set_title("Top Misinformation Indicators")

        # Credible indicators
        credible = feature_importance["credible_indicators"][:10]
        words_c = [f[0] for f in credible]
        scores_c = [abs(f[1]) for f in credible]  # Use absolute values for display

        axes[1].barh(range(len(words_c)), scores_c, color="green", alpha=0.7)
        axes[1].set_yticks(range(len(words_c)))
        axes[1].set_yticklabels(words_c)
        axes[1].invert_yaxis()
        axes[1].set_xlabel("Coefficient (absolute)")
        axes[1].set_title("Top Credible Indicators")

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved feature importance visualization to {save_path}")

        return fig

    def generate_explanation_report(
        self, text: str, traditional_model, bert_model=None, save_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive explanation report for a text.

        Args:
            text: Input text to explain
            traditional_model: Traditional ML model (e.g., TfidfLogisticClassifier)
            bert_model: Optional BERT model for attention-based explanation
            save_dir: Directory to save visualizations

        Returns:
            Dictionary with all explanations
        """
        save_dir = save_dir or RESULTS_DIR
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)

        report = {"text": text}

        # LIME explanation for traditional model
        print("Generating LIME explanation...")
        lime_result = self.explain_with_lime(
            text, lambda x: traditional_model.predict_proba(pd.Series(x))
        )
        report["lime_explanation"] = lime_result

        self.visualize_lime_explanation(lime_result, save_path=save_dir / "lime_explanation.png")

        # Attention explanation for BERT
        if bert_model is not None and bert_model.is_trained:
            print("Generating attention explanation...")
            attention_result = self.explain_attention(bert_model, text)
            report["attention_explanation"] = attention_result

            self.visualize_attention(
                attention_result, save_path=save_dir / "attention_explanation.png"
            )

        # Feature importance
        if hasattr(traditional_model, "get_feature_importance"):
            print("Extracting feature importance...")
            fi = self.get_feature_importance_tfidf(traditional_model)
            report["feature_importance"] = fi

            self.visualize_feature_importance(fi, save_path=save_dir / "feature_importance.png")

        return report


def explain_prediction(text: str, model, model_type: str = "traditional") -> Dict[str, Any]:
    """
    Quick function to explain a single prediction.

    Args:
        text: Input text
        model: Trained model
        model_type: 'traditional' or 'bert'

    Returns:
        Explanation dictionary
    """
    explainer = ModelExplainer()

    if model_type == "traditional":
        result = explainer.explain_with_lime(text, lambda x: model.predict_proba(pd.Series(x)))
    else:
        result = explainer.explain_attention(model, text)

    return result


if __name__ == "__main__":
    from src.data_preprocessing import prepare_data
    from src.traditional_ml import TfidfLogisticClassifier

    print("Preparing data...")
    train_df, val_df, test_df = prepare_data()

    print("\nTraining model...")
    model = TfidfLogisticClassifier()
    model.fit(train_df["combined_text"], train_df["label"])

    print("\nGenerating explanations...")
    explainer = ModelExplainer()

    # Test with a sample
    sample_text = test_df.iloc[0]["combined_text"]
    print(f"\nSample text: {sample_text[:200]}...")

    # LIME explanation
    lime_result = explainer.explain_with_lime(
        sample_text, lambda x: model.predict_proba(pd.Series(x))
    )

    print(f"\nPrediction: {lime_result['predicted_class']}")
    print(f"Confidence: {lime_result['confidence']:.2%}")
    print("\nTop features:")
    for word, weight in lime_result["feature_weights"][:5]:
        print(f"  {word}: {weight:.4f}")

    # Feature importance
    fi = explainer.get_feature_importance_tfidf(model)
    print("\nGlobal Misinformation Indicators:")
    for word, score in fi["misinformation_indicators"][:5]:
        print(f"  {word}: {score:.4f}")

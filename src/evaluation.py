"""
Evaluation module for misinformation detection models.
Provides comprehensive metrics, visualizations, and model comparison.
"""

import json
from typing import Dict, Any, List, Optional
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)

from src.config import RESULTS_DIR, LABEL_NAMES


class ModelEvaluator:
    """Comprehensive evaluation for misinformation detection models."""

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or RESULTS_DIR
        self.output_dir.mkdir(exist_ok=True)
        self.results = {}

    def compute_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Compute comprehensive evaluation metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Probability scores for positive class

        Returns:
            Dictionary with all metrics
        """
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision_macro": float(precision_score(y_true, y_pred, average="macro")),
            "recall_macro": float(recall_score(y_true, y_pred, average="macro")),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
            "precision_per_class": precision_score(y_true, y_pred, average=None).tolist(),
            "recall_per_class": recall_score(y_true, y_pred, average=None).tolist(),
            "f1_per_class": f1_score(y_true, y_pred, average=None).tolist(),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
            "support": np.bincount(y_true).tolist(),
        }

        if y_proba is not None:
            metrics["roc_auc"] = float(self._compute_roc_auc(y_true, y_proba))
            metrics["average_precision"] = float(average_precision_score(y_true, y_proba))

        return metrics

    def _compute_roc_auc(self, y_true: np.ndarray, y_proba: np.ndarray) -> float:
        """Compute ROC AUC score safely."""
        try:
            from sklearn.metrics import roc_auc_score

            return roc_auc_score(y_true, y_proba)
        except ValueError:
            return 0.0

    def evaluate_model(
        self,
        model_name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        latency: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate a single model and store results.

        Args:
            model_name: Name of the model
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Probability scores
            latency: Latency metrics

        Returns:
            Evaluation results
        """
        metrics = self.compute_metrics(y_true, y_pred, y_proba)

        if latency:
            metrics["latency"] = latency

        self.results[model_name] = metrics
        return metrics

    def compare_models(self) -> pd.DataFrame:
        """
        Create comparison table of all evaluated models.

        Returns:
            DataFrame with model comparison
        """
        if not self.results:
            raise ValueError("No models have been evaluated yet")

        comparison_data = []
        for model_name, metrics in self.results.items():
            row = {
                "Model": model_name,
                "Accuracy": metrics["accuracy"],
                "Precision": metrics["precision_macro"],
                "Recall": metrics["recall_macro"],
                "F1 Score": metrics["f1_macro"],
            }

            if "roc_auc" in metrics:
                row["ROC-AUC"] = metrics["roc_auc"]

            if "latency" in metrics:
                row["Mean Latency (ms)"] = metrics["latency"].get("mean_latency_ms", 0)

            comparison_data.append(row)

        df = pd.DataFrame(comparison_data)
        df = df.sort_values("F1 Score", ascending=False).reset_index(drop=True)

        return df

    def plot_confusion_matrix(self, model_name: str, save: bool = True) -> plt.Figure:
        """
        Plot confusion matrix for a model.

        Args:
            model_name: Name of the model
            save: Whether to save the figure

        Returns:
            Matplotlib figure
        """
        if model_name not in self.results:
            raise ValueError(f"Model {model_name} not found in results")

        cm = np.array(self.results[model_name]["confusion_matrix"])

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=list(LABEL_NAMES.values()),
            yticklabels=list(LABEL_NAMES.values()),
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix - {model_name}")

        plt.tight_layout()

        if save:
            path = self.output_dir / f"confusion_matrix_{model_name}.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"Saved confusion matrix to {path}")

        return fig

    def plot_roc_curves(
        self, y_true: np.ndarray, model_probas: Dict[str, np.ndarray], save: bool = True
    ) -> plt.Figure:
        """
        Plot ROC curves for multiple models.

        Args:
            y_true: True labels
            model_probas: Dictionary mapping model names to probability scores
            save: Whether to save the figure

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        for model_name, y_proba in model_probas.items():
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.3f})")

        ax.plot([0, 1], [0, 1], "k--", label="Random Classifier")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves Comparison")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            path = self.output_dir / "roc_curves_comparison.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"Saved ROC curves to {path}")

        return fig

    def plot_precision_recall_curves(
        self, y_true: np.ndarray, model_probas: Dict[str, np.ndarray], save: bool = True
    ) -> plt.Figure:
        """
        Plot Precision-Recall curves for multiple models.

        Args:
            y_true: True labels
            model_probas: Dictionary mapping model names to probability scores
            save: Whether to save the figure

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        for model_name, y_proba in model_probas.items():
            precision, recall, _ = precision_recall_curve(y_true, y_proba)
            ap = average_precision_score(y_true, y_proba)
            ax.plot(recall, precision, label=f"{model_name} (AP = {ap:.3f})")

        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curves Comparison")
        ax.legend(loc="lower left")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            path = self.output_dir / "pr_curves_comparison.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"Saved PR curves to {path}")

        return fig

    def plot_metrics_comparison(self, save: bool = True) -> plt.Figure:
        """
        Plot bar chart comparing models across metrics.

        Args:
            save: Whether to save the figure

        Returns:
            Matplotlib figure
        """
        df = self.compare_models()

        metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
        available_metrics = [m for m in metrics if m in df.columns]

        x = np.arange(len(df))
        width = 0.2

        fig, ax = plt.subplots(figsize=(12, 6))

        for i, metric in enumerate(available_metrics):
            offset = (i - len(available_metrics) / 2) * width
            bars = ax.bar(x + offset, df[metric], width, label=metric)

        ax.set_xlabel("Model")
        ax.set_ylabel("Score")
        ax.set_title("Model Performance Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels(df["Model"], rotation=45, ha="right")
        ax.legend()
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()

        if save:
            path = self.output_dir / "metrics_comparison.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"Saved metrics comparison to {path}")

        return fig

    def plot_latency_comparison(self, save: bool = True) -> plt.Figure:
        """
        Plot latency comparison across models.

        Args:
            save: Whether to save the figure

        Returns:
            Matplotlib figure
        """
        latency_data = []
        for model_name, metrics in self.results.items():
            if "latency" in metrics:
                latency_data.append(
                    {
                        "Model": model_name,
                        "Mean": metrics["latency"].get("mean_latency_ms", 0),
                        "Std": metrics["latency"].get("std_latency_ms", 0),
                    }
                )

        if not latency_data:
            print("No latency data available")
            return None

        df = pd.DataFrame(latency_data)

        fig, ax = plt.subplots(figsize=(10, 6))

        bars = ax.bar(df["Model"], df["Mean"], yerr=df["Std"], capsize=5)

        ax.axhline(y=500, color="r", linestyle="--", label="500ms threshold")

        ax.set_xlabel("Model")
        ax.set_ylabel("Latency (ms)")
        ax.set_title("Model Inference Latency Comparison")
        ax.legend()

        for bar, val in zip(bars, df["Mean"]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + df["Std"].max() * 0.1,
                f"{val:.1f}ms",
                ha="center",
                va="bottom",
            )

        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        if save:
            path = self.output_dir / "latency_comparison.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"Saved latency comparison to {path}")

        return fig

    def generate_report(self, save: bool = True) -> str:
        """
        Generate a text report summarizing all evaluations.

        Args:
            save: Whether to save the report

        Returns:
            Report string
        """
        report_lines = ["=" * 60, "MISINFORMATION DETECTION MODEL EVALUATION REPORT", "=" * 60, ""]

        # Model comparison table
        df = self.compare_models()
        report_lines.append("MODEL COMPARISON:")
        report_lines.append("-" * 40)
        report_lines.append(df.to_string(index=False))
        report_lines.append("")

        # Best model
        best_model = df.iloc[0]["Model"]
        best_f1 = df.iloc[0]["F1 Score"]
        report_lines.append(f"Best Model: {best_model} (F1 Score: {best_f1:.4f})")
        report_lines.append("")

        # Detailed results per model
        report_lines.append("DETAILED RESULTS PER MODEL:")
        report_lines.append("-" * 40)

        for model_name, metrics in self.results.items():
            report_lines.append(f"\n{model_name}:")
            report_lines.append(f"  Accuracy: {metrics['accuracy']:.4f}")
            report_lines.append(f"  Precision (macro): {metrics['precision_macro']:.4f}")
            report_lines.append(f"  Recall (macro): {metrics['recall_macro']:.4f}")
            report_lines.append(f"  F1 Score (macro): {metrics['f1_macro']:.4f}")

            if "roc_auc" in metrics:
                report_lines.append(f"  ROC-AUC: {metrics['roc_auc']:.4f}")

            if "latency" in metrics:
                lat = metrics["latency"]
                report_lines.append(f"  Mean Latency: {lat.get('mean_latency_ms', 0):.2f}ms")

            report_lines.append("\n  Confusion Matrix:")
            cm = np.array(metrics["confusion_matrix"])
            report_lines.append(f"    {cm}")

        report_lines.append("")
        report_lines.append("=" * 60)

        report = "\n".join(report_lines)

        if save:
            path = self.output_dir / "evaluation_report.txt"
            with open(path, "w") as f:
                f.write(report)
            print(f"Saved evaluation report to {path}")

        return report

    def save_results(self, filename: str = "evaluation_results.json"):
        """Save all results to JSON file."""
        path = self.output_dir / filename

        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for model_name, metrics in self.results.items():
            serializable_results[model_name] = {}
            for key, value in metrics.items():
                if isinstance(value, np.ndarray):
                    serializable_results[model_name][key] = value.tolist()
                else:
                    serializable_results[model_name][key] = value

        with open(path, "w") as f:
            json.dump(serializable_results, f, indent=2)

        print(f"Saved results to {path}")
        return path


def evaluate_all_models(
    test_df: pd.DataFrame,
    models: Dict[str, Any],
    text_column: str = "combined_text",
    label_column: str = "label",
) -> ModelEvaluator:
    """
    Evaluate all models on test data.

    Args:
        test_df: Test DataFrame
        models: Dictionary of trained models
        text_column: Column with text data
        label_column: Column with labels

    Returns:
        ModelEvaluator with all results
    """
    evaluator = ModelEvaluator()

    X_test = test_df[text_column]
    y_true = test_df[label_column].values

    model_probas = {}

    for model_name, model_info in models.items():
        print(f"\nEvaluating {model_name}...")

        model = model_info["model"]

        # Get predictions
        if hasattr(model, "predict_batch"):
            y_pred, y_proba = model.predict_batch(X_test.tolist())
            y_proba_pos = y_proba[:, 1]
        else:
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)
            y_proba_pos = y_proba[:, 1]

        # Get latency if available
        latency = model_info.get("latency")

        # Evaluate
        evaluator.evaluate_model(model_name, y_true, y_pred, y_proba_pos, latency)
        model_probas[model_name] = y_proba_pos

    # Generate visualizations
    print("\nGenerating visualizations...")
    evaluator.plot_metrics_comparison()
    evaluator.plot_latency_comparison()

    if model_probas:
        evaluator.plot_roc_curves(y_true, model_probas)
        evaluator.plot_precision_recall_curves(y_true, model_probas)

    for model_name in models.keys():
        evaluator.plot_confusion_matrix(model_name)

    # Generate report
    report = evaluator.generate_report()
    print("\n" + report)

    # Save results
    evaluator.save_results()

    return evaluator


if __name__ == "__main__":
    from src.data_preprocessing import prepare_data
    from src.traditional_ml import train_single_model, TfidfLogisticClassifier
    from src.config import MODELS_DIR

    print("Preparing data...")
    train_df, val_df, test_df = prepare_data()
    print("\nTraining TF-IDF + Logistic Regression...")
    train_single_model(train_df, val_df)
    tfidf_path = MODELS_DIR / "tfidf_logistic.pkl"
    tfidf_lr = TfidfLogisticClassifier()
    tfidf_lr.load(tfidf_path)
    X_test = test_df["combined_text"]
    y_true = test_df["label"].values
    y_pred = tfidf_lr.predict(X_test)
    y_proba = tfidf_lr.predict_proba(X_test)[:, 1]
    latency = tfidf_lr.measure_inference_latency(X_test)
    evaluator = ModelEvaluator()
    evaluator.evaluate_model("TF-IDF + LR", y_true, y_pred, y_proba, latency)
    print("\nComparison Table:")
    print(evaluator.compare_models())

import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
from src.utils.logger import get_logger  # noqa: E402


class EvaluationDashboard:
    """
    Generates evaluation_dashboard.png visual report.
    Part of EvaluationPipeline.

    Args:
        output_dir (str): directory for saved images
    """

    def __init__(self, output_dir="reports"):
        self.output_dir = output_dir
        self.logger = get_logger(__name__)
        os.makedirs(output_dir, exist_ok=True)

    def generate(self, eval_data, output_filename="evaluation_dashboard.png"):
        """
        Generate and save the full evaluation dashboard.

        Args:
            eval_data (dict): models dict plus optional judge_metrics
            output_filename (str): output PNG filename
        Returns:
            str: full path to saved image
        """
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(20, 24))
        fig.suptitle(
            "Misinformation Detector - Evaluation Dashboard", fontsize=16, fontweight="bold", y=0.98
        )
        self._plot_confusion_matrices(fig, eval_data)
        self._plot_roc_curves(fig, eval_data)
        self._plot_fuzzy_membership(fig)
        self._plot_judge_agreement(fig, eval_data)
        self._plot_calibration_curves(fig, eval_data)
        self._plot_agreement_heatmap(fig, eval_data)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        out_path = os.path.join(self.output_dir, output_filename)
        plt.savefig(out_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        self.logger.info("Dashboard saved to %s", out_path)
        return out_path

    def _plot_confusion_matrices(self, fig, eval_data):
        """4-panel confusion matrices, one per model + ensemble."""
        import seaborn as sns
        from sklearn.metrics import confusion_matrix

        for i, name in enumerate(["bert", "tfidf", "naive_bayes", "ensemble"]):
            ax = fig.add_subplot(6, 4, i + 1)
            data = eval_data.get("models", {}).get(name, {})
            if not data:
                ax.set_title("%s (no data)" % name)
                continue
            cm = confusion_matrix(data.get("y_true", [0, 1]), data.get("y_pred", [0, 1]))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                ax=ax,
                cbar=False,
                xticklabels=["Credible", "Misinfo"],
                yticklabels=["Credible", "Misinfo"],
            )
            ax.set_title("%s Confusion Matrix" % name.upper())
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")

    def _plot_roc_curves(self, fig, eval_data):
        """ROC curves for all 4 models on same axes."""
        from sklearn.metrics import auc, roc_curve

        ax = fig.add_subplot(6, 2, 3)
        colors = {"bert": "blue", "tfidf": "green", "naive_bayes": "orange", "ensemble": "red"}
        plotted = False
        for name, color in colors.items():
            data = eval_data.get("models", {}).get(name, {})
            if not data or "y_prob" not in data:
                continue
            try:
                fpr, tpr, _ = roc_curve(data["y_true"], data["y_prob"])
                auc_val = auc(fpr, tpr)
                ax.plot(
                    fpr, tpr, color=color, lw=2, label="%s (AUC=%.3f)" % (name.upper(), auc_val)
                )
                plotted = True
            except Exception:
                pass
        if not plotted:
            ax.text(0.5, 0.5, "No ROC data", ha="center", va="center", transform=ax.transAxes)
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves - All Models")
        ax.legend(loc="lower right", fontsize=8)

    def _plot_fuzzy_membership(self, fig):
        """Fuzzy output membership function visualization."""
        import skfuzzy as fuzz

        import src.utils.skfuzzy_compat  # noqa: F401

        ax = fig.add_subplot(6, 2, 4)
        universe = np.arange(0.0, 1.01, 0.01)
        ax.plot(
            universe, fuzz.trapmf(universe, [0.0, 0.0, 0.25, 0.40]), "b-", lw=2, label="Credible"
        )
        ax.plot(
            universe,
            fuzz.trapmf(universe, [0.30, 0.45, 0.55, 0.70]),
            "y-",
            lw=2,
            label="Suspicious",
        )
        ax.plot(
            universe,
            fuzz.trapmf(universe, [0.60, 0.75, 1.0, 1.0]),
            "r-",
            lw=2,
            label="Misinformation",
        )
        ax.set_title("Fuzzy Output Membership Functions")
        ax.set_xlabel("Misinfo Score")
        ax.set_ylabel("Membership Degree")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_judge_agreement(self, fig, eval_data):
        """Bar chart: model accuracy vs LLM judge agreement."""
        ax = fig.add_subplot(6, 2, 5)
        judge = eval_data.get("judge_metrics", {})
        model_names = ["bert", "tfidf", "naive_bayes", "ensemble"]
        agree_rates = [judge.get(m, {}).get("agreement_rate", 0.0) for m in model_names]
        accuracies = [
            eval_data.get("models", {}).get(m, {}).get("accuracy", 0.0) for m in model_names
        ]
        x = np.arange(len(model_names))
        w = 0.35
        ax.bar(x - w / 2, accuracies, w, label="Model Accuracy", color="steelblue", alpha=0.8)
        ax.bar(x + w / 2, agree_rates, w, label="Judge Agreement", color="coral", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([n.upper() for n in model_names])
        ax.set_ylim([0, 1.1])
        ax.axhline(y=0.75, color="green", linestyle="--", linewidth=1, label="75% gate")
        ax.set_title("Model Accuracy vs LLM Judge Agreement")
        ax.set_ylabel("Score")
        ax.legend(fontsize=8)

    def _plot_calibration_curves(self, fig, eval_data):
        """Confidence calibration curves for all models."""
        ax = fig.add_subplot(6, 2, 6)
        colors = {"bert": "blue", "tfidf": "green", "naive_bayes": "orange", "ensemble": "red"}
        plotted = False
        for name, color in colors.items():
            data = eval_data.get("models", {}).get(name, {})
            if not data or "y_prob" not in data:
                continue
            try:
                yt = np.array(data["y_true"])
                yp = np.array(data["y_prob"])
                bins = np.linspace(0, 1, 11)
                mp, fp = [], []
                for i in range(10):
                    mask = (yp >= bins[i]) & (yp < bins[i + 1])
                    if mask.sum() > 0:
                        mp.append(float(yp[mask].mean()))
                        fp.append(float(yt[mask].mean()))
                ax.plot(mp, fp, "s-", color=color, label=name.upper(), markersize=4)
                plotted = True
            except Exception:
                pass
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect")
        if not plotted:
            ax.text(
                0.5, 0.5, "No calibration data", ha="center", va="center", transform=ax.transAxes
            )
        ax.set_title("Confidence Calibration Curves")
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of Positives")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    def _plot_agreement_heatmap(self, fig, eval_data):
        """Heatmap of pairwise model prediction agreement."""
        import seaborn as sns

        ax = fig.add_subplot(6, 1, 6)
        models = eval_data.get("models", {})
        m_names = ["bert", "tfidf", "naive_bayes"]
        valid = {
            m: np.array(models[m]["y_pred"])
            for m in m_names
            if m in models and "y_pred" in models[m]
        }
        if len(valid) < 2:
            ax.text(
                0.5,
                0.5,
                "Insufficient data for heatmap",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Model Agreement Heatmap")
            return
        names = list(valid.keys())
        preds = np.stack([valid[m] for m in names], axis=1)
        matrix = np.array(
            [
                [float(np.mean(preds[:, i] == preds[:, j])) for j in range(len(names))]
                for i in range(len(names))
            ]
        )
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".2f",
            cmap="YlOrRd",
            ax=ax,
            xticklabels=[n.upper() for n in names],
            yticklabels=[n.upper() for n in names],
            vmin=0,
            vmax=1,
        )
        ax.set_title("Model Agreement Heatmap " "(fraction of samples where models agree)")

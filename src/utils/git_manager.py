"""Git automation helpers for the misinformation detector pipeline.

This module belongs to the *utils* component of the pipeline. It wraps common
git operations used by the feedback loop and training scripts.
"""

from __future__ import annotations

import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from src.utils.logger import get_logger

LOGGER = get_logger(__name__)


class GitManager:
    """Utility class for scripted git operations.

    Component:
        Utils / GitManager.
    """

    def __init__(self, repo_root: str | Path | None = None) -> None:
        self.repo_root = Path(repo_root or ".").resolve()

    # ----------------------------------------------------------------- internals
    def _run(self, *args: str) -> None:
        """Run a git command, logging but not raising on failure."""

        try:
            subprocess.run(
                ["git", *args],
                cwd=self.repo_root,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning("Git command failed: git %s (%s)", " ".join(args), exc)

    def _append_changelog(self, text: str) -> None:
        path = self.repo_root / "CHANGELOG.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        header = f"\n\n### Automated update — {stamp}\n"
        with path.open("a", encoding="utf-8") as f:
            f.write(header)
            f.write(text.rstrip() + "\n")

    # ----------------------------------------------------- public commit helpers
    def commit_cycle_results(self, cycle_num: int, metrics: Dict[str, Any]) -> None:
        """Commit feedback cycle artefacts and update CHANGELOG.

        Args:
            cycle_num: Feedback cycle number.
            metrics: Dictionary with per-model metrics (F1, accuracy, etc.).
        """

        try:
            ens_f1 = float(metrics.get("ensemble_f1", 0.0))
            acc = float(metrics.get("ensemble_acc", ens_f1))
        except Exception:
            ens_f1 = float(metrics.get("ensemble_f1", 0.0))
            acc = ens_f1

        body = (
            f"- Feedback cycle #{cycle_num}: "
            f"ensemble F1={ens_f1:.4f}, acc={acc:.4f}, "
            f"bert_f1={metrics.get('bert_f1')}, "
            f"tfidf_f1={metrics.get('tfidf_f1')}, "
            f"nb_f1={metrics.get('nb_f1')}"
        )
        self._append_changelog(body)

        # Stage and commit selected artefacts.
        self._run("add", "reports/evaluation_report.json", "reports/evaluation_dashboard.png")
        self._run("add", "config.yaml", "CHANGELOG.md")
        msg = f"feat: feedback cycle #{cycle_num} — ensemble F1={ens_f1:.4f} acc={acc:.4f}"
        self._run("commit", "-m", msg)
        self._run("push")

    def commit_model_checkpoint(self, model_name: str, version: str, metrics: Dict[str, Any]) -> None:
        """Commit a model checkpoint entry to CHANGELOG and push.

        Args:
            model_name: Logical model name (e.g. ``bert``).
            version: Version string (e.g. timestamp).
            metrics: Metrics dict containing at least ``f1``.
        """

        f1 = float(metrics.get("f1", 0.0))
        body = (
            f"- Checkpoint {model_name} v{version}: "
            f"F1={f1:.4f}, metrics={metrics}"
        )
        self._append_changelog(body)

        config_path = self.repo_root / "models" / f"{model_name}_config.json"
        if config_path.exists():
            self._run("add", "CHANGELOG.md", str(config_path.relative_to(self.repo_root)))
        else:
            self._run("add", "CHANGELOG.md")
        msg = f"chore: checkpoint {model_name} v{version} — F1={f1:.4f}"
        self._run("commit", "-m", msg)
        self._run("push")


import json
import os
import re
from typing import Optional, Tuple

from src.utils.logger import get_logger


class DomainCredibility:
    """Source credibility scoring based on a domain reputation database.

    Scores are in [0.0, 1.0], where 1.0 means highly credible and 0.0 means
    highly unreliable.
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        self.logger = get_logger(__name__)
        self.db_path = db_path or os.path.join("data", "domain_reputation.json")
        self.db: dict = {}
        self._load()

    def _load(self) -> None:
        """Load domain reputation database from disk."""
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, encoding="utf-8") as f:
                    self.db = json.load(f)
                self.logger.info(
                    "Domain credibility DB loaded: %d domains", len(self.db)
                )
            except Exception as e:  # pragma: no cover - defensive
                self.logger.warning("Could not load domain DB: %s", e)
        else:
            self.logger.warning("Domain DB not found at %s", self.db_path)

    def extract_domain(self, text: Optional[str]) -> Optional[str]:
        """Extract domain from a URL or text string."""
        if not text:
            return None

        s = str(text)

        # Try to extract from URL
        url_pattern = r"https?://(?:www\.)?([a-zA-Z0-9.-]+)"
        match = re.search(url_pattern, s)
        if match:
            return match.group(1).lower()

        # Try bare domain pattern
        domain_pattern = (
            r"\b([a-zA-Z0-9-]+\.[a-zA-Z]{2,}(?:\.[a-zA-Z]{2,})?)\b"
        )
        match = re.search(domain_pattern, s)
        if match:
            candidate = match.group(1).lower()
            if "." in candidate and len(candidate) > 4:
                return candidate

        return None

    def _lookup_score(self, domain: str) -> Optional[float]:
        """Lookup score with some normalization heuristics."""
        if not domain:
            return None

        # Exact match
        if domain in self.db:
            return float(self.db[domain])

        # Try with www prefix removed
        if domain.startswith("www."):
            bare = domain[4:]
            if bare in self.db:
                return float(self.db[bare])

        # Try parent domain (e.g. news.bbc.com -> bbc.com)
        parts = domain.split(".")
        if len(parts) > 2:
            parent = ".".join(parts[-2:])
            if parent in self.db:
                return float(self.db[parent])

        return None

    def get_score(self, url_or_domain: Optional[str]) -> float:
        """Return credibility score for a domain (0.0 to 1.0).

        Returns 0.5 if domain not found (neutral).
        """
        if not url_or_domain:
            return 0.5

        domain = self.extract_domain(url_or_domain)
        if not domain:
            # Treat input directly as a domain if it looks like one.
            domain = str(url_or_domain).lower().strip()

        score = self._lookup_score(domain)
        return 0.5 if score is None else score

    def adjust_probability(self, base_prob: float, url_or_domain: Optional[str]) -> float:
        """Adjust ensemble probability based on source credibility.

        - If domain score >= 0.75: pull base_prob toward credible (lower misinfo)
        - If domain score <= 0.25: pull base_prob toward misinfo (higher misinfo)
        - If domain score is neutral/unknown (0.5): return base_prob unchanged
        """
        score = self.get_score(url_or_domain)
        if abs(score - 0.5) < 1e-9:
            return float(base_prob)

        base_prob = float(base_prob)
        base_prob = max(0.0, min(1.0, base_prob))

        # High credibility source: nudge toward credible (lower misinfo prob)
        if score >= 0.75:
            adjustment = (score - 0.75) / 0.25 * 0.10
            adjusted = base_prob - adjustment
            return float(max(0.0, min(1.0, adjusted)))

        # Low credibility source: nudge toward misinfo (higher misinfo prob)
        if score <= 0.25:
            adjustment = (0.25 - score) / 0.25 * 0.10
            adjusted = base_prob + adjustment
            return float(max(0.0, min(1.0, adjusted)))

        return float(base_prob)

    def get_label(self, score: float) -> str:
        """Return human readable label for a credibility score."""
        score = float(score)
        if score >= 0.85:
            return "highly credible"
        if score >= 0.70:
            return "credible"
        if score >= 0.50:
            return "mixed reliability"
        if score >= 0.30:
            return "questionable"
        return "unreliable"


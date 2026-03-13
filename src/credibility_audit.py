"""
Comprehensive credibility audit module.
Computes: Sensationalism, Political Bias, Source Credibility (NewsGuard-style DB),
Factuality Index, Flagged Terms, and Lexical Diversity (vocabulary richness, language complexity).
"""

import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
from urllib.parse import urlparse

from src.config import DATA_DIR

# Sensational / clickbait-style terms (push toward misinformation)
SENSATIONAL_WORDS = {
    "breaking",
    "shocking",
    "exposed",
    "secret",
    "bombshell",
    "urgent",
    "alert",
    "revealed",
    "truth",
    "lying",
    "cover-up",
    "censored",
    "won't believe",
    "must read",
    "you won't believe",
    "they don't want",
    "wake up",
    "conspiracy",
    "proof",
    "insider",
    "mainstream media",
    "bombshell",
    "devastating",
    "explosive",
    "stunning",
    "outrageous",
}

# Simplified political bias indicators (illustrative; not comprehensive)
BIAS_LEFT_INDICATORS = {"democrat", "progressive", "liberal", "leftist"}
BIAS_RIGHT_INDICATORS = {"republican", "conservative", "alt-right", "trump"}

# Sample domain credibility (extend for production)
TRUSTED_DOMAINS = {
    "reuters.com",
    "apnews.com",
    "bbc.com",
    "npr.org",
    "pbs.org",
    "nytimes.com",
    "washingtonpost.com",
    "theguardian.com",
    "nature.com",
    "science.org",
    "gov",
    "edu",
}
UNTRUSTED_DOMAINS = {
    "blogspot.com",
    "wordpress.com",
    "tumblr.com",  # generic blogs
}

# NewsGuard-style domain reputation database (loaded from data/domain_reputation.json)
_domain_reputation_db: Optional[Dict[str, Dict[str, Any]]] = None


def _load_domain_reputation_db() -> Dict[str, Dict[str, Any]]:
    """Load NewsGuard-style domain reputation database for scoring."""
    global _domain_reputation_db
    if _domain_reputation_db is not None:
        return _domain_reputation_db
    path = DATA_DIR / "domain_reputation.json"
    if not path.exists():
        _domain_reputation_db = {}
        return _domain_reputation_db
    try:
        with open(path, "r", encoding="utf-8") as f:
            _domain_reputation_db = json.load(f)
    except Exception:
        _domain_reputation_db = {}
    return _domain_reputation_db


def compute_lexical_diversity(text: str) -> Dict[str, Any]:
    """
    Lexical diversity analysis: vocabulary richness and language complexity metrics.
    Returns type-token ratio (TTR), unique/total word counts, avg word length, avg sentence length.
    """
    if not text or not str(text).strip():
        return {
            "type_token_ratio": 0.0,
            "unique_words": 0,
            "total_words": 0,
            "avg_word_length": 0.0,
            "avg_sentence_length": 0.0,
            "sentence_count": 0,
        }
    t = str(text).strip()
    words = re.findall(r"\w+", t.lower())
    total_words = len(words)
    unique_words = len(set(words))
    type_token_ratio = round(unique_words / total_words, 4) if total_words else 0.0
    # Average word length (characters)
    char_count = sum(len(w) for w in words)
    avg_word_length = round(char_count / total_words, 2) if total_words else 0.0
    # Sentences: split on . ! ?
    sentences = [s.strip() for s in re.split(r"[.!?]+", t) if s.strip()]
    sentence_count = len(sentences)
    sentence_lengths = [len(re.findall(r"\w+", s)) for s in sentences]
    avg_sentence_length = (
        round(sum(sentence_lengths) / sentence_count, 2) if sentence_count else 0.0
    )
    return {
        "type_token_ratio": type_token_ratio,
        "unique_words": unique_words,
        "total_words": total_words,
        "avg_word_length": avg_word_length,
        "avg_sentence_length": avg_sentence_length,
        "sentence_count": sentence_count,
    }


def compute_sensationalism(text: str) -> float:
    """
    Estimate sensationalism score 0–1 (higher = more sensational).
    Uses caps ratio, exclamation marks, and sensational word density.
    """
    if not text or not text.strip():
        return 0.0
    t = text.lower()
    words = re.findall(r"\w+", t)
    n = len(words)
    if n == 0:
        return 0.0

    # Caps ratio
    caps = sum(1 for c in text if c.isupper())
    caps_ratio = min(1.0, caps / max(1, len(text) * 0.3))

    # Exclamation density
    excl = text.count("!") + text.count("?")
    excl_score = min(1.0, excl / max(1, n) * 5)

    # Sensational word density
    hit = sum(1 for w in words if w in SENSATIONAL_WORDS)
    word_score = min(1.0, hit / max(1, n) * 10)

    # Combined (weighted)
    score = 0.2 * caps_ratio + 0.3 * excl_score + 0.5 * word_score
    return round(min(1.0, score), 4)


def compute_political_bias(text: str) -> Dict[str, Any]:
    """
    Simple keyword-based political bias estimate.
    Returns direction (left/right/neutral) and strength 0–1.
    """
    if not text or not text.strip():
        return {"direction": "neutral", "score": 0.0, "confidence": 0.0}
    t = text.lower()
    words = set(re.findall(r"\w+", t))

    left = sum(1 for w in BIAS_LEFT_INDICATORS if w in words)
    right = sum(1 for w in BIAS_RIGHT_INDICATORS if w in words)

    if left > right:
        direction = "left"
        score = min(1.0, left / 3)
    elif right > left:
        direction = "right"
        score = min(1.0, right / 3)
    else:
        direction = "neutral"
        score = 0.0

    confidence = min(1.0, (left + right) / 4)
    return {
        "direction": direction,
        "score": round(score, 4),
        "confidence": round(confidence, 4),
    }


def compute_source_credibility(url: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    NewsGuard-style source credibility: domain reputation scoring via database lookup.
    Loads data/domain_reputation.json for domain -> score/tier; falls back to TRUSTED/UNTRUSTED lists.
    """
    if not url or not str(url).strip():
        return None
    try:
        parsed = urlparse(str(url))
        host = (parsed.netloc or parsed.path or "").lower()
        if not host:
            return None
        host = re.sub(r"^www\.", "", host)
        parts = host.split(".")
        domain = ".".join(parts[-2:]) if len(parts) >= 2 else host

        db = _load_domain_reputation_db()
        # Lookup: try full host then base domain
        for key in (host, domain):
            if key in db:
                entry = db[key]
                return {
                    "score": round(float(entry.get("score", 0.5)), 4),
                    "domain": host,
                    "tier": entry.get("tier", "unknown"),
                    "source": "reputation_db",
                }
        # Fallback to list-based scoring
        score = 0.5
        if any(d in host or domain.endswith(d) for d in TRUSTED_DOMAINS):
            score = 0.9
        elif any(d in host or domain.endswith(d) for d in UNTRUSTED_DOMAINS):
            score = 0.3
        return {
            "score": round(score, 4),
            "domain": host,
            "tier": "trusted" if score >= 0.8 else "unknown" if score >= 0.5 else "caution",
            "source": "list",
        }
    except Exception:
        return None


def compute_factuality_index(misinformation_prob: float) -> float:
    """
    Factuality index 0–1: higher = more factual.
    Derived from 1 - P(misinformation).
    """
    return round(1.0 - misinformation_prob, 4)


def extract_flagged_terms(
    text: str,
    explanation_features: Optional[List[tuple]] = None,
    sensational_words: bool = True,
    predicted_label: int = 1,
) -> List[Dict[str, Any]]:
    """
    Extract terms that contribute to non-credible classification.
    LIME returns weights for the predicted class:
    - predicted=Misinformation (1): positive weight = pushes toward misinfo = flagged
    - predicted=Credible (0): negative weight = hurts credibility = flagged
    """
    flagged = []
    seen = set()

    if explanation_features:
        for term, weight in explanation_features:
            # Terms that push toward misinformation
            is_flagged = (predicted_label == 1 and weight > 0) or (
                predicted_label == 0 and weight < 0
            )
            if is_flagged and term and term.lower() not in seen:
                seen.add(term.lower())
                flagged.append(
                    {
                        "term": term,
                        "weight": round(abs(float(weight)), 4),
                        "reason": "model_contribution",
                    }
                )

    if sensational_words:
        words = re.findall(r"\w+", text)
        for w in words:
            wl = w.lower()
            if wl in SENSATIONAL_WORDS and wl not in seen:
                seen.add(wl)
                flagged.append(
                    {
                        "term": w,
                        "weight": 0.5,
                        "reason": "sensational",
                    }
                )

    return flagged[:20]


def run_credibility_audit(
    text: str,
    url: Optional[str] = None,
    misinformation_prob: float = 0.0,
    explanation_features: Optional[List[tuple]] = None,
    predicted_label: int = 1,
) -> Dict[str, Any]:
    """
    Run full credibility audit and return metrics.
    """
    sensationalism = compute_sensationalism(text)
    political_bias = compute_political_bias(text)
    source_cred = compute_source_credibility(url)
    factuality = compute_factuality_index(misinformation_prob)
    flagged_terms = extract_flagged_terms(
        text,
        explanation_features=explanation_features,
        sensational_words=True,
        predicted_label=predicted_label,
    )
    lexical_diversity = compute_lexical_diversity(text)

    audit = {
        "sensationalism": sensationalism,
        "political_bias": political_bias,
        "source_credibility": (
            source_cred
            if source_cred
            else {
                "score": None,
                "domain": None,
                "tier": "n/a",
            }
        ),
        "factuality_index": factuality,
        "flagged_terms": flagged_terms,
        "lexical_diversity": lexical_diversity,
    }
    return audit

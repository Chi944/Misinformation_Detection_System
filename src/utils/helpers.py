import os
import re
import json
import hashlib
import unicodedata
from src.utils.logger import get_logger

logger = get_logger(__name__)


def clean_text(text, max_length=512):
    """
    Clean and normalise input text for model processing.

    Steps: unicode normalisation, remove control chars, collapse
    whitespace, strip, truncate to max_length words.

    Args:
        text (str): raw input text
        max_length (int): max word count. Default 512
    Returns:
        str: cleaned text
    """
    if not isinstance(text, str):
        text = str(text)
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()
    if len(words) > max_length:
        text = " ".join(words[:max_length])
    return text


def safe_divide(numerator, denominator, default=0.0):
    """
    Divide two numbers safely, returning default on zero division.

    Args:
        numerator (float): dividend
        denominator (float): divisor
        default (float): value to return if denominator is zero
    Returns:
        float: result or default
    """
    if denominator == 0 or denominator is None:
        return default
    return numerator / denominator


def clamp(value, min_val=0.0, max_val=1.0):
    """
    Clamp a numeric value to [min_val, max_val].

    Args:
        value (float): input value
        min_val (float): minimum bound
        max_val (float): maximum bound
    Returns:
        float: clamped value
    """
    return max(min_val, min(max_val, value))


def hash_text(text):
    """
    Return a short SHA-256 hash of text for deduplication.

    Args:
        text (str): input text
    Returns:
        str: 16-char hex digest
    """
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]


def load_json_safe(path, default=None):
    """
    Load a JSON file, returning default on any error.

    Args:
        path (str): file path
        default: value to return on error. Default None
    Returns:
        parsed JSON or default
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning("Could not load JSON from %s: %s", path, e)
        return default


def save_json_safe(data, path):
    """
    Save data to a JSON file safely, creating parent dirs as needed.

    Args:
        data: JSON-serialisable object
        path (str): destination file path
    Returns:
        bool: True on success, False on failure
    """
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        return True
    except Exception as e:
        logger.warning("Could not save JSON to %s: %s", path, e)
        return False


def truncate_text(text, max_chars=200, suffix="..."):
    """
    Truncate text to max_chars for display purposes.

    Args:
        text (str): input text
        max_chars (int): maximum character count
        suffix (str): appended when truncated
    Returns:
        str: possibly truncated text
    """
    if len(text) <= max_chars:
        return text
    return text[: max_chars - len(suffix)] + suffix


def label_to_str(label):
    """
    Convert integer label to human-readable string.

    Args:
        label (int or str): 0 or 1
    Returns:
        str: 'credible' or 'misinformation'
    """
    return "misinformation" if int(label) == 1 else "credible"


def str_to_label(label_str):
    """
    Convert label string to integer.

    Args:
        label_str (str): 'credible' or 'misinformation'
    Returns:
        int: 0 or 1
    """
    return 1 if str(label_str).lower() == "misinformation" else 0

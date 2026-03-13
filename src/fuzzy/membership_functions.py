"""Fuzzy membership function definitions for misinformation scoring.

This module belongs to the *fuzzy* component of the pipeline. It defines
the universes of discourse and membership functions for model confidences,
source credibility, model agreement, feedback score, and the final
misinformation score.

All antecedents are defined on [0, 1] with step 0.01 and share the same
triangular membership functions:

- low    = trimf [0.0, 0.0, 0.45]
- medium = trimf [0.35, 0.5, 0.65]
- high   = trimf [0.55, 1.0, 1.0]

The consequent `misinfo_score` is also on [0, 1] with:

- credible       = trapmf [0.0, 0.0, 0.25, 0.40]
- suspicious     = trapmf [0.30, 0.45, 0.55, 0.70]
- misinformation = trapmf [0.60, 0.75, 1.0, 1.0]
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

import src.utils.skfuzzy_compat  # noqa: F401


def _make_antecedent(name: str, universe: np.ndarray) -> ctrl.Antecedent:
    """Create an antecedent with low/medium/high triangular membership.

    Args:
        name: Name of the fuzzy variable.
        universe: Numpy array representing the universe of discourse.

    Returns:
        skfuzzy.control.Antecedent: Antecedent with ``low``, ``medium`` and
        ``high`` membership functions defined on ``universe``.
    """
    ant = ctrl.Antecedent(universe, name)
    ant["low"] = fuzz.trimf(universe, [0.0, 0.0, 0.45])
    ant["medium"] = fuzz.trimf(universe, [0.35, 0.5, 0.65])
    ant["high"] = fuzz.trimf(universe, [0.55, 1.0, 1.0])
    return ant


def _make_consequent(name: str, universe: np.ndarray) -> ctrl.Consequent:
    """Create the consequent ``misinfo_score`` with trapezoidal membership."""

    cons = ctrl.Consequent(universe, name, defuzzify_method="centroid")
    cons["credible"] = fuzz.trapmf(universe, [0.0, 0.0, 0.25, 0.40])
    cons["suspicious"] = fuzz.trapmf(universe, [0.30, 0.45, 0.55, 0.70])
    cons["misinformation"] = fuzz.trapmf(universe, [0.60, 0.75, 1.0, 1.0])
    return cons


def build_membership_functions() -> Dict[str, ctrl.ControlVariable]:
    """Build all fuzzy membership functions and return them in a dict.

    Returns:
        dict: Mapping from variable name to its
        :class:`skfuzzy.control.Antecedent` or
        :class:`skfuzzy.control.Consequent` instance.
    """
    # Universe for all inputs: [0, 1] with step 0.01
    u = np.arange(0.0, 1.0 + 0.01, 0.01)

    source_cred = _make_antecedent("source_credibility", u)
    bert_conf = _make_antecedent("bert_confidence", u)
    tfidf_conf = _make_antecedent("tfidf_confidence", u)
    nb_conf = _make_antecedent("nb_confidence", u)
    model_agree = _make_antecedent("model_agreement", u)
    feedback_score = _make_antecedent("feedback_score", u)

    misinfo_score = _make_consequent("misinfo_score", u)

    return {
        "source_credibility": source_cred,
        "bert_confidence": bert_conf,
        "tfidf_confidence": tfidf_conf,
        "nb_confidence": nb_conf,
        "model_agreement": model_agree,
        "feedback_score": feedback_score,
        "misinfo_score": misinfo_score,
    }

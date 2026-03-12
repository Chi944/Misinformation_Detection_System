# src/fuzzy/fuzzy_engine.py
"""Fuzzy logic engine for misinformation scoring.

This module belongs to the *fuzzy* component of the pipeline. It implements a
Mamdani fuzzy inference system with centroid defuzzification over:

- source_credibility
- bert_confidence
- tfidf_confidence
- nb_confidence
- model_agreement
- feedback_score

and produces a scalar misinfo_score in [0, 1].
"""

from __future__ import annotations

from typing import Dict, Any

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

from src.fuzzy.membership_functions import build_membership_functions


class FuzzyMisinformationEngine:
    """Mamdani fuzzy inference engine for misinformation scoring.

    This engine:

    - Loads antecedents and consequent from `build_membership_functions()`.
    - Defines 18 fuzzy rules (strong misinformation, strong credibility,
      suspicious/uncertain, feedback-adjusted).
    - Uses centroid defuzzification on `misinfo_score`.

    Attributes:
        threshold_suspicious: Float in [0,1] for suspicious cut-off.
        threshold_misinformation: Float in [0,1] for misinformation cut-off.

    Component:
        Fuzzy / Engine.
    """

    def __init__(self) -> None:
        terms = build_membership_functions()
        self.source_cred = terms["source_credibility"]
        self.bert_conf = terms["bert_confidence"]
        self.tfidf_conf = terms["tfidf_confidence"]
        self.nb_conf = terms["nb_confidence"]
        self.model_agree = terms["model_agreement"]
        self.feedback_score = terms["feedback_score"]
        self.misinfo_score = terms["misinfo_score"]

        self._control_system = self._build_control_system()
        self._sim = ctrl.ControlSystemSimulation(self._control_system)

        # Mutable thresholds for feedback loop
        self.threshold_suspicious: float = 0.45
        self.threshold_misinformation: float = 0.65

    # ---------------------------------------------------------------- rules
    def _build_control_system(self) -> ctrl.ControlSystem:
        """Define all 18 fuzzy rules and return a ControlSystem."""

        s_cred = self.source_cred
        b_conf = self.bert_conf
        t_conf = self.tfidf_conf
        n_conf = self.nb_conf
        m_agree = self.model_agree
        f_score = self.feedback_score
        mis = self.misinfo_score

        rules = []

        # STRONG MISINFORMATION
        # 1. source_cred LOW  AND bert_conf HIGH        → misinformation
        rules.append(
            ctrl.Rule(
                s_cred["low"] & b_conf["high"],
                mis["misinformation"],
            )
        )
        # 2. model_agree HIGH AND bert_conf HIGH AND tfidf_conf HIGH → misinformation
        rules.append(
            ctrl.Rule(
                m_agree["high"] & b_conf["high"] & t_conf["high"],
                mis["misinformation"],
            )
        )
        # 3. source_cred LOW  AND model_agree HIGH       → misinformation
        rules.append(
            ctrl.Rule(
                s_cred["low"] & m_agree["high"],
                mis["misinformation"],
            )
        )
        # 4. bert_conf HIGH   AND tfidf_conf HIGH        → misinformation
        rules.append(
            ctrl.Rule(
                b_conf["high"] & t_conf["high"],
                mis["misinformation"],
            )
        )
        # 5. feedback_score HIGH AND bert_conf HIGH      → misinformation
        rules.append(
            ctrl.Rule(
                f_score["high"] & b_conf["high"],
                mis["misinformation"],
            )
        )

        # STRONG CREDIBILITY
        # 6. source_cred HIGH AND bert_conf LOW          → credible
        rules.append(
            ctrl.Rule(
                s_cred["high"] & b_conf["low"],
                mis["credible"],
            )
        )
        # 7. source_cred HIGH AND model_agree HIGH AND nb_conf LOW → credible
        rules.append(
            ctrl.Rule(
                s_cred["high"] & m_agree["high"] & n_conf["low"],
                mis["credible"],
            )
        )
        # 8. source_cred HIGH AND tfidf_conf LOW         → credible
        rules.append(
            ctrl.Rule(
                s_cred["high"] & t_conf["low"],
                mis["credible"],
            )
        )
        # 9. feedback_score LOW AND source_cred HIGH     → credible
        rules.append(
            ctrl.Rule(
                f_score["low"] & s_cred["high"],
                mis["credible"],
            )
        )

        # SUSPICIOUS / UNCERTAIN
        # 10. model_agree LOW                            → suspicious
        rules.append(
            ctrl.Rule(
                m_agree["low"],
                mis["suspicious"],
            )
        )
        # 11. source_cred MEDIUM AND bert_conf MEDIUM    → suspicious
        rules.append(
            ctrl.Rule(
                s_cred["medium"] & b_conf["medium"],
                mis["suspicious"],
            )
        )
        # 12. bert_conf MEDIUM AND tfidf_conf MEDIUM     → suspicious
        rules.append(
            ctrl.Rule(
                b_conf["medium"] & t_conf["medium"],
                mis["suspicious"],
            )
        )
        # 13. nb_conf MEDIUM AND model_agree MEDIUM      → suspicious
        rules.append(
            ctrl.Rule(
                n_conf["medium"] & m_agree["medium"],
                mis["suspicious"],
            )
        )
        # 14. source_cred LOW AND bert_conf MEDIUM       → suspicious
        rules.append(
            ctrl.Rule(
                s_cred["low"] & b_conf["medium"],
                mis["suspicious"],
            )
        )
        # 15. feedback_score MEDIUM AND model_agree MEDIUM → suspicious
        rules.append(
            ctrl.Rule(
                f_score["medium"] & m_agree["medium"],
                mis["suspicious"],
            )
        )

        # FEEDBACK-ADJUSTED
        # 16. feedback_score HIGH AND source_cred MEDIUM → misinformation
        rules.append(
            ctrl.Rule(
                f_score["high"] & s_cred["medium"],
                mis["misinformation"],
            )
        )
        # 17. feedback_score LOW AND model_agree HIGH AND source_cred MEDIUM → credible
        rules.append(
            ctrl.Rule(
                f_score["low"] & m_agree["high"] & s_cred["medium"],
                mis["credible"],
            )
        )
        # 18. bert_conf LOW AND tfidf_conf LOW AND nb_conf LOW → credible
        rules.append(
            ctrl.Rule(
                b_conf["low"] & t_conf["low"] & n_conf["low"],
                mis["credible"],
            )
        )

        return ctrl.ControlSystem(rules)

    # ---------------------------------------------------------------- compute
    def compute(self, inputs: Dict[str, Any]) -> float:
        """Compute defuzzified misinfo_score from crisp inputs.

        Args:
            inputs: Dict with keys:
                - source_credibility
                - bert_confidence
                - tfidf_confidence
                - nb_confidence
                - model_agreement
                - feedback_score

        Returns:
            float: misinfo_score in [0.0, 1.0].
        """
        # Clip all inputs to [0, 1]
        def _clip(v: Any) -> float:
            try:
                x = float(v)
            except Exception:
                x = 0.0
            return max(0.0, min(1.0, x))

        vals = {
            "source_credibility": _clip(inputs.get("source_credibility", 0.5)),
            "bert_confidence": _clip(inputs.get("bert_confidence", 0.5)),
            "tfidf_confidence": _clip(inputs.get("tfidf_confidence", 0.5)),
            "nb_confidence": _clip(inputs.get("nb_confidence", 0.5)),
            "model_agreement": _clip(inputs.get("model_agreement", 0.5)),
            "feedback_score": _clip(inputs.get("feedback_score", 0.5)),
        }

        # Reset simulation and set inputs
        self._sim = ctrl.ControlSystemSimulation(self._control_system)
        self._sim.input["source_credibility"] = vals["source_credibility"]
        self._sim.input["bert_confidence"] = vals["bert_confidence"]
        self._sim.input["tfidf_confidence"] = vals["tfidf_confidence"]
        self._sim.input["nb_confidence"] = vals["nb_confidence"]
        self._sim.input["model_agreement"] = vals["model_agreement"]
        self._sim.input["feedback_score"] = vals["feedback_score"]

        self._sim.compute()
        score = float(self._sim.output["misinfo_score"])

        # Ensure numeric and clipped
        if np.isnan(score):  # pragma: no cover
            score = 0.5
        return max(0.0, min(1.0, score))
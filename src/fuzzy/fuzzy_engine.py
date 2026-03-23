from __future__ import annotations

import numpy as np
import skfuzzy as fuzz

import src.utils.skfuzzy_compat  # noqa: F401 — must be first, fixes Python 3.12
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)


class FuzzyMisinformationEngine:
    """
    Mamdani fuzzy inference engine for misinformation scoring.

    Implements 18 fuzzy rules across 6 input variables to produce a
    continuous misinformation score in [0.0, 1.0].

    Uses pure manual Mamdani inference (fuzz.interp_membership +
    fuzz.defuzz) instead of skfuzzy ControlSystem, which is unstable
    under Python 3.12 with scikit-fuzzy 0.4.2.

    Part of the ensemble prediction pipeline in MisinformationDetector.
    """

    UNIVERSE = np.arange(0.0, 1.01, 0.01)

    # Triangular MF parameters: [left, center, right]
    LOW_PARAMS = [0.0, 0.0, 0.45]
    MEDIUM_PARAMS = [0.35, 0.5, 0.65]
    HIGH_PARAMS = [0.55, 1.0, 1.0]

    # Trapezoidal output MF parameters: [a, b, c, d]
    CREDIBLE_PARAMS = [0.0, 0.0, 0.25, 0.40]
    SUSPICIOUS_PARAMS = [0.30, 0.45, 0.55, 0.70]
    MISINFORMATION_PARAMS = [0.60, 0.75, 1.0, 1.0]

    def __init__(self) -> None:
        """
        Initialise the fuzzy engine.

        Pre-computes all membership function arrays for performance and sets
        mutable threshold attributes used by the feedback loop.
        """
        u = self.UNIVERSE

        # Pre-compute input MFs
        self._low_mf = fuzz.trimf(u, self.LOW_PARAMS)
        self._medium_mf = fuzz.trimf(u, self.MEDIUM_PARAMS)
        self._high_mf = fuzz.trimf(u, self.HIGH_PARAMS)

        # Pre-compute output MFs
        self._credible_mf = fuzz.trapmf(u, self.CREDIBLE_PARAMS)
        self._suspicious_mf = fuzz.trapmf(u, self.SUSPICIOUS_PARAMS)
        self._misinformation_mf = fuzz.trapmf(u, self.MISINFORMATION_PARAMS)

        # Mutable thresholds — updated by feedback loop hill climbing
        self.threshold_suspicious: float = 0.45
        self.threshold_misinformation: float = 0.65

        LOGGER.info("FuzzyMisinformationEngine initialised (manual Mamdani)")

    def _membership(self, value: float, mf_array: np.ndarray) -> float:
        """
        Compute membership degree of a crisp value in a pre-computed MF.

        Args:
            value: Crisp input value, will be clipped to (0.001, 0.999).
            mf_array: Pre-computed membership function array.

        Returns:
            float: Membership degree in [0.0, 1.0].
        """
        clipped = float(np.clip(value, 0.001, 0.999))
        return float(fuzz.interp_membership(self.UNIVERSE, mf_array, clipped))

    def compute(self, inputs: dict) -> float:
        """
        Run manual Mamdani fuzzy inference and return a defuzzified score.

        Args:
            inputs: Dictionary with any of these keys (missing keys default to
                0.5):

                - source_credibility
                - bert_confidence
                - tfidf_confidence
                - nb_confidence
                - model_agreement
                - feedback_score

        Returns:
            float: Misinformation score in [0.0, 1.0]:

                - 0.0 → definitely credible
                - 1.0 → definitely misinformation
                - 0.5 → uncertain / fallback on error
        """
        try:
            return self._manual_compute(inputs)
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning("Fuzzy compute failed (%s), returning 0.5", exc)
            return 0.5

    # ------------------------------------------------------------------ API
    def evaluate(
        self,
        source_credibility: float,
        bert_confidence: float,
        tfidf_confidence: float,
        nb_confidence: float = 0.5,
        model_agreement: float = 0.5,
        feedback_score: float = 0.5,
    ) -> float:
        """
        Backwards-compatible wrapper expected by older tooling/tests.

        Args:
            source_credibility: credibility score in [0,1]
            bert_confidence: BERT confidence for misinfo in [0,1]
            tfidf_confidence: TF-IDF confidence for misinfo in [0,1]
            nb_confidence: NB confidence for misinfo (defaults to neutral 0.5)
            model_agreement: agreement score (defaults to neutral 0.5)
            feedback_score: feedback/error score (defaults to neutral 0.5)
        """
        return float(
            self.compute(
                {
                    "source_credibility": source_credibility,
                    "bert_confidence": bert_confidence,
                    "tfidf_confidence": tfidf_confidence,
                    "nb_confidence": nb_confidence,
                    "model_agreement": model_agreement,
                    "feedback_score": feedback_score,
                }
            )
        )

    def _manual_compute(self, inputs: dict) -> float:
        """
        Core manual Mamdani implementation.

        Evaluates all 18 rules and defuzzifies with the centroid method.

        Args:
            inputs: Dictionary of input variable values.

        Returns:
            float: Defuzzified misinfo_score in [0.0, 1.0].
        """
        # Extract inputs with neutral 0.5 default for missing keys
        s = float(inputs.get("source_credibility", 0.5))
        b = float(inputs.get("bert_confidence", 0.5))
        t = float(inputs.get("tfidf_confidence", 0.5))
        n = float(inputs.get("nb_confidence", 0.5))
        a = float(inputs.get("model_agreement", 0.5))
        f = float(inputs.get("feedback_score", 0.5))

        # Compute membership degrees for each input variable
        s_low = self._membership(s, self._low_mf)
        s_med = self._membership(s, self._medium_mf)
        s_high = self._membership(s, self._high_mf)

        b_low = self._membership(b, self._low_mf)
        b_med = self._membership(b, self._medium_mf)
        b_high = self._membership(b, self._high_mf)

        t_low = self._membership(t, self._low_mf)
        t_med = self._membership(t, self._medium_mf)
        t_high = self._membership(t, self._high_mf)

        n_low = self._membership(n, self._low_mf)
        n_med = self._membership(n, self._medium_mf)

        a_low = self._membership(a, self._low_mf)
        a_med = self._membership(a, self._medium_mf)
        a_high = self._membership(a, self._high_mf)

        f_low = self._membership(f, self._low_mf)
        f_med = self._membership(f, self._medium_mf)
        f_high = self._membership(f, self._high_mf)

        # STRONG MISINFORMATION RULES (1–5 + 16)
        misinfo_strength = max(
            min(s_low, b_high),  # 1
            min(a_high, b_high, t_high),  # 2
            min(s_low, a_high),  # 3
            min(b_high, t_high),  # 4
            min(f_high, b_high),  # 5
            min(f_high, s_med),  # 16
        )

        # STRONG CREDIBILITY RULES (6–9 + 17 + 18)
        credible_strength = max(
            min(s_high, b_low),  # 6
            min(s_high, a_high, n_low),  # 7
            min(s_high, t_low),  # 8
            min(f_low, s_high),  # 9
            min(f_low, a_high, s_med),  # 17
            min(b_low, t_low, n_low),  # 18
        )

        # SUSPICIOUS RULES (10–15)
        suspicious_strength = max(
            a_low,  # 10
            min(s_med, b_med),  # 11
            min(b_med, t_med),  # 12
            min(n_med, a_med),  # 13
            min(s_low, b_med),  # 14
            min(f_med, a_med),  # 15
        )

        # MAMDANI IMPLICATION
        misinfo_clipped = np.fmin(misinfo_strength, self._misinformation_mf)
        credible_clipped = np.fmin(credible_strength, self._credible_mf)
        suspicious_clipped = np.fmin(suspicious_strength, self._suspicious_mf)

        # AGGREGATION
        aggregated = np.fmax(misinfo_clipped, np.fmax(credible_clipped, suspicious_clipped))

        # DEFUZZIFICATION
        if aggregated.sum() == 0:
            LOGGER.debug("Empty aggregate MF, returning 0.5")
            return 0.5

        result = fuzz.defuzz(self.UNIVERSE, aggregated, "centroid")
        return float(np.clip(result, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Backwards-compatible alias.
# Some tooling/tests expect `FuzzyEngine` to be importable from this module.
# ---------------------------------------------------------------------------
FuzzyEngine = FuzzyMisinformationEngine

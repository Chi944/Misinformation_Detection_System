"""Master misinformation detector interface.

This module belongs to the *detector* component of the pipeline. It will later
wire together all models, the ensemble, fuzzy engine, feedback loop, and
evaluation pipeline.
"""

from typing import Any, Dict


class MisinformationDetector:
    """Stub master detector class."""

    def __init__(self, config: str = "config.yaml", fast_mode: bool = False) -> None:
        """Initialise the detector.

        Args:
            config: Path to configuration YAML.
            fast_mode: Whether to run in reduced-resource mode.
        """

        self.config_path = config
        self.fast_mode = fast_mode
        raise NotImplementedError("MisinformationDetector not implemented yet.")

    def predict(self, text: str) -> Dict[str, Any]:
        """Run a prediction on a single text.

        Args:
            text: Input text.

        Returns:
            dict: Structured prediction dictionary.
        """

        raise NotImplementedError("MisinformationDetector.predict not implemented yet.")


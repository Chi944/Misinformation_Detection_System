"""TF-IDF + neural network model for misinformation detection.

This module belongs to the *models* component of the pipeline. It will later
provide a Keras-based deep model over rich TF-IDF and auxiliary features.
"""

from typing import Any, Dict


class TfidfDNNClassifier:
    """Stub TF-IDF + DNN classifier.

    The final implementation will build TF-IDF and character n-gram features,
    concatenate additional handcrafted features, and train a multi-layer
    neural network in TensorFlow/Keras.

    Args:
        config: Optional configuration dictionary.
    """

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        self.config = config or {}
        raise NotImplementedError("TfidfDNNClassifier not implemented yet.")


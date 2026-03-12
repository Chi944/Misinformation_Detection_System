"""BERT-based misinformation classifier.

This module belongs to the *models* component of the pipeline. It will later
implement a PyTorch BERT classifier and training loop as specified in the
design. For now, it exposes stub classes so other parts of the system can
import them without breaking.
"""

from typing import Any, Dict


class BERTMisinformationClassifier:
    """Stub BERT classifier for misinformation detection.

    This class will wrap a Hugging Face `bert-base-uncased` model and expose
    a `forward` / `predict` API that returns probabilities for
    credible vs misinformation.

    Args:
        config: Optional configuration dictionary.
    """

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        self.config = config or {}
        raise NotImplementedError("BERTMisinformationClassifier not implemented yet.")


class BERTTrainer:
    """Stub trainer for the BERTMisinformationClassifier.

    This trainer will handle data loaders, mixed-precision training,
    gradient clipping, early stopping, and checkpoint saving.

    Args:
        model: BERTMisinformationClassifier instance.
        config: Optional configuration dictionary.
    """

    def __init__(
        self,
        model: BERTMisinformationClassifier,
        config: Dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.config = config or {}
        raise NotImplementedError("BERTTrainer not implemented yet.")


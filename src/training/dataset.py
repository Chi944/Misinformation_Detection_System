"""Dataset utilities for misinformation detection.

This module belongs to the *training* component of the pipeline. It will later
provide unified dataset representations for PyTorch, TensorFlow, and scikit-learn.
"""


class MisinformationDataset:
    """Stub dataset wrapper for misinformation tasks."""

    def __init__(self, csv_path: str) -> None:
        self.csv_path = csv_path
        raise NotImplementedError("MisinformationDataset not implemented yet.")


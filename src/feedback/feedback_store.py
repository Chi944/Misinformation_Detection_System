"""Feedback storage for misinformation detection.

This module belongs to the *feedback* component of the pipeline. It will later
persist model predictions, ground truth labels, and error signals in SQLite.
"""


class FeedbackStore:
    """Stub feedback store backed by SQLite.

    The final implementation will provide methods to save feedback entries,
    compute feedback scores using nearest neighbors, and export history.
    """

    def __init__(self, path: str = "feedback.db") -> None:
        self.path = path
        raise NotImplementedError("FeedbackStore not implemented yet.")


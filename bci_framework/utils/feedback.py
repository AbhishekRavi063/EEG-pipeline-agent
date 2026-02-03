"""
Human-in-the-loop feedback placeholder.
API for correction labels during live use; online label correction.
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FeedbackEntry:
    """Single feedback: trial index, predicted label, user correction (optional)."""
    trial_id: int
    predicted: int
    corrected: int | None = None
    timestamp_sec: float = 0.0


class HumanFeedbackAPI:
    """
    Placeholder for human-in-the-loop: submit corrections during live use.
    Corrections can be used for online retraining or evaluation.
    """

    def __init__(self, max_history: int = 1000) -> None:
        self._history: deque[FeedbackEntry] = deque(maxlen=max_history)
        self._pending_corrections: list[tuple[int, int, int]] = []  # trial_id, pred, correct

    def submit_correction(self, trial_id: int, predicted: int, corrected: int) -> None:
        """User indicates true label (corrected) for a trial."""
        self._pending_corrections.append((trial_id, predicted, corrected))
        self._history.append(FeedbackEntry(trial_id=trial_id, predicted=predicted, corrected=corrected))
        logger.debug("Feedback: trial %d pred %d -> corrected %d", trial_id, predicted, corrected)

    def get_pending_corrections(self) -> list[tuple[int, int, int]]:
        """Return list of (trial_id, predicted, corrected) for online learning."""
        out = list(self._pending_corrections)
        self._pending_corrections.clear()
        return out

    def get_correction_accuracy(self) -> float | None:
        """Accuracy of predictions where user gave a correction (pred == corrected)."""
        if not self._history:
            return None
        corrected_entries = [e for e in self._history if e.corrected is not None]
        if not corrected_entries:
            return None
        return float(np.mean([e.predicted == e.corrected for e in corrected_entries]))

    def clear(self) -> None:
        self._history.clear()
        self._pending_corrections.clear()

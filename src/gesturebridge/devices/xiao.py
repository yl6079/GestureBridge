from __future__ import annotations

from dataclasses import dataclass
from time import time


@dataclass(slots=True)
class WakeEvent:
    timestamp_ms: int
    activity_level: float
    event_type: str = "hand_activity"


@dataclass(slots=True)
class XIAODetector:
    threshold: float = 0.5

    def detect_activity(self, activity_level: float) -> WakeEvent | None:
        if activity_level >= self.threshold:
            return WakeEvent(timestamp_ms=int(time() * 1000), activity_level=activity_level)
        return None

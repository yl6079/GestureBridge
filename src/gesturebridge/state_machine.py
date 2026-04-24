from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class SystemState(Enum):
    IDLE_LOW_POWER = auto()
    WAKE_REQUESTED = auto()
    ACTIVE = auto()
    COOLDOWN = auto()


class ModeState(Enum):
    MODE_SELECT = auto()
    TRANSLATE_SIGN_TO_SPEECH = auto()
    TRANSLATE_SPEECH_TO_SIGN = auto()
    LEARN_TEACHING = auto()
    LEARN_PRACTICE = auto()


@dataclass(slots=True)
class SystemStateMachine:
    state: SystemState = SystemState.IDLE_LOW_POWER
    mode: ModeState = ModeState.MODE_SELECT

    def on_activity_detected(self, score: float, threshold: float) -> bool:
        if self.state == SystemState.IDLE_LOW_POWER and score >= threshold:
            self.state = SystemState.WAKE_REQUESTED
            return True
        return False

    def on_wake_ack(self) -> None:
        if self.state == SystemState.WAKE_REQUESTED:
            self.state = SystemState.ACTIVE

    def on_mode_selected(self, mode: ModeState) -> None:
        if self.state == SystemState.ACTIVE:
            self.mode = mode

    def on_inactivity_timeout(self) -> None:
        if self.state == SystemState.ACTIVE:
            self.state = SystemState.COOLDOWN

    def on_sleep_complete(self) -> None:
        if self.state == SystemState.COOLDOWN:
            self.state = SystemState.IDLE_LOW_POWER
            self.mode = ModeState.MODE_SELECT

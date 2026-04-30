from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from time import monotonic


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


class DaemonState(Enum):
    STANDBY = auto()
    WAKING = auto()
    ACTIVE = auto()
    IDLE_TIMEOUT = auto()
    SHUTDOWN = auto()


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


@dataclass(slots=True)
class DaemonStateMachine:
    idle_timeout_seconds: int
    min_active_seconds: int
    state: DaemonState = DaemonState.STANDBY
    active_since: float = 0.0
    last_activity: float = 0.0

    def __post_init__(self) -> None:
        now = monotonic()
        self.active_since = now
        self.last_activity = now

    def on_human_on(self) -> bool:
        now = monotonic()
        self.last_activity = now
        if self.state == DaemonState.STANDBY:
            self.state = DaemonState.WAKING
            return True
        if self.state in {DaemonState.ACTIVE, DaemonState.IDLE_TIMEOUT}:
            self.state = DaemonState.ACTIVE
        return False

    def on_main_started(self) -> None:
        now = monotonic()
        self.active_since = now
        self.last_activity = now
        self.state = DaemonState.ACTIVE

    def on_human_off(self) -> None:
        self.last_activity = monotonic()
        if self.state == DaemonState.ACTIVE:
            self.state = DaemonState.IDLE_TIMEOUT

    def on_interaction(self) -> None:
        self.last_activity = monotonic()
        if self.state in {DaemonState.ACTIVE, DaemonState.IDLE_TIMEOUT}:
            self.state = DaemonState.ACTIVE

    def should_shutdown(self) -> bool:
        now = monotonic()
        active_for = now - self.active_since
        idle_for = now - self.last_activity
        return active_for >= self.min_active_seconds and idle_for >= self.idle_timeout_seconds

    def on_shutdown(self) -> None:
        self.state = DaemonState.SHUTDOWN

    def on_shutdown_complete(self) -> None:
        self.state = DaemonState.STANDBY
        self.active_since = monotonic()
        self.last_activity = self.active_since

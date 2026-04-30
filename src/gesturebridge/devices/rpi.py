from __future__ import annotations

from dataclasses import dataclass, field
from time import monotonic

from gesturebridge.devices.xiao import WakeEvent


@dataclass(slots=True)
class RPiRuntime:
    awake: bool = False
    last_active: float = field(default_factory=monotonic)
    wake_count: int = 0

    def handle_wake(self, wake_event: WakeEvent) -> str:
        if self.awake:
            self.last_active = monotonic()
            return "busy_reject"
        self.awake = True
        self.wake_count += 1
        self.last_active = monotonic()
        return "wake_ack"

    def touch(self) -> None:
        self.last_active = monotonic()

    def maybe_sleep(self, inactivity_seconds: int) -> bool:
        if self.awake and monotonic() - self.last_active >= inactivity_seconds:
            self.awake = False
            return True
        return False

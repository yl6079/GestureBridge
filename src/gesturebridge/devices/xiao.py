from __future__ import annotations

from dataclasses import dataclass
import re
from time import time
from typing import Iterable

from gesturebridge.config import SerialConfig

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


@dataclass(slots=True)
class SerialEvent:
    event_type: str
    payload: str
    timestamp_ms: int
    score: float | None = None


def parse_serial_line(line: str, config: SerialConfig) -> SerialEvent | None:
    token = line.strip()
    if not token:
        return None
    ts = int(time() * 1000)
    if token == config.human_on_token:
        return SerialEvent(event_type="HUMAN_ON", payload=token, timestamp_ms=ts)
    if token == config.human_off_token:
        return SerialEvent(event_type="HUMAN_OFF", payload=token, timestamp_ms=ts)
    if token == config.ping_token:
        return SerialEvent(event_type="PING", payload=token, timestamp_ms=ts)
    if token.startswith(f"{config.err_prefix}:"):
        return SerialEvent(event_type="ERR", payload=token, timestamp_ms=ts)
    # Support ESP32 Edge Impulse style logs, e.g. "Hand: 0.95312", "Empty: 0.04688"
    score_match = re.match(r"^\s*([A-Za-z_]+)\s*:\s*([0-9]*\.?[0-9]+)\s*$", token)
    if score_match:
        label = score_match.group(1)
        score = float(score_match.group(2))
        if label.lower() == config.hand_label.lower():
            if score >= config.hand_on_threshold:
                return SerialEvent(event_type="HUMAN_ON", payload=token, timestamp_ms=ts, score=score)
            if score <= config.hand_off_threshold:
                return SerialEvent(event_type="HUMAN_OFF", payload=token, timestamp_ms=ts, score=score)
            return SerialEvent(event_type="UNKNOWN", payload=token, timestamp_ms=ts, score=score)
        if label.lower() == config.empty_label.lower():
            # High empty probability is treated as "no person in frame".
            if score >= config.empty_off_threshold:
                return SerialEvent(event_type="HUMAN_OFF", payload=token, timestamp_ms=ts, score=score)
            return SerialEvent(event_type="UNKNOWN", payload=token, timestamp_ms=ts, score=score)
    return SerialEvent(event_type="UNKNOWN", payload=token, timestamp_ms=ts)


def iter_mock_serial_events(lines: Iterable[str], config: SerialConfig) -> list[SerialEvent]:
    events: list[SerialEvent] = []
    for line in lines:
        evt = parse_serial_line(line, config)
        if evt is not None:
            events.append(evt)
    return events


@dataclass(slots=True)
class SerialBridge:
    config: SerialConfig
    serial_port: object | None = None

    def connect(self) -> bool:
        if self.serial_port is not None:
            return True
        try:
            import serial  # type: ignore[import-not-found]

            self.serial_port = serial.Serial(
                self.config.port,
                self.config.baudrate,
                timeout=self.config.timeout_seconds,
            )
            return True
        except Exception:
            self.serial_port = None
            return False

    def disconnect(self) -> None:
        port = self.serial_port
        self.serial_port = None
        if port is not None:
            try:
                port.close()
            except Exception:
                pass

    def read_event(self) -> SerialEvent | None:
        if self.serial_port is None and not self.connect():
            return None
        assert self.serial_port is not None
        try:
            raw = self.serial_port.readline()
            if not raw:
                return None
            token = raw.decode("utf-8", errors="ignore")
            return parse_serial_line(token, self.config)
        except Exception:
            self.disconnect()
            return None

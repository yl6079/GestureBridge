from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from time import monotonic, sleep

from gesturebridge.config import SystemConfig
from gesturebridge.devices.xiao import SerialBridge, SerialEvent, parse_serial_line
from gesturebridge.state_machine import DaemonState, DaemonStateMachine


@dataclass(slots=True)
class ProcessHandle:
    process: subprocess.Popen[str] | None = None
    started_at: float = 0.0

    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def start(self, command: tuple[str, ...]) -> None:
        if self.is_running():
            return
        self.process = subprocess.Popen(command)
        self.started_at = monotonic()

    def stop(self) -> None:
        if not self.is_running():
            return
        assert self.process is not None
        self.process.terminate()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=3)


@dataclass(slots=True)
class StandbyDaemon:
    config: SystemConfig
    main_process: ProcessHandle = field(default_factory=ProcessHandle)
    state_machine: DaemonStateMachine | None = None
    serial: SerialBridge | None = None
    last_human_on: float = 0.0
    last_human_off: float = 0.0
    last_ping: float = field(default_factory=monotonic)
    last_activity: float = field(default_factory=monotonic)
    last_idle_log_second: int = -1

    def __post_init__(self) -> None:
        self.state_machine = DaemonStateMachine(
            idle_timeout_seconds=self.config.daemon.idle_timeout_seconds,
            min_active_seconds=self.config.daemon.min_active_seconds,
        )
        self.serial = SerialBridge(self.config.serial)
        print(
            f"[daemon] boot state={self.state_machine.state.name} "
            f"serial={self.config.serial.port}@{self.config.serial.baudrate}"
        )

    def _log(self, message: str) -> None:
        assert self.state_machine is not None
        print(f"[daemon] state={self.state_machine.state.name} {message}")

    def apply_serial_event(self, event: SerialEvent) -> str:
        now = monotonic()
        self._log(f"event={event.event_type} payload='{event.payload}' score={event.score}")
        if event.event_type == "PING":
            self.last_ping = now
            return "ping"

        if event.event_type == "HUMAN_ON":
            if now - self.last_human_on < self.config.daemon.debounce_human_on_seconds:
                return "debounced_on"
            self.last_human_on = now
            self.last_activity = now
            assert self.state_machine is not None
            should_wake = self.state_machine.on_human_on()
            if should_wake:
                self.main_process.start(self.config.daemon.main_command)
                self.state_machine.on_main_started()
                self._log(f"main_started cmd={' '.join(self.config.daemon.main_command)}")
                return "started_main"
            return "already_active"

        if event.event_type == "HUMAN_OFF":
            if now - self.last_human_off < self.config.daemon.debounce_human_off_seconds:
                return "debounced_off"
            self.last_human_off = now
            self.last_activity = now
            assert self.state_machine is not None
            self.state_machine.on_human_off()
            return "human_off"

        if event.event_type == "ERR":
            self.last_activity = now
            return "serial_error"

        self.last_activity = now
        return "ignored"

    def tick(self) -> str:
        now = monotonic()
        assert self.state_machine is not None
        if self.state_machine.state in {DaemonState.ACTIVE, DaemonState.IDLE_TIMEOUT}:
            if self.state_machine.state == DaemonState.IDLE_TIMEOUT:
                idle_seconds = int(now - self.state_machine.last_activity)
                if idle_seconds != self.last_idle_log_second:
                    self.last_idle_log_second = idle_seconds
                    self._log(f"idle_countdown={idle_seconds}s/{self.config.daemon.idle_timeout_seconds}s")
            if not self.main_process.is_running():
                self.state_machine.on_shutdown_complete()
                self._log("main_exited -> standby")
                return "main_exited"

            if self.state_machine.should_shutdown():
                self.state_machine.on_shutdown()
                self.main_process.stop()
                self.state_machine.on_shutdown_complete()
                self._log("idle_timeout reached -> main_stopped")
                return "idle_shutdown"

        if now - self.last_ping > self.config.serial.heartbeat_timeout_seconds and self.state_machine.state == DaemonState.ACTIVE:
            self.state_machine.on_shutdown()
            self.main_process.stop()
            self.state_machine.on_shutdown_complete()
            self._log("heartbeat timeout -> main_stopped")
            return "heartbeat_shutdown"
        return "noop"

    def run_mock(self, serial_lines: list[str]) -> list[str]:
        outputs: list[str] = []
        for line in serial_lines:
            event = parse_serial_line(line, self.config.serial)
            if event is not None:
                outputs.append(self.apply_serial_event(event))
            outputs.append(self.tick())
        return outputs

    def run_forever(self) -> None:
        # This loop intentionally keeps the standby process lightweight.
        assert self.serial is not None
        while True:
            event = self.serial.read_event()
            if event is not None:
                self.apply_serial_event(event)
            self.tick()
            sleep(self.config.daemon.poll_interval_seconds)

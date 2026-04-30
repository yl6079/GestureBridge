from gesturebridge.config import SystemConfig
from gesturebridge.devices.xiao import parse_serial_line
from gesturebridge.state_machine import DaemonStateMachine
from gesturebridge.system.daemon import StandbyDaemon


def test_parse_serial_line_tokens() -> None:
    cfg = SystemConfig()
    on_evt = parse_serial_line("HUMAN_ON", cfg.serial)
    off_evt = parse_serial_line("HUMAN_OFF", cfg.serial)
    ping_evt = parse_serial_line("PING", cfg.serial)
    assert on_evt is not None and on_evt.event_type == "HUMAN_ON"
    assert off_evt is not None and off_evt.event_type == "HUMAN_OFF"
    assert ping_evt is not None and ping_evt.event_type == "PING"

def test_parse_serial_line_edge_impulse_scores() -> None:
    cfg = SystemConfig()
    cfg.serial.hand_on_threshold = 0.6
    cfg.serial.hand_off_threshold = 0.3
    on_evt = parse_serial_line("Hand: 0.91", cfg.serial)
    off_evt = parse_serial_line("Hand: 0.08", cfg.serial)
    empty_evt = parse_serial_line("Empty: 0.97", cfg.serial)
    assert on_evt is not None and on_evt.event_type == "HUMAN_ON"
    assert off_evt is not None and off_evt.event_type == "HUMAN_OFF"
    assert empty_evt is not None and empty_evt.event_type == "HUMAN_OFF"


def test_daemon_state_machine_shutdown_logic() -> None:
    sm = DaemonStateMachine(idle_timeout_seconds=0, min_active_seconds=0)
    should_wake = sm.on_human_on()
    assert should_wake is True
    sm.on_main_started()
    sm.on_human_off()
    assert sm.should_shutdown() is True


def test_standby_daemon_mock_flow() -> None:
    cfg = SystemConfig()
    cfg.daemon.main_command = ("python", "-c", "print('main')")
    cfg.daemon.idle_timeout_seconds = 0
    cfg.daemon.min_active_seconds = 0
    daemon = StandbyDaemon(config=cfg)
    outputs = daemon.run_mock(["PING", "HUMAN_ON", "HUMAN_OFF"])
    assert "started_main" in outputs

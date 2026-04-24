from __future__ import annotations

import json
from pathlib import Path
from time import sleep
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gesturebridge.bootstrap import build_controller


def main() -> None:
    controller = build_controller()
    controller.config.thresholds.inactivity_seconds = 1

    events = []
    events.append({"step": "idle", "state": controller.state_machine.state.name, "awake": controller.rpi.awake})

    events.append({"step": "wake_low_signal", "result": controller.wake_if_needed(0.1), "awake": controller.rpi.awake})
    events.append({"step": "wake_valid_signal", "result": controller.wake_if_needed(0.9), "awake": controller.rpi.awake})

    sleep(1.2)
    events.append({"step": "housekeeping", "result": controller.housekeeping(), "awake": controller.rpi.awake})

    summary = {
        "wake_count": controller.rpi.wake_count,
        "final_state": controller.state_machine.state.name,
        "events": events,
    }

    path = Path("artifacts/power_report.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved report to {path}")


if __name__ == "__main__":
    main()

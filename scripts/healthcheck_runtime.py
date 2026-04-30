from __future__ import annotations

import json
from pathlib import Path
from urllib.request import urlopen

from gesturebridge.config import SystemConfig


def main() -> None:
    cfg = SystemConfig()
    url = f"http://{cfg.web.host}:{cfg.web.port}/api/state"
    with urlopen(url, timeout=2) as response:  # noqa: S310 - local healthcheck endpoint
        payload = json.loads(response.read().decode("utf-8"))
    report = {
        "url": url,
        "status": payload.get("status", "unknown"),
        "mode": payload.get("mode", "unknown"),
    }
    output = Path("artifacts/healthcheck_report.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

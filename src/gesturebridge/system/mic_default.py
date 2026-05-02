"""Best-effort PulseAudio default input (e.g. Logitech C270) for Web Speech on Linux."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def prefer_c270_default_mic() -> None:
    """Run scripts/set_default_mic_c270.sh if present (repo or cwd). No-op on non-Linux."""
    if sys.platform != "linux":
        return
    here = Path(__file__).resolve().parent
    # gesturebridge/system -> src -> repo root (scripts/set_default_mic_c270.sh)
    root = here.parents[2]
    candidates = [root / "scripts" / "set_default_mic_c270.sh", Path.cwd() / "scripts" / "set_default_mic_c270.sh"]
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        return
    try:
        subprocess.run(["/bin/bash", str(path)], check=False, timeout=8)
    except (OSError, subprocess.TimeoutExpired):
        pass

from __future__ import annotations

import argparse
import re
import sys
import time
from datetime import datetime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realtime serial monitor for ESP32 messages.")
    parser.add_argument("--port", default="/dev/ttyACM0", help="Serial device path, e.g. /dev/ttyACM0")
    parser.add_argument("--baudrate", type=int, default=115200, help="Serial baudrate")
    parser.add_argument(
        "--only-scores",
        action="store_true",
        help="Only print score lines (e.g. 'Hand: 0.81', 'Empty: 0.95')",
    )
    return parser.parse_args()


def now_ts() -> str:
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def main() -> None:
    args = parse_args()
    score_re = re.compile(r"^\s*([A-Za-z_]+)\s*:\s*([0-9]*\.?[0-9]+)\s*$")

    try:
        import serial  # type: ignore[import-not-found]
    except Exception as exc:
        print(f"[error] pyserial not available: {exc}", file=sys.stderr)
        print("[hint] install with: pip install pyserial", file=sys.stderr)
        raise SystemExit(1)

    print(f"[serial] opening {args.port} @ {args.baudrate}")
    print("[serial] press Ctrl+C to stop")

    try:
        with serial.Serial(args.port, args.baudrate, timeout=0.2) as ser:
            while True:
                raw = ser.readline()
                if not raw:
                    continue
                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue

                m = score_re.match(line)
                if args.only_scores and not m:
                    continue

                if m:
                    label = m.group(1)
                    score = float(m.group(2))
                    print(f"[{now_ts()}] SCORE {label:<8} {score:.5f}")
                else:
                    print(f"[{now_ts()}] RAW   {line}")
    except KeyboardInterrupt:
        print("\n[serial] stopped by user")
    except Exception as exc:
        print(f"[error] serial monitor failed: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()


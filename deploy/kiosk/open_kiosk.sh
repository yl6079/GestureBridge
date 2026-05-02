#!/usr/bin/env bash
# Opens GestureBridge in Chromium. Default: fullscreen kiosk. Use KIOSK=0 for windowed.
set -euo pipefail

URL="${1:-http://127.0.0.1:8080}"

SCRIPT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
if [ -f "$SCRIPT_DIR/scripts/set_default_mic_c270.sh" ]; then
  bash "$SCRIPT_DIR/scripts/set_default_mic_c270.sh" || true
fi

CHROME_FLAGS=(--noerrdialogs --disable-infobars)
if [ "${KIOSK:-1}" = "1" ]; then
  CHROME_FLAGS=(--kiosk "${CHROME_FLAGS[@]}")
else
  CHROME_FLAGS=(--new-window "${CHROME_FLAGS[@]}")
fi

if command -v chromium-browser >/dev/null 2>&1; then
  exec chromium-browser "${CHROME_FLAGS[@]}" "$URL"
fi

if command -v chromium >/dev/null 2>&1; then
  exec chromium "${CHROME_FLAGS[@]}" "$URL"
fi

echo "No chromium browser found" >&2
exit 1


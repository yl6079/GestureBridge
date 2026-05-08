#!/usr/bin/env bash
# GestureBridge launcher. Runtime overrides:
#   GESTUREBRIDGE_FORCE_DAEMON=1     force daemon mode (otherwise auto-detect)
#   GESTUREBRIDGE_VOSK_INPUT_DEVICE  override Vosk audio input device
#   GESTUREBRIDGE_VOSK_SKIP_PULSE=1  skip Pulse/PipeWire preference logic
#   GESTUREBRIDGE_DISABLE_TTS=1      disable speech output
#   FETCH_VOSK_FORCE_PROXY=1         keep proxy env while downloading Vosk model
set -euo pipefail

PROJECT_DIR="/home/elen6908/Documents/GestureBridge"
VENV_PY="$PROJECT_DIR/.venv311/bin/python"

cd "$PROJECT_DIR"
export PATH="$PROJECT_DIR/.venv311/bin:$PATH"

if [ -L "$VENV_PY" ] && [ ! -e "$VENV_PY" ]; then
  echo "[ERROR] Broken virtual environment interpreter symlink: $VENV_PY"
  echo "The .venv311 directory exists, but python points to a missing target."
  echo "Please recreate the virtual environment:"
  echo "  cd $PROJECT_DIR"
  echo "  rm -rf .venv311"
  echo "  python3 -m venv .venv311"
  echo "  source .venv311/bin/activate"
  echo "  pip install -e '.[dev,ml]'"
  read -rp "Press Enter to exit..."
  exit 1
fi

if [ ! -e "$VENV_PY" ]; then
  echo "[ERROR] Virtual environment Python not found: $VENV_PY"
  echo "Please create the virtual environment first:"
  echo "  cd $PROJECT_DIR"
  echo "  python3 -m venv .venv311"
  echo "  source .venv311/bin/activate"
  echo "  pip install -e '.[dev,ml]'"
  read -rp "Press Enter to exit..."
  exit 1
fi

if [ ! -x "$VENV_PY" ]; then
  echo "[ERROR] Virtual environment Python is not executable: $VENV_PY"
  echo "Try fixing permissions:"
  echo "  chmod +x $VENV_PY"
  read -rp "Press Enter to exit..."
  exit 1
fi

if [ -f "$PROJECT_DIR/scripts/set_default_mic_c270.sh" ]; then
  bash "$PROJECT_DIR/scripts/set_default_mic_c270.sh" || true
fi

# Mode auto-detect: with the ESP32 wake-trigger plugged in we use the
# power-saving daemon (it spawns the main app on a HUMAN_ON event).
# Without ESP32 the daemon would sit in standby forever and the web UI
# at localhost:8080 returns "fail to fetch" in Chromium kiosk — so fall
# back to running the main app directly (always-on). Override via
# `GESTUREBRIDGE_FORCE_DAEMON=1 ./start_gesturebridge.sh` if needed.
HAS_SERIAL=0
if [ "${GESTUREBRIDGE_FORCE_DAEMON:-0}" = "1" ]; then
  HAS_SERIAL=1
elif compgen -G "/dev/ttyUSB*" > /dev/null 2>&1 || compgen -G "/dev/ttyACM*" > /dev/null 2>&1; then
  HAS_SERIAL=1
fi

if [ "$HAS_SERIAL" = "1" ]; then
  echo "[start] ESP32 serial detected -> daemon mode (wake-gated)"
  exec "$VENV_PY" -m gesturebridge.app --run-daemon
else
  echo "[start] no ESP32 serial -> main mode (always-on)"
  exec "$VENV_PY" -m gesturebridge.app --run-main --camera-index 0
fi

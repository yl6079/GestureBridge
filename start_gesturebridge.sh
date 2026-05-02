#!/usr/bin/env bash
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

exec "$VENV_PY" -m gesturebridge.app --run-daemon

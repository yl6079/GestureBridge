#!/usr/bin/env bash
# Best-effort: set PulseAudio/PipeWire (via pipewire-pulse) default capture device
# to the Logitech C270 webcam microphone so Chromium Web Speech uses it.
set -uo pipefail

if ! command -v pactl >/dev/null 2>&1; then
  echo "[gesturebridge-mic] pactl not found; skip setting default source"
  exit 0
fi

SOURCE=""
while IFS= read -r line; do
  [[ -z "${line// /}" ]] && continue
  if printf '%s\n' "$line" | grep -qiF "c270"; then
    SOURCE=$(printf '%s\n' "$line" | awk '{print $2}')
    break
  fi
done < <(pactl list short sources 2>/dev/null || true)

if [[ -z "${SOURCE}" ]]; then
  echo "[gesturebridge-mic] WARN: no source name containing 'c270' (see: pactl list short sources)"
  exit 0
fi

if pactl set-default-source "$SOURCE"; then
  echo "[gesturebridge-mic] default source -> ${SOURCE}"
fi
exit 0

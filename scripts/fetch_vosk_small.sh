#!/usr/bin/env bash
# Download and extract the Vosk small English model (offline STT).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${1:-$ROOT/artifacts/vosk}"
mkdir -p "$DEST"
NAME="vosk-model-small-en-us-0.15"
ZIP="$DEST/${NAME}.zip"
URL="https://alphacephei.com/vosk/models/${NAME}.zip"

# Direct download: many shells set ALL_PROXY / HTTP(S)_PROXY to 127.0.0.1:8080; if nothing
# listens there, curl fails with "Failed to connect to 127.0.0.1 port 8080".
# Set FETCH_VOSK_FORCE_PROXY=1 to keep your normal proxy env for this script.
curl_model() {
  if [[ "${FETCH_VOSK_FORCE_PROXY:-}" == 1 ]]; then
    curl -fsSL "$@"
  else
    env -u ALL_PROXY -u all_proxy -u HTTP_PROXY -u http_proxy -u HTTPS_PROXY -u https_proxy \
      curl -fsSL "$@"
  fi
}

if [[ ! -d "$DEST/$NAME" ]]; then
  if [[ ! -f "$ZIP" ]]; then
    echo "Downloading $URL ..."
    curl_model -o "$ZIP" "$URL"
  fi
  echo "Extracting to $DEST ..."
  unzip -o -q "$ZIP" -d "$DEST"
  echo "Model ready at $DEST/$NAME"
else
  echo "Already present: $DEST/$NAME"
fi

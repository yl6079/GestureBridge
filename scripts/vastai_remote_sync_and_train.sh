#!/usr/bin/env bash
# From the Mac: sync repo + dataset to a vast.ai 5090, run the sweep, pull
# results back. One command end-to-end.
#
# Usage:
#   bash scripts/vastai_remote_sync_and_train.sh user@host -p PORT [-i KEY]
#
# Example (vast.ai SSH form):
#   bash scripts/vastai_remote_sync_and_train.sh root@ssh4.vast.ai -p 12345 -i ~/.ssh/vastai_key

set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 user@host [-p PORT] [-i KEY]" >&2
  exit 1
fi

REMOTE="$1"; shift
SSH_OPTS=(-o StrictHostKeyChecking=accept-new)
RSYNC_SSH=("ssh" "-o" "StrictHostKeyChecking=accept-new")

while [ $# -gt 0 ]; do
  case "$1" in
    -p) SSH_OPTS+=(-p "$2"); RSYNC_SSH+=("-p" "$2"); shift 2;;
    -i) SSH_OPTS+=(-i "$2"); RSYNC_SSH+=("-i" "$2"); shift 2;;
    *) echo "unknown arg: $1"; exit 1;;
  esac
done

LOCAL_REPO="$(cd "$(dirname "$0")/.." && pwd)"
LOCAL_DATA="${LOCAL_DATA:-$HOME/Desktop/Elen6908/data/asl29_raw/asl_alphabet_train/asl_alphabet_train}"
REMOTE_REPO="/workspace/gesture-bridge"
REMOTE_DATA="/workspace/asl29_raw"

echo "==> Syncing repo to $REMOTE:$REMOTE_REPO"
rsync -az --delete \
  --exclude '.venv*' --exclude '.uv-python' --exclude '__pycache__' \
  --exclude '.pytest_cache' --exclude 'data/' --exclude 'artifacts/' \
  -e "${RSYNC_SSH[*]}" "$LOCAL_REPO/" "$REMOTE:$REMOTE_REPO/"

echo "==> Syncing dataset to $REMOTE:$REMOTE_DATA (~1.2 GB, ~1-3 min on a fast box)"
rsync -az -e "${RSYNC_SSH[*]}" "$LOCAL_DATA/" "$REMOTE:$REMOTE_DATA/"

echo "==> Installing python deps on remote"
ssh "${SSH_OPTS[@]}" "$REMOTE" bash <<EOF
  set -euo pipefail
  cd $REMOTE_REPO
  python -m pip install --upgrade pip
  pip install -e ".[ml]"
  pip install pandas scikit-learn
EOF

echo "==> Running sweep on remote"
ssh "${SSH_OPTS[@]}" "$REMOTE" bash <<EOF
  set -euo pipefail
  cd $REMOTE_REPO
  REPO_DIR=$REMOTE_REPO DATA_ROOT=$REMOTE_DATA bash scripts/vastai_train.sh
EOF

echo "==> Pulling results back to Mac"
mkdir -p "$LOCAL_REPO/artifacts/asl29"
rsync -az -e "${RSYNC_SSH[*]}" \
  "$REMOTE:$REMOTE_REPO/artifacts/asl29/" \
  "$LOCAL_REPO/artifacts/asl29/"

echo
echo "Done. New artifacts under artifacts/asl29/. Notable files:"
echo "  artifacts/asl29/best_pointer.json"
echo "  artifacts/asl29/eval/contig_test_metrics_newmodel.json"
echo "  artifacts/asl29/tflite/model_fp32.tflite"

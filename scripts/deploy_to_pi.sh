#!/usr/bin/env bash
# Deploy updated model artifacts + landmark MLP + MediaPipe asset to the Pi.
#
# Designed to be non-destructive: rsyncs only the artifacts/ subtree
# under the deployed test/ workspace and never touches Yizheng's source
# code or the venv. Coordinate with him before running so we don't ship
# while he's mid-demo.
#
# Usage:
#   bash scripts/deploy_to_pi.sh
#
# Override targets via env:
#   PI_USER=elen6908 PI_HOST=100.127.215.9 PI_PATH=/home/elen6908/Documents/GestureBridge \
#     bash scripts/deploy_to_pi.sh
#
# Requires sshpass with the Pi password (held in env PI_PASS, NOT hardcoded).

set -euo pipefail

PI_USER="${PI_USER:-elen6908}"
PI_HOST="${PI_HOST:-100.127.215.9}"
# Project moved off the old `test/` subdir on 2026-04-30; new canonical path:
PI_PATH="${PI_PATH:-/home/elen6908/Documents/GestureBridge}"

LOCAL_ARTIFACTS="$(cd "$(dirname "$0")/.." && pwd)/artifacts"

if [ -z "${PI_PASS:-}" ]; then
  echo "Set PI_PASS env var with the Pi's SSH password before running." >&2
  exit 2
fi

if ! command -v sshpass >/dev/null 2>&1; then
  echo "sshpass not found. brew install sshpass-2.0 (or compile from source)." >&2
  exit 2
fi

REMOTE="${PI_USER}@${PI_HOST}"
echo "==> Pre-flight check on $REMOTE"
sshpass -p "$PI_PASS" ssh -o StrictHostKeyChecking=accept-new "$REMOTE" \
  "cd $PI_PATH && git rev-parse HEAD && ls artifacts/asl29/tflite/"

echo "==> Backing up current artifacts on Pi"
sshpass -p "$PI_PASS" ssh "$REMOTE" \
  "cd $PI_PATH && tar -czf /tmp/artifacts_pre_deploy_$(date +%Y%m%d_%H%M%S).tgz artifacts/ 2>/dev/null && ls -lh /tmp/artifacts_pre_deploy_*.tgz | tail -1"

echo "==> Rsync new artifacts (asl29 tflite + landmark_mlp + mediapipe)"
sshpass -p "$PI_PASS" rsync -az \
  -e "ssh -o StrictHostKeyChecking=accept-new" \
  "$LOCAL_ARTIFACTS/asl29/tflite/" \
  "$REMOTE:$PI_PATH/artifacts/asl29/tflite/"

if [ -d "$LOCAL_ARTIFACTS/asl29/landmark_mlp" ]; then
  sshpass -p "$PI_PASS" rsync -az \
    -e "ssh -o StrictHostKeyChecking=accept-new" \
    "$LOCAL_ARTIFACTS/asl29/landmark_mlp/" \
    "$REMOTE:$PI_PATH/artifacts/asl29/landmark_mlp/"
fi

if [ -f "$LOCAL_ARTIFACTS/mediapipe/hand_landmarker.task" ]; then
  sshpass -p "$PI_PASS" rsync -az \
    -e "ssh -o StrictHostKeyChecking=accept-new" \
    "$LOCAL_ARTIFACTS/mediapipe/" \
    "$REMOTE:$PI_PATH/artifacts/mediapipe/"
fi

# Phase 2 word recognition: ship the Conv1D + GRU npz weights, labels,
# eval JSON, and demo vocabulary. Pure-numpy at runtime so no extra
# Python deps on the Pi.
if [ -d "$LOCAL_ARTIFACTS/wlasl100" ]; then
  sshpass -p "$PI_PASS" rsync -az \
    -e "ssh -o StrictHostKeyChecking=accept-new" \
    --include="*.npz" --include="labels.txt" --include="eval.json" \
    --include="demo_vocab.txt" --include="per_class_test*.json" \
    --exclude="*" \
    "$LOCAL_ARTIFACTS/wlasl100/" \
    "$REMOTE:$PI_PATH/artifacts/wlasl100/"
fi

# Word reference clips for speech-to-sign visual output. Each clip is
# small (≤6 MB); stays under license terms (not redistributed via git).
if [ -d "$(dirname "$LOCAL_ARTIFACTS")/assets/word_clips" ]; then
  sshpass -p "$PI_PASS" rsync -az \
    -e "ssh -o StrictHostKeyChecking=accept-new" \
    --include="*.mp4" --include="*.webm" --include="SOURCES.md" \
    --exclude="*" \
    "$(dirname "$LOCAL_ARTIFACTS")/assets/word_clips/" \
    "$REMOTE:$PI_PATH/assets/word_clips/"
fi

echo "==> Verifying on Pi"
sshpass -p "$PI_PASS" ssh "$REMOTE" \
  "cd $PI_PATH && ls -lh artifacts/asl29/tflite/ artifacts/asl29/landmark_mlp/ 2>/dev/null && ls -lh artifacts/wlasl100/ 2>/dev/null && ls -lh assets/word_clips/ 2>/dev/null"

echo
echo "Done. Restart the web app on the Pi to pick up the new artifacts:"
echo "  ssh $REMOTE 'cd $PI_PATH && .venv311/bin/python -m gesturebridge.app --run-main'"

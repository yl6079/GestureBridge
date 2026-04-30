#!/usr/bin/env bash
# Run the ASL29 training sweep on a vast.ai 5090 (or any CUDA box).
#
# Assumes:
# - You've SSH'd or rsync'd this repo to the box at $REMOTE_REPO_DIR.
# - Dataset is at $DATA_ROOT (needs the {A..Z,del,nothing,space} subdirs).
# - Python 3.11+ with CUDA-enabled tensorflow available.
#
# Local usage (from the Mac, after vast.ai instance is up):
#   bash scripts/vastai_remote_sync_and_train.sh user@host:port  # see that
#   helper script for the upload+launch flow.
#
# This script is what the helper invokes on the remote box.

set -euo pipefail

REPO_DIR="${REPO_DIR:-/workspace/gesture-bridge}"
DATA_ROOT="${DATA_ROOT:-/workspace/asl29_raw}"
SPLIT_MODE="${SPLIT_MODE:-contiguous}"
SWEEP_TAG_PREFIX="${SWEEP_TAG_PREFIX:-c_}"

cd "$REPO_DIR"

# 1. Ensure splits exist (use --no-copy: train manifests can point at raw paths).
if [ ! -f "data/asl29/splits/train.csv" ]; then
  python scripts/prepare_asl29.py \
    --no-copy \
    --split-mode "$SPLIT_MODE" \
    --input-dir "$DATA_ROOT"
  python scripts/sanity_check_split.py
fi

# 2. Sweep configurations. Each run takes ~5-10 min on a 5090.
declare -a CONFIGS=(
  "u15_d20  --unfreeze-layers 15 --dropout 0.2"
  "u30_d20  --unfreeze-layers 30 --dropout 0.2"
  "u50_d20  --unfreeze-layers 50 --dropout 0.2"
  "u30_d30  --unfreeze-layers 30 --dropout 0.3"
)

mkdir -p artifacts/asl29/sweep_log

for entry in "${CONFIGS[@]}"; do
  name="${entry%% *}"
  rest="${entry#* }"
  tag="${SWEEP_TAG_PREFIX}${name}"
  echo "===== sweep run: $tag ====="
  python scripts/train_mobilenetv3_asl29.py \
    --tag "$tag" \
    --label-smoothing 0.05 \
    $rest \
    2>&1 | tee "artifacts/asl29/sweep_log/${tag}.log"
done

# 3. Pick the best by val_accuracy and export FP32 TFLite.
python - <<'PY'
import json
from pathlib import Path

metrics_dir = Path("artifacts/asl29")
candidates = []
for p in sorted(metrics_dir.glob("train_metrics_*.json")):
    candidates.append((json.loads(p.read_text()), p))
if not candidates:
    raise SystemExit("no sweep metrics found")
best, best_path = max(candidates, key=lambda kv: kv[0]["best_val_accuracy"])
print(json.dumps(best, indent=2))
print(f"Best metrics file: {best_path}")
Path("artifacts/asl29/best_pointer.json").write_text(json.dumps({
    "best_metrics_path": str(best_path),
    "best_model_path": best["model_path"],
}, indent=2))
PY

# 4. Export TFLite from the best Keras checkpoint.
python scripts/export_tflite_int8_asl29.py

# 5. Evaluate FP32 on the contiguous test split.
python scripts/eval_split.py \
  --model artifacts/asl29/tflite/model_fp32.tflite \
  --labels artifacts/asl29/labels.txt \
  --split-csv data/asl29/splits/test.csv \
  --out-json artifacts/asl29/eval/contig_test_metrics_newmodel.json

echo "Sweep done. Pull these back to Mac:"
echo "  artifacts/asl29/tflite/model_fp32.tflite"
echo "  artifacts/asl29/checkpoints/best_*.keras"
echo "  artifacts/asl29/eval/contig_test_metrics_newmodel.json"
echo "  artifacts/asl29/sweep_log/"

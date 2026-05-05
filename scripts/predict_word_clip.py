"""Predict the WLASL gloss for a single video clip — for live testing.

Wires the existing MediaPipe HandCropper + numpy WordClassifier and reports
top-5 gloss predictions for any short MP4. Lets us validate the trained
model end-to-end on Mac before the Pi UI integration lands.

Usage:
    python scripts/predict_word_clip.py path/to/clip.mp4
    python scripts/predict_word_clip.py path/to/clip.mp4 --frames 30
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gesturebridge.pipelines.hand_crop import HandCropper  # noqa: E402
from gesturebridge.pipelines.word_classifier import WordClassifier  # noqa: E402

# Reuse the same extraction logic as the training data pipeline so that
# inference distribution matches training distribution exactly.
sys.path.insert(0, str(ROOT / "scripts"))
from extract_wlasl_landmarks import extract_clip  # type: ignore[import-not-found]  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("video", type=Path, help="path to a short ASL video clip")
    p.add_argument("--frames", type=int, default=30)
    p.add_argument(
        "--model",
        type=Path,
        default=Path("artifacts/wlasl100/conv1d_small.npz"),
    )
    p.add_argument(
        "--labels",
        type=Path,
        default=Path("artifacts/wlasl100/labels.txt"),
    )
    p.add_argument("--top", type=int, default=5)
    args = p.parse_args()

    if not args.video.exists():
        print(f"video not found: {args.video}", file=sys.stderr)
        return 1

    cropper = HandCropper(output_size=224, padding_ratio=0.25, min_confidence=0.3)
    seq, detect = extract_clip(cropper, args.video, args.frames, fs=1, fe=-1)
    cropper.close()

    detect_rate = float(detect.mean())
    print(f"[predict] hand detected in {int(detect.sum())}/{args.frames} frames "
          f"({detect_rate*100:.0f}% detect rate)")
    if detect_rate < 0.2:
        print("[predict] WARNING: very low hand-detection rate; predictions may be noise.")

    clf = WordClassifier(model_path=args.model, labels_path=args.labels)
    preds = clf.predict(seq, top_k=args.top)
    print(f"[predict] top-{args.top} predictions for {args.video.name}:")
    for i, (lbl, prob) in enumerate(preds, 1):
        bar = "█" * int(prob * 40)
        print(f"  {i}. {lbl:18s} {prob*100:5.1f}%  {bar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

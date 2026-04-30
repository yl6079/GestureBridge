"""Capture C270 frames for INT8 calibration on the Pi.

Run this on the Pi (or any machine with a webcam attached). Captures up
to N frames, optionally guided by class labels (cycle through A..Z + del/
nothing/space). Each frame is written as a PNG under
artifacts/asl29/calibration/<class>/<seq>.png.

Why on-device calibration: Yizheng's previous INT8 export used the
training set for representative data, but the training set is uniformly
lit and tightly cropped, while the C270 in the actual lab has very
different statistics (color cast, contrast, motion blur). INT8 quant
needs calibration data that matches the *deployment* distribution, not
the *training* distribution.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import sleep

import cv2

DEFAULT_CLASSES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["del", "nothing", "space"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/asl29/calibration"))
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--per-class", type=int, default=12, help="Frames per class")
    parser.add_argument("--countdown", type=int, default=3, help="Seconds before each class block")
    parser.add_argument("--interval", type=float, default=0.4, help="Seconds between captures within a class")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--classes", nargs="*", default=None)
    args = parser.parse_args()

    classes = args.classes if args.classes else DEFAULT_CLASSES
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"ERROR: cannot open camera {args.camera}", file=sys.stderr)
        return 1
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    try:
        for class_name in classes:
            class_dir = args.out_dir / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            existing = len(list(class_dir.glob("*.png")))
            print(f"\n=== {class_name} (existing {existing}, will add {args.per_class}) ===")
            for i in range(args.countdown, 0, -1):
                print(f"  starting in {i}...", flush=True)
                # Drain frames during countdown so the capture isn't stale.
                for _ in range(int(10 * 1)):
                    cap.read()
                sleep(0.9)
            for k in range(args.per_class):
                ok, frame = cap.read()
                if not ok:
                    print("  frame read failed", file=sys.stderr)
                    continue
                seq = existing + k
                out_path = class_dir / f"{class_name}_{seq:04d}.png"
                cv2.imwrite(str(out_path), frame)
                print(f"  saved {out_path.name}", flush=True)
                sleep(args.interval)
    finally:
        cap.release()
    print(f"\nDone. Calibration set under: {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

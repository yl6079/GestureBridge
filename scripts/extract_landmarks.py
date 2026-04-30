"""Run MediaPipe HandLandmarker over a split CSV and write a NPZ of
(landmarks_xyz, labels) for fast landmark-MLP training.

Output schema (NPZ):
    X : float32 array, shape (N, 63)  -- (x, y, z) for 21 landmarks,
        flattened. Coordinates are normalized to the *crop* (i.e. relative
        to the bounding box of the detected hand), not the original image,
        so the representation is translation-invariant.
    y : int32 array, shape (N,)       -- class labels
    detected : uint8 array, shape (N,) -- 1 if a hand was detected, else 0
        (rows with detected=0 should typically be filtered before training)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gesturebridge.pipelines.hand_crop import HandCropper


def _normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """Translate to wrist origin, scale by max-abs distance from wrist.

    landmarks: (21, 3) in pixel/relative-z coords.
    Returns:   (63,) float32, scale-and-translation invariant.
    """
    wrist = landmarks[0:1]
    rel = landmarks - wrist
    scale = float(np.linalg.norm(rel[:, :2], axis=1).max())
    if scale < 1e-6:
        scale = 1.0
    rel = rel / scale
    return rel.astype(np.float32).reshape(-1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits-dir", type=Path, default=Path("data/asl29/splits"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/asl29/landmarks"))
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    # Lower confidence for offline data prep — we have ground-truth labels so
    # false positives don't hurt, and the Kaggle ASL dataset's tight cropping
    # + uniform background trips MediaPipe at default 0.3. Empirically: 0.05
    # → 91% detection, 0.3 → 85% on a 300-sample probe.
    cropper = HandCropper(output_size=224, padding_ratio=0.25, min_confidence=0.05)

    for split in ("train", "val", "test"):
        df = pd.read_csv(args.splits_dir / f"{split}.csv")
        if args.limit > 0:
            df = df.sample(n=min(args.limit, len(df)), random_state=0).reset_index(drop=True)

        N = len(df)
        X = np.zeros((N, 63), dtype=np.float32)
        y = np.zeros(N, dtype=np.int32)
        detected = np.zeros(N, dtype=np.uint8)

        misses = 0
        print(f"[{split}] {N} samples")
        for i, row in enumerate(df.itertuples(index=False)):
            img = cv2.imread(str(row.path))
            if img is None:
                misses += 1
                y[i] = int(row.label)
                continue
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            res = cropper.crop(rgb)
            y[i] = int(row.label)
            if not res.found or res.landmarks is None:
                misses += 1
                continue
            X[i] = _normalize_landmarks(res.landmarks)
            detected[i] = 1
            if (i + 1) % 1000 == 0:
                print(f"  {i+1}/{N}  miss-rate so far {misses/(i+1):.3f}", flush=True)

        out_path = args.out_dir / f"{split}.npz"
        np.savez_compressed(out_path, X=X, y=y, detected=detected)
        print(f"  wrote {out_path}  detected={int(detected.sum())}/{N}")

    cropper.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

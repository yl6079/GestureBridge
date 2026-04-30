"""Hand-region cropping using MediaPipe Tasks (HandLandmarker).

Used both at training-data prep and at runtime so the model sees the same
distribution. If no hand is detected, returns a centered resize of the
whole image with `found=False`; the caller decides what to do.

Requires `artifacts/mediapipe/hand_landmarker.task` (downloaded by
`scripts/setup_mediapipe.sh` or available from the Google MediaPipe
storage bucket).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

DEFAULT_MODEL_PATH = Path("artifacts/mediapipe/hand_landmarker.task")


def _build_landmarker(model_path: Path, max_hands: int, min_confidence: float):
    import mediapipe as mp
    from mediapipe.tasks.python import vision, BaseOptions

    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=max_hands,
        min_hand_detection_confidence=min_confidence,
        min_hand_presence_confidence=min_confidence,
        min_tracking_confidence=min_confidence,
    )
    return vision.HandLandmarker.create_from_options(options)


@dataclass(slots=True)
class CropResult:
    image: np.ndarray  # cropped (or resized) RGB image, output_size x output_size
    found: bool
    bbox: Optional[tuple[int, int, int, int]] = None  # x, y, w, h in original-image coords
    landmarks: Optional[np.ndarray] = None  # (21, 3) in original-image coords (px, px, relative-z)


@dataclass(slots=True)
class HandCropper:
    output_size: int = 224
    padding_ratio: float = 0.25
    min_confidence: float = 0.3
    model_path: Path = field(default_factory=lambda: DEFAULT_MODEL_PATH)
    _detector: object = None

    def __post_init__(self) -> None:
        path = self.model_path if self.model_path.is_absolute() else (Path.cwd() / self.model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"hand_landmarker.task not found at {path}. "
                "Download from "
                "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            )
        self._detector = _build_landmarker(path, max_hands=1, min_confidence=self.min_confidence)

    def close(self) -> None:
        if self._detector is not None:
            try:
                self._detector.close()
            except Exception:
                pass
            self._detector = None

    def __del__(self) -> None:
        self.close()

    def crop(self, image_rgb: np.ndarray) -> CropResult:
        import mediapipe as mp

        h, w = image_rgb.shape[:2]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(image_rgb))
        result = self._detector.detect(mp_image)

        if not result.hand_landmarks:
            resized = cv2.resize(image_rgb, (self.output_size, self.output_size), interpolation=cv2.INTER_LINEAR)
            return CropResult(image=resized, found=False)

        lm = result.hand_landmarks[0]
        xs = np.array([p.x for p in lm]) * w
        ys = np.array([p.y for p in lm]) * h
        zs = np.array([p.z for p in lm])

        x0, x1 = float(xs.min()), float(xs.max())
        y0, y1 = float(ys.min()), float(ys.max())
        bw, bh = x1 - x0, y1 - y0
        side = max(bw, bh)
        pad = side * self.padding_ratio
        side_padded = side + 2 * pad
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0

        half = side_padded / 2.0
        x_lo = int(max(0, round(cx - half)))
        y_lo = int(max(0, round(cy - half)))
        x_hi = int(min(w, round(cx + half)))
        y_hi = int(min(h, round(cy + half)))
        if x_hi - x_lo < 4 or y_hi - y_lo < 4:
            resized = cv2.resize(image_rgb, (self.output_size, self.output_size), interpolation=cv2.INTER_LINEAR)
            return CropResult(image=resized, found=False)

        crop_arr = image_rgb[y_lo:y_hi, x_lo:x_hi]
        crop_resized = cv2.resize(crop_arr, (self.output_size, self.output_size), interpolation=cv2.INTER_LINEAR)
        landmarks = np.stack([xs, ys, zs], axis=1).astype(np.float32)
        return CropResult(
            image=crop_resized,
            found=True,
            bbox=(x_lo, y_lo, x_hi - x_lo, y_hi - y_lo),
            landmarks=landmarks,
        )

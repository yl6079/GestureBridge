from __future__ import annotations

import os

import numpy as np


class LandmarkExtractor:
    """Real MediaPipe Hands landmark extractor.

    Extracts 21 hand landmarks (x, y, z) from a BGR camera frame using
    MediaPipe, returning a 63-dim float32 vector compatible with the
    existing classifier interface.

    Falls back to a stub (resize + normalize) when:
    - mediapipe is not installed (test environments, Python 3.13 venv)
    - GESTUREBRIDGE_MOCK_LANDMARKS=1 is set
    - No hand is detected in the frame (returns zeros)
    """

    def __init__(self, output_dim: int = 63) -> None:
        self.output_dim = output_dim
        self._use_mock = os.environ.get("GESTUREBRIDGE_MOCK_LANDMARKS", "0") == "1"
        self._hands = None

        if not self._use_mock:
            try:
                import cv2  # noqa: F401 — ensure opencv is available
                import mediapipe as mp
                self._hands = mp.solutions.hands.Hands(
                    static_image_mode=False,
                    max_num_hands=1,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
            except ImportError:
                # mediapipe not installed — fall back to stub silently
                self._use_mock = True

    def extract(self, frame: np.ndarray) -> np.ndarray:
        if self._use_mock or self._hands is None:
            return self._stub_extract(frame)
        return self._mediapipe_extract(frame)

    def _mediapipe_extract(self, frame: np.ndarray) -> np.ndarray:
        import cv2
        # Convert to RGB as required by MediaPipe
        if frame.ndim == 3 and frame.shape[2] == 3:
            rgb = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_BGR2RGB)
        else:
            rgb = frame.astype(np.uint8)

        results = self._hands.process(rgb)

        if not results.multi_hand_landmarks:
            # No hand detected — return zeros (triggers LOW_CONFIDENCE in classifier)
            return np.zeros(self.output_dim, dtype=np.float32)

        lm = results.multi_hand_landmarks[0].landmark
        coords = np.array([[l.x, l.y, l.z] for l in lm], dtype=np.float32)
        flat = coords.flatten()
        # Pad or trim to output_dim
        if flat.size >= self.output_dim:
            return flat[: self.output_dim]
        return np.pad(flat, (0, self.output_dim - flat.size))

    def _stub_extract(self, frame: np.ndarray) -> np.ndarray:
        """Original stub behavior — used in tests and when mediapipe unavailable."""
        flat = frame.astype(np.float32).reshape(-1)
        if flat.size == 0:
            return np.zeros(self.output_dim, dtype=np.float32)
        repeated = np.resize(flat, self.output_dim)
        normalized = (repeated - repeated.mean()) / (repeated.std() + 1e-6)
        return normalized.astype(np.float32)

    def close(self) -> None:
        if self._hands is not None:
            self._hands.close()

"""TFLite-backed landmark MLP classifier.

Inputs are 21x3 hand-landmark arrays (in original-image pixel/relative-z
coords, as produced by HandCropper). The classifier normalizes
(wrist-centered, scale-normalized) and runs a tiny MLP.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter

import numpy as np


def _normalize_landmarks(landmarks_21x3: np.ndarray) -> np.ndarray:
    wrist = landmarks_21x3[0:1]
    rel = landmarks_21x3 - wrist
    scale = float(np.linalg.norm(rel[:, :2], axis=1).max())
    if scale < 1e-6:
        scale = 1.0
    rel = rel / scale
    return rel.astype(np.float32).reshape(-1)


@dataclass(slots=True)
class LandmarkPrediction:
    label: str
    confidence: float
    latency_ms: float


@dataclass(slots=True)
class LandmarkClassifier:
    model_path: Path
    labels_path: Path
    threads: int = 2
    labels: list[str] = field(init=False)
    interpreter: object = field(init=False)
    input_details: dict = field(init=False)
    output_details: dict = field(init=False)

    def __post_init__(self) -> None:
        self.labels = [
            line.strip() for line in self.labels_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        Interpreter = None
        try:
            from tflite_runtime.interpreter import Interpreter as _I
            Interpreter = _I
        except ImportError:
            try:
                from ai_edge_litert.interpreter import Interpreter as _I
                Interpreter = _I
            except ImportError:
                import tensorflow as tf
                Interpreter = tf.lite.Interpreter
        try:
            self.interpreter = Interpreter(model_path=str(self.model_path), num_threads=self.threads)
        except TypeError:
            self.interpreter = Interpreter(model_path=str(self.model_path))
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()[0]

    def predict(self, landmarks_21x3: np.ndarray) -> LandmarkPrediction:
        t0 = perf_counter()
        x = _normalize_landmarks(landmarks_21x3)[None, ...].astype(np.float32)
        self.interpreter.set_tensor(self.input_details["index"], x)
        self.interpreter.invoke()
        out = self.interpreter.get_tensor(self.output_details["index"])[0].astype(np.float32)
        if out.min() < -0.01 or abs(out.sum() - 1.0) > 0.05:
            e = np.exp(out - out.max())
            out = e / e.sum()
        idx = int(np.argmax(out))
        return LandmarkPrediction(
            label=self.labels[idx] if idx < len(self.labels) else f"class_{idx}",
            confidence=float(out[idx]),
            latency_ms=(perf_counter() - t0) * 1000.0,
        )

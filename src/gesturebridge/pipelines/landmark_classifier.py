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
    # TFLite path
    interpreter: object = field(init=False, default=None)
    input_details: dict = field(init=False, default=None)
    output_details: dict = field(init=False, default=None)
    # Numpy MLP path (used when model_path ends in .npz)
    _coefs: list = field(init=False, default=None)
    _intercepts: list = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.labels = [
            line.strip() for line in self.labels_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if str(self.model_path).endswith(".npz"):
            self._load_numpy(self.model_path)
        else:
            self._load_tflite(self.model_path)

    def _load_numpy(self, path: Path) -> None:
        data = np.load(path)
        n = int(data["n_layers"])
        self._coefs = [data[f"W{i}"] for i in range(n)]
        self._intercepts = [data[f"b{i}"] for i in range(n)]

    def _load_tflite(self, path: Path) -> None:
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
            self.interpreter = Interpreter(model_path=str(path), num_threads=self.threads)
        except TypeError:
            self.interpreter = Interpreter(model_path=str(path))
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()[0]

    def predict(self, landmarks_21x3: np.ndarray) -> LandmarkPrediction:
        t0 = perf_counter()
        x = _normalize_landmarks(landmarks_21x3).astype(np.float32)
        if self._coefs is not None:
            h = x
            for W, b in zip(self._coefs[:-1], self._intercepts[:-1]):
                h = np.maximum(0.0, h @ W + b)
            logits = h @ self._coefs[-1] + self._intercepts[-1]
            e = np.exp(logits - logits.max())
            out = e / e.sum()
        else:
            inp = x[None, ...]
            self.interpreter.set_tensor(self.input_details["index"], inp)
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

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


def softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


@dataclass(slots=True)
class NearestCentroidClassifier:
    class_centroids: np.ndarray

    @classmethod
    def fit(cls, features: np.ndarray, labels: np.ndarray, num_classes: int) -> "NearestCentroidClassifier":
        centroids = []
        for class_id in range(num_classes):
            mask = labels == class_id
            if np.any(mask):
                centroids.append(features[mask].mean(axis=0))
            else:
                centroids.append(np.zeros(features.shape[1], dtype=np.float32))
        return cls(class_centroids=np.stack(centroids).astype(np.float32))

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        # Negative squared distance as logits.
        diffs = features[:, None, :] - self.class_centroids[None, :, :]
        logits = -np.sum(diffs * diffs, axis=2)
        return softmax(logits)

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(features), axis=1)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, class_centroids=self.class_centroids)

    @classmethod
    def load(cls, path: Path) -> "NearestCentroidClassifier":
        data = np.load(path)
        return cls(class_centroids=data["class_centroids"].astype(np.float32))


@dataclass(slots=True)
class QuantizedCentroidClassifier:
    scale: float
    zero_point: int
    centroids_int8: np.ndarray

    @classmethod
    def from_float(cls, model: NearestCentroidClassifier) -> "QuantizedCentroidClassifier":
        centroids = model.class_centroids
        max_abs = float(np.max(np.abs(centroids)))
        scale = max(max_abs / 127.0, 1e-6)
        zero_point = 0
        quantized = np.clip(np.round(centroids / scale), -128, 127).astype(np.int8)
        return cls(scale=scale, zero_point=zero_point, centroids_int8=quantized)

    def _dequantize(self) -> np.ndarray:
        return (self.centroids_int8.astype(np.float32) - self.zero_point) * self.scale

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        centroids = self._dequantize()
        diffs = features[:, None, :] - centroids[None, :, :]
        logits = -np.sum(diffs * diffs, axis=2)
        return softmax(logits)

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(features), axis=1)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            scale=np.array([self.scale], dtype=np.float32),
            zero_point=np.array([self.zero_point], dtype=np.int32),
            centroids_int8=self.centroids_int8,
        )

    @classmethod
    def load(cls, path: Path) -> "QuantizedCentroidClassifier":
        data = np.load(path)
        return cls(
            scale=float(data["scale"][0]),
            zero_point=int(data["zero_point"][0]),
            centroids_int8=data["centroids_int8"].astype(np.int8),
        )

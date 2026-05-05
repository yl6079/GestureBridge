"""Numpy-only WLASL-100 word classifier — Pi-friendly inference.

Loads `artifacts/wlasl100/conv1d_small.npz` (produced by
`scripts/train_wlasl100_pose.py`) and runs the Conv1D forward pass with
pure numpy. No TensorFlow / PyTorch dependency at runtime; matches the
philosophy of the existing landmark MLP (npz + numpy).

Architecture (must match `train_wlasl100_pose.build_conv1d_small`):
    Conv1D(64, k=3, same, ReLU)
    Conv1D(64, k=3, same, ReLU)
    MaxPool1D(2)
    Conv1D(128, k=3, same, ReLU)
    GlobalAveragePool1D
    Dense(128, ReLU)
    Dense(n_classes, softmax)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _conv1d_same(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    """SAME-padded 1D convolution. x:(T,Cin), w:(K,Cin,Cout), b:(Cout,)."""
    K, _Cin, Cout = w.shape
    pad = (K - 1) // 2
    xp = np.pad(x, ((pad, pad), (0, 0)))
    T = x.shape[0]
    # (T, K, Cin) sliding windows
    cols = np.stack([xp[i : i + T] for i in range(K)], axis=1)
    # (T, Cout)
    return np.einsum("tki,kio->to", cols, w) + b


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _maxpool1d(x: np.ndarray, pool: int = 2) -> np.ndarray:
    T = x.shape[0] - x.shape[0] % pool
    return x[:T].reshape(T // pool, pool, -1).max(axis=1)


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max()
    e = np.exp(z)
    return e / e.sum()


@dataclass(slots=True)
class WordClassifier:
    """Conv1D-Small classifier loaded from a single npz weights file."""

    model_path: Path
    labels_path: Path
    _weights: dict = None  # type: ignore[assignment]
    _labels: list = None  # type: ignore[assignment]
    _input_shape: tuple = (30, 63)
    _n_classes: int = 0

    def __post_init__(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(f"word model not found: {self.model_path}")
        if not self.labels_path.exists():
            raise FileNotFoundError(f"word labels not found: {self.labels_path}")
        d = np.load(self.model_path, allow_pickle=True)
        self._weights = {k: d[k] for k in d.keys() if not k.startswith("__")}
        ish = d["__input_shape__"]
        self._input_shape = (int(ish[0]), int(ish[1]))
        self._n_classes = int(d["__n_classes__"][0])
        self._labels = [
            line.strip() for line in self.labels_path.read_text(encoding="utf-8").splitlines() if line.strip()
        ]
        if len(self._labels) != self._n_classes:
            raise ValueError(
                f"labels count {len(self._labels)} != n_classes {self._n_classes}"
            )

    @property
    def input_shape(self) -> tuple[int, int]:
        return self._input_shape

    @property
    def labels(self) -> list[str]:
        return list(self._labels)

    def _forward(self, x: np.ndarray) -> np.ndarray:
        """x: (T, 63). Returns logits over n_classes."""
        w = self._weights
        # conv1d (k=3, in=63, out=64)
        h = _relu(_conv1d_same(x, w["conv1d__0"], w["conv1d__1"]))
        # conv1d_1 (k=3, 64->64)
        h = _relu(_conv1d_same(h, w["conv1d_1__0"], w["conv1d_1__1"]))
        # maxpool 2
        h = _maxpool1d(h, 2)
        # conv1d_2 (k=3, 64->128)
        h = _relu(_conv1d_same(h, w["conv1d_2__0"], w["conv1d_2__1"]))
        # global avg pool
        h = h.mean(axis=0)
        # dense (128->128) ReLU
        h = _relu(h @ w["dense__0"] + w["dense__1"])
        # dense_1 (128->100) — return as logits; caller may apply softmax
        return h @ w["dense_1__0"] + w["dense_1__1"]

    def predict(self, sequence: np.ndarray, top_k: int = 5) -> list[tuple[str, float]]:
        """Predict from a (T, 63) landmark sequence; returns top-k (label, prob)."""
        if sequence.shape != self._input_shape:
            raise ValueError(
                f"expected input {self._input_shape}, got {sequence.shape}"
            )
        logits = self._forward(sequence.astype(np.float32))
        probs = _softmax(logits)
        idx = np.argsort(-probs)[:top_k]
        return [(self._labels[int(i)], float(probs[int(i)])) for i in idx]

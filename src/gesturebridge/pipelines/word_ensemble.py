"""Pure-numpy ensemble: Conv1D + GRU on WLASL-100 landmark sequences.

Loads both `conv1d_small.npz` and `gru_small.npz` (trained by
`scripts/train_wlasl100_pose.py`) and averages their softmax outputs.
Empirically: Conv1D alone test top-1 = 52.7 %, GRU alone = 50.2 %,
ensemble (0.5 / 0.5) = 57.7 %, top-5 = 87.0 % on the official Kaggle
WLASL-100 test split.

Pi-friendly: zero external ML deps at runtime, just numpy.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gesturebridge.pipelines.word_classifier import WordClassifier, _softmax


def _sigmoid(x: np.ndarray) -> np.ndarray:
    # Numerically-stable sigmoid.
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    e = np.exp(x[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def _gru_forward(x_seq: np.ndarray, w_xh: np.ndarray, w_hh: np.ndarray, biases: np.ndarray,
                 return_sequences: bool) -> np.ndarray:
    """Numpy GRU matching Keras `reset_after=True` (its default).

    x_seq: (T, C_in)
    w_xh: (C_in, 3*H)            input kernel
    w_hh: (H,    3*H)            recurrent kernel
    biases: (2, 3*H)             input bias and recurrent bias rows
    Returns (T, H) or (H,) depending on return_sequences.
    """
    T, _ = x_seq.shape
    H = w_xh.shape[1] // 3
    b_x, b_h = biases[0], biases[1]
    h = np.zeros(H, dtype=np.float32)
    out_seq = np.zeros((T, H), dtype=np.float32) if return_sequences else None

    for t in range(T):
        x = x_seq[t]
        x_g = x @ w_xh + b_x        # (3H,)
        h_g = h @ w_hh + b_h        # (3H,)
        zx, rx, nx = x_g[:H], x_g[H:2*H], x_g[2*H:]
        zh, rh, nh = h_g[:H], h_g[H:2*H], h_g[2*H:]
        z = _sigmoid(zx + zh)
        r = _sigmoid(rx + rh)
        # reset_after=True: hidden gate combines (input, r*recurrent) AFTER the matmul.
        n = np.tanh(nx + r * nh)
        h = (1.0 - z) * n + z * h
        if return_sequences:
            out_seq[t] = h
    return out_seq if return_sequences else h


@dataclass(slots=True)
class GRUClassifier:
    model_path: Path
    labels_path: Path
    _weights: dict = None  # type: ignore[assignment]
    _labels: list = None  # type: ignore[assignment]
    _input_shape: tuple = (30, 63)
    _n_classes: int = 0

    def __post_init__(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(f"GRU model not found: {self.model_path}")
        d = np.load(self.model_path, allow_pickle=True)
        self._weights = {k: d[k] for k in d.keys() if not k.startswith("__")}
        ish = d["__input_shape__"]
        self._input_shape = (int(ish[0]), int(ish[1]))
        self._n_classes = int(d["__n_classes__"][0])
        self._labels = [
            line.strip() for line in self.labels_path.read_text(encoding="utf-8").splitlines() if line.strip()
        ]

    @property
    def labels(self) -> list[str]:
        return list(self._labels)

    def forward_logits(self, x: np.ndarray) -> np.ndarray:
        w = self._weights
        # gru (in=63, out=64, return_sequences=True)
        h1 = _gru_forward(x, w["gru__0"], w["gru__1"], w["gru__2"], return_sequences=True)
        # gru_1 (in=64, out=64, return_sequences=False)
        h2 = _gru_forward(h1, w["gru_1__0"], w["gru_1__1"], w["gru_1__2"], return_sequences=False)
        # dense (64 -> 128) ReLU
        d1 = h2 @ w["dense__0"] + w["dense__1"]
        d1 = np.maximum(0, d1)
        # dense_1 (128 -> n_classes), no activation; caller applies softmax
        return d1 @ w["dense_1__0"] + w["dense_1__1"]

    def predict(self, sequence: np.ndarray, top_k: int = 5) -> list[tuple[str, float]]:
        logits = self.forward_logits(sequence.astype(np.float32))
        probs = _softmax(logits)
        idx = np.argsort(-probs)[:top_k]
        return [(self._labels[int(i)], float(probs[int(i)])) for i in idx]


@dataclass(slots=True)
class EnsembleWordClassifier:
    """Average softmax of Conv1D + GRU. Same `predict()` interface as
    `WordClassifier` so it drops into MainRuntime without changes."""
    conv: WordClassifier
    gru: GRUClassifier
    weight_conv: float = 0.5

    @property
    def labels(self) -> list[str]:
        return self.conv.labels

    @property
    def input_shape(self) -> tuple[int, int]:
        return self.conv.input_shape

    def predict(self, sequence: np.ndarray, top_k: int = 5) -> list[tuple[str, float]]:
        x = sequence.astype(np.float32)
        l1 = self.conv._forward(x)
        l2 = self.gru.forward_logits(x)
        p1 = _softmax(l1)
        p2 = _softmax(l2)
        probs = self.weight_conv * p1 + (1.0 - self.weight_conv) * p2
        idx = np.argsort(-probs)[:top_k]
        return [(self.conv.labels[int(i)], float(probs[int(i)])) for i in idx]

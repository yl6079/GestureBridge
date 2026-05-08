"""Numpy-only forward pass for BigConv1D.

Mirrors the PyTorch architecture in `scripts/train_conv1d_a100.py`:
three residual ConvBlocks (63→96→128→192), a temporal attention pool,
and a two-layer classifier head. BatchNorm runs in eval mode using the
stored running statistics. Loaded from a single npz produced by
`scripts/export_bigconv1d_to_npz.py`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gesturebridge.pipelines.word_classifier import _softmax


_BN_EPS = 1e-5


def _conv1d_same(x: np.ndarray, w: np.ndarray, b: np.ndarray, stride: int = 1) -> np.ndarray:
    """1D convolution with SAME padding. x:(C_in, T), w:(C_out, C_in, K), b:(C_out,) -> (C_out, T_out)."""
    C_out, C_in, K = w.shape
    pad = (K - 1) // 2
    xp = np.pad(x, ((0, 0), (pad, pad)))  # (C_in, T+2pad)
    T = x.shape[1]
    # build (T, K, C_in) -> reduce to (T, C_out)
    cols = np.stack([xp[:, i:i + T] for i in range(K)], axis=1)  # (C_in, K, T)
    # Reorder to (T, K, C_in) and contract with w (C_out, C_in, K) over (K, C_in).
    cols = np.transpose(cols, (2, 1, 0))  # (T, K, C_in)
    out = np.einsum("tki,oik->to", cols, w) + b  # (T, C_out)
    if stride > 1:
        out = out[::stride]
    return out.T  # (C_out, T_out)


def _bn_eval(x: np.ndarray, weight: np.ndarray, bias: np.ndarray,
             running_mean: np.ndarray, running_var: np.ndarray) -> np.ndarray:
    """BatchNorm 1D in eval mode. x: (C, T)."""
    inv = 1.0 / np.sqrt(running_var + _BN_EPS)
    return (x - running_mean[:, None]) * (weight[:, None] * inv[:, None]) + bias[:, None]


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _conv_block(x: np.ndarray, w: dict, prefix: str) -> np.ndarray:
    """Run one ConvBlock. x: (C_in, T) -> (C_out, T)."""
    s = x
    # c1 + bn1 + relu
    h = _conv1d_same(x, w[f"{prefix}.c1.weight"], w[f"{prefix}.c1.bias"])
    h = _bn_eval(h, w[f"{prefix}.bn1.weight"], w[f"{prefix}.bn1.bias"],
                 w[f"{prefix}.bn1.running_mean"], w[f"{prefix}.bn1.running_var"])
    h = _relu(h)
    # c2 + bn2
    h = _conv1d_same(h, w[f"{prefix}.c2.weight"], w[f"{prefix}.c2.bias"])
    h = _bn_eval(h, w[f"{prefix}.bn2.weight"], w[f"{prefix}.bn2.bias"],
                 w[f"{prefix}.bn2.running_mean"], w[f"{prefix}.bn2.running_var"])
    # skip
    skip_w_key = f"{prefix}.skip.weight"
    if skip_w_key in w:
        s = _conv1d_same(s, w[skip_w_key], w[f"{prefix}.skip.bias"])
    # else: identity (in_ch == out_ch)
    return _relu(h + s)


@dataclass(slots=True)
class BigConv1DClassifier:
    model_path: Path
    labels_path: Path
    _weights: dict = None  # type: ignore[assignment]
    _labels: list = None  # type: ignore[assignment]
    _n_classes: int = 0
    _input_shape: tuple = (30, 63)

    def __post_init__(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(f"BigConv1D npz not found: {self.model_path}")
        d = np.load(self.model_path, allow_pickle=True)
        self._weights = {k: d[k] for k in d.keys() if not k.startswith("__")}
        self._n_classes = int(d["__num_classes__"][0])
        ish = d["__input_shape__"]
        self._input_shape = (int(ish[0]), int(ish[1]))
        self._labels = [
            line.strip() for line in self.labels_path.read_text(encoding="utf-8").splitlines() if line.strip()
        ]

    @property
    def labels(self) -> list[str]:
        return list(self._labels)

    @property
    def input_shape(self) -> tuple[int, int]:
        return self._input_shape

    def forward_logits(self, sequence: np.ndarray) -> np.ndarray:
        """sequence: (T, C). Returns (n_classes,)."""
        w = self._weights
        # PyTorch model takes (B, T, C) and transposes to (B, C, T) inside forward.
        x = sequence.astype(np.float32).T  # (C, T)
        x = _conv_block(x, w, "b1")
        x = _conv_block(x, w, "b2")
        x = _conv_block(x, w, "b3")
        # attention pool over time
        # h = x.transpose: (T, C); a = h @ attn_w + attn_b -> (T, 1); softmax over T; weighted sum
        h = x.T  # (T, C)
        attn_w = w["attn.weight"]  # (1, C)
        attn_b = w["attn.bias"]    # (1,)
        a = h @ attn_w.T + attn_b  # (T, 1)
        a_max = a.max()
        ea = np.exp(a - a_max)
        weights = ea / ea.sum()  # (T, 1)
        z = (h * weights).sum(axis=0)  # (C,)
        # fc: Sequential(Dropout, Linear(C, 128), ReLU, Dropout, Linear(128, num_classes))
        # In PyTorch state_dict the indices are fc.0 (no params, dropout), fc.1, fc.2 (no params), fc.3, fc.4
        # Linear weight shape: (out, in)
        z = z @ w["fc.1.weight"].T + w["fc.1.bias"]
        z = _relu(z)
        z = z @ w["fc.4.weight"].T + w["fc.4.bias"]
        return z

    def predict(self, sequence: np.ndarray, top_k: int = 5) -> list[tuple[str, float]]:
        if sequence.shape != self._input_shape:
            raise ValueError(f"expected input {self._input_shape}, got {sequence.shape}")
        logits = self.forward_logits(sequence.astype(np.float32))
        probs = _softmax(logits)
        idx = np.argsort(-probs)[:top_k]
        return [(self._labels[int(i)], float(probs[int(i)])) for i in idx]

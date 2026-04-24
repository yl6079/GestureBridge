from __future__ import annotations

from pathlib import Path

import numpy as np


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def generate_synthetic_dataset(
    output_path: Path,
    num_samples: int,
    feature_dim: int,
    num_classes: int,
    seed: int,
    class_centers: np.ndarray | None = None,
) -> None:
    rng = np.random.default_rng(seed)
    if class_centers is None:
        class_centers = rng.normal(0, 2.0, size=(num_classes, feature_dim)).astype(np.float32)
    labels = rng.integers(0, num_classes, size=(num_samples,), endpoint=False, dtype=np.int64)
    noise = rng.normal(0, 0.7, size=(num_samples, feature_dim)).astype(np.float32)
    features = class_centers[labels] + noise
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, x=features.astype(np.float32), y=labels.astype(np.int64))


def load_dataset(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    return data["x"].astype(np.float32), data["y"].astype(np.int64)

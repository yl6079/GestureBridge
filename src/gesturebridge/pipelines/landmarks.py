from __future__ import annotations

import numpy as np


class LandmarkExtractor:
    """Simulated MediaPipe-like landmark extractor.

    Input frame can be any numeric ndarray. The extractor returns a stable
    63-dim vector as placeholder for 21 hand landmarks x (x,y,z).
    """

    def __init__(self, output_dim: int = 63) -> None:
        self.output_dim = output_dim

    def extract(self, frame: np.ndarray) -> np.ndarray:
        flat = frame.astype(np.float32).reshape(-1)
        if flat.size == 0:
            return np.zeros(self.output_dim, dtype=np.float32)
        repeated = np.resize(flat, self.output_dim)
        normalized = (repeated - repeated.mean()) / (repeated.std() + 1e-6)
        return normalized.astype(np.float32)

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np

from gesturebridge.config import SystemConfig
from gesturebridge.data import load_dataset
from gesturebridge.pipelines.classifier import QuantizedCentroidClassifier


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    mat = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred, strict=True):
        mat[t, p] += 1
    return mat


def main() -> None:
    cfg = SystemConfig()
    x_test, y_test = load_dataset(cfg.data.test_path)
    model = QuantizedCentroidClassifier.load(cfg.model.quantized_model_path)

    start = perf_counter()
    y_pred = model.predict(x_test)
    elapsed = perf_counter() - start
    acc = float((y_pred == y_test).mean())
    latency_ms = (elapsed / max(len(x_test), 1)) * 1000.0

    cm = confusion_matrix(y_test, y_pred, cfg.model.num_classes)
    output = {
        "accuracy": acc,
        "avg_inference_ms_per_sample": latency_ms,
        "num_classes": cfg.model.num_classes,
        "confusion_matrix": cm.tolist(),
    }

    out_path = Path("artifacts/evaluation_report.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))
    print(f"Saved report to {out_path}")


if __name__ == "__main__":
    main()

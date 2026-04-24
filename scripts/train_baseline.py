from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np

from gesturebridge.config import SystemConfig
from gesturebridge.data import generate_synthetic_dataset, load_dataset
from gesturebridge.pipelines.classifier import NearestCentroidClassifier, QuantizedCentroidClassifier


def accuracy(pred: np.ndarray, true: np.ndarray) -> float:
    return float((pred == true).mean())


def main() -> None:
    cfg = SystemConfig()
    cfg.data.dataset_dir.mkdir(parents=True, exist_ok=True)
    centers_path = cfg.data.dataset_dir / "class_centers.npy"
    if centers_path.exists():
        class_centers = np.load(centers_path).astype(np.float32)
    else:
        rng = np.random.default_rng(5)
        class_centers = rng.normal(
            0,
            2.0,
            size=(cfg.model.num_classes, cfg.model.feature_dim),
        ).astype(np.float32)
        np.save(centers_path, class_centers)

    generate_synthetic_dataset(
        cfg.data.train_path,
        1200,
        cfg.model.feature_dim,
        cfg.model.num_classes,
        seed=7,
        class_centers=class_centers,
    )
    generate_synthetic_dataset(
        cfg.data.val_path,
        300,
        cfg.model.feature_dim,
        cfg.model.num_classes,
        seed=17,
        class_centers=class_centers,
    )
    generate_synthetic_dataset(
        cfg.data.test_path,
        300,
        cfg.model.feature_dim,
        cfg.model.num_classes,
        seed=23,
        class_centers=class_centers,
    )

    x_train, y_train = load_dataset(cfg.data.train_path)
    x_val, y_val = load_dataset(cfg.data.val_path)
    x_test, y_test = load_dataset(cfg.data.test_path)

    float_model = NearestCentroidClassifier.fit(x_train, y_train, cfg.model.num_classes)
    float_model.save(cfg.model.model_path)

    q_model = QuantizedCentroidClassifier.from_float(float_model)
    q_model.save(cfg.model.quantized_model_path)

    train_acc = accuracy(float_model.predict(x_train), y_train)
    val_acc_float = accuracy(float_model.predict(x_val), y_val)
    val_acc_q = accuracy(q_model.predict(x_val), y_val)
    test_acc_q = accuracy(q_model.predict(x_test), y_test)

    report_dir = Path("artifacts")
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "training_report.txt"
    report = "\n".join(
        [
            "GestureBridge baseline training report",
            f"train_accuracy_float={train_acc:.4f}",
            f"val_accuracy_float={val_acc_float:.4f}",
            f"val_accuracy_int8={val_acc_q:.4f}",
            f"test_accuracy_int8={test_acc_q:.4f}",
        ]
    )
    report_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved report to {report_path}")


if __name__ == "__main__":
    main()

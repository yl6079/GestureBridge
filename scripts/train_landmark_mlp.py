"""Train the 63-d landmark MLP on extracted ASL29 landmarks.

Reads NPZ files produced by `scripts/extract_landmarks.py` and trains a
small MLP. Filters out rows with detected=0 (no hand found by MediaPipe)
since the landmark vector is meaningless there.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _load(npz_path: Path, drop_undetected: bool) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(npz_path)
    X, y, det = data["X"], data["y"], data["detected"]
    if drop_undetected:
        mask = det.astype(bool)
        return X[mask], y[mask]
    return X, y


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--landmarks-dir", type=Path, default=Path("artifacts/asl29/landmarks"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/asl29/landmark_mlp"))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    import tensorflow as tf
    from gesturebridge.ml.models.landmark_mlp import build_landmark_mlp

    args.out_dir.mkdir(parents=True, exist_ok=True)

    Xt, yt = _load(args.landmarks_dir / "train.npz", drop_undetected=True)
    Xv, yv = _load(args.landmarks_dir / "val.npz", drop_undetected=True)
    Xs, ys = _load(args.landmarks_dir / "test.npz", drop_undetected=True)
    print(f"shapes: train {Xt.shape} val {Xv.shape} test {Xs.shape}")

    num_classes = int(max(int(yt.max()), int(yv.max()), int(ys.max())) + 1)
    model = build_landmark_mlp(num_classes=num_classes, dropout=args.dropout)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    print(model.summary())

    cb = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5),
    ]
    history = model.fit(
        Xt, yt,
        validation_data=(Xv, yv),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=cb,
        verbose=2,
    )

    test_loss, test_acc = model.evaluate(Xs, ys, verbose=0)
    print(f"\nTest accuracy on detected hands: {test_acc:.4f}")

    keras_path = args.out_dir / "best.keras"
    model.save(keras_path)

    # Also export to TFLite (tiny, useful for the Pi)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_bytes = converter.convert()
    tflite_path = args.out_dir / "landmark_mlp.tflite"
    tflite_path.write_bytes(tflite_bytes)

    metrics = {
        "test_accuracy_on_detected": float(test_acc),
        "test_loss_on_detected": float(test_loss),
        "n_train": int(len(yt)),
        "n_val": int(len(yv)),
        "n_test": int(len(ys)),
        "best_val_accuracy": float(max(history.history.get("val_accuracy", [0.0]))),
        "keras_path": str(keras_path),
        "tflite_path": str(tflite_path),
        "tflite_size_bytes": int(tflite_path.stat().st_size),
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

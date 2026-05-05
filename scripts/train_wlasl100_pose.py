"""Train a Conv1D-Small word classifier on WLASL-100 landmark sequences.

Reads `data/wlasl100/landmarks.npz` (produced by extract_wlasl_landmarks.py)
and trains a tiny temporal Conv1D on (T, 63) landmark windows. Exports two
artifacts:
  - artifacts/wlasl100/conv1d_v1.keras       (full Keras model, dev)
  - artifacts/wlasl100/conv1d_v1.npz         (weights only, numpy-only Pi inference)
  - artifacts/wlasl100/eval.json             (val/test top-1 / top-5)
  - artifacts/wlasl100/labels.txt            (gloss list)

Usage:
    python scripts/train_wlasl100_pose.py
    python scripts/train_wlasl100_pose.py --arch gru_small
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]


def build_conv1d_small(t: int, c: int, n_classes: int) -> tf.keras.Model:
    """~50K-param Conv1D over (T, 63) landmark sequences."""
    inputs = tf.keras.Input(shape=(t, c), name="landmarks")
    x = tf.keras.layers.Conv1D(64, 3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.Conv1D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPool1D(2)(x)
    x = tf.keras.layers.Conv1D(128, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    outputs = tf.keras.layers.Dense(n_classes, activation="softmax")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="conv1d_small")


def build_gru_small(t: int, c: int, n_classes: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(t, c), name="landmarks")
    x = tf.keras.layers.GRU(64, return_sequences=True)(inputs)
    x = tf.keras.layers.GRU(64)(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    outputs = tf.keras.layers.Dense(n_classes, activation="softmax")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="gru_small")


ARCHES = {
    "conv1d_small": build_conv1d_small,
    "gru_small": build_gru_small,
}


def topk_accuracy(probs: np.ndarray, y: np.ndarray, k: int) -> float:
    topk = np.argsort(-probs, axis=1)[:, :k]
    return float((topk == y[:, None]).any(axis=1).mean())


def augment_batch(X: tf.Tensor) -> tf.Tensor:
    """Train-time augmentation: temporal jitter + spatial scale."""
    # Spatial scale 0.9 - 1.1
    scale = tf.random.uniform([tf.shape(X)[0], 1, 1], 0.9, 1.1)
    X = X * scale
    # Temporal jitter ±2 frames (circular shift)
    shift = tf.random.uniform([], -2, 3, dtype=tf.int32)
    X = tf.roll(X, shift=shift, axis=1)
    return X


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/wlasl100/landmarks.npz"))
    ap.add_argument("--labels", type=Path, default=Path("data/wlasl100/labels.txt"))
    ap.add_argument("--out-dir", type=Path, default=Path("artifacts/wlasl100"))
    ap.add_argument("--arch", choices=list(ARCHES), default="conv1d_small")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    d = np.load(args.data, allow_pickle=True)
    X, y, sp = d["X"].astype(np.float32), d["y"].astype(np.int64), d["split"]
    labels = [g.strip() for g in args.labels.read_text(encoding="utf-8").splitlines() if g.strip()]
    n_classes = len(labels)

    train_idx = np.where(sp == 0)[0]
    val_idx = np.where(sp == 1)[0]
    test_idx = np.where(sp == 2)[0]
    print(f"[train] X={X.shape} train={len(train_idx)} val={len(val_idx)} test={len(test_idx)} classes={n_classes}")

    Xtr, ytr = X[train_idx], y[train_idx]
    Xva, yva = X[val_idx], y[val_idx]
    Xte, yte = X[test_idx], y[test_idx]

    def make_ds(Xs, ys, training: bool):
        ds = tf.data.Dataset.from_tensor_slices((Xs, ys))
        if training:
            ds = ds.shuffle(len(Xs), seed=args.seed)
            ds = ds.batch(args.batch).map(lambda Xb, yb: (augment_batch(Xb), yb), num_parallel_calls=tf.data.AUTOTUNE)
        else:
            ds = ds.batch(args.batch)
        return ds.prefetch(tf.data.AUTOTUNE)

    train_ds = make_ds(Xtr, ytr, training=True)
    val_ds = make_ds(Xva, yva, training=False)
    test_ds = make_ds(Xte, yte, training=False)

    model = ARCHES[args.arch](X.shape[1], X.shape[2], n_classes)
    model.summary()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="top1"),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5"),
        ],
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_top1", mode="max", patience=15, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_top1", mode="max", factor=0.5, patience=6, min_lr=1e-5),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=2,
    )

    # Eval
    def report(name, ds, ys):
        probs = model.predict(ds, verbose=0)
        top1 = topk_accuracy(probs, ys, 1)
        top5 = topk_accuracy(probs, ys, 5)
        print(f"[{name}] top1={top1:.4f} top5={top5:.4f}")
        return {"top1": top1, "top5": top5, "n": int(len(ys))}

    metrics = {
        "arch": args.arch,
        "epochs_run": int(model.history.epoch[-1] + 1) if hasattr(model, "history") else args.epochs,
        "train": report("train", make_ds(Xtr, ytr, training=False), ytr),
        "val": report("val", val_ds, yva),
        "test": report("test", test_ds, yte),
        "n_classes": n_classes,
        "labels_path": str(args.labels),
        "data_path": str(args.data),
    }

    keras_path = args.out_dir / f"{args.arch}.keras"
    model.save(keras_path)
    print(f"[train] saved keras model: {keras_path}")

    # Export weights as a single npz so the Pi can do pure-numpy inference.
    npz_path = args.out_dir / f"{args.arch}.npz"
    weights: dict[str, np.ndarray] = {}
    for layer in model.layers:
        for i, w in enumerate(layer.get_weights()):
            weights[f"{layer.name}__{i}"] = w
    weights["__arch__"] = np.array([args.arch], dtype="<U32")
    weights["__input_shape__"] = np.array(X.shape[1:], dtype=np.int32)
    weights["__n_classes__"] = np.array([n_classes], dtype=np.int32)
    np.savez_compressed(npz_path, **weights)
    print(f"[train] saved weights: {npz_path}")

    # Copy labels next to the model.
    out_labels = args.out_dir / "labels.txt"
    out_labels.write_text("\n".join(labels) + "\n", encoding="utf-8")

    eval_path = args.out_dir / "eval.json"
    eval_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[train] eval: {eval_path}")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

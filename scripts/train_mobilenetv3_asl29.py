from __future__ import annotations

import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import tensorflow as tf

from gesturebridge.config import SystemConfig
from gesturebridge.ml.data_pipeline import build_dataset
from gesturebridge.ml.models.mobilenetv3 import (
    build_mobilenetv3_small_classifier,
    set_backbone_trainable_layers,
)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def _compile_model(model: tf.keras.Model, learning_rate: float, weight_decay: float) -> None:
    if weight_decay > 0:
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3_accuracy"),
        ],
    )


def _callbacks(model_path: Path, patience: int) -> list[tf.keras.callbacks.Callback]:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=max(patience // 2, 1),
            min_lr=1e-6,
            verbose=1,
        ),
    ]


def _history_to_serializable(history: tf.keras.callbacks.History) -> dict[str, list[float]]:
    return {key: [float(v) for v in values] for key, values in history.history.items()}


def main() -> None:
    cfg = SystemConfig().asl29
    _seed_everything(cfg.training.random_seed)

    train_ds = build_dataset(
        csv_path=cfg.data.train_csv,
        image_size=cfg.data.image_size,
        batch_size=cfg.training.batch_size,
        training=True,
        shuffle_seed=cfg.training.random_seed,
    )
    val_ds = build_dataset(
        csv_path=cfg.data.val_csv,
        image_size=cfg.data.image_size,
        batch_size=cfg.training.batch_size,
        training=False,
        shuffle_seed=cfg.training.random_seed,
    )

    model = build_mobilenetv3_small_classifier(
        image_size=cfg.data.image_size,
        num_classes=cfg.num_classes,
        dropout=cfg.training.dropout,
        train_backbone=False,
    )

    _compile_model(
        model,
        learning_rate=cfg.training.frozen_learning_rate,
        weight_decay=cfg.training.weight_decay,
    )
    cb = _callbacks(cfg.training.model_path, cfg.training.patience)
    frozen_history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.training.frozen_epochs,
        callbacks=cb,
        verbose=2,
    )

    set_backbone_trainable_layers(model, unfreeze_layers=cfg.training.unfreeze_layers)
    _compile_model(
        model,
        learning_rate=cfg.training.finetune_learning_rate,
        weight_decay=cfg.training.weight_decay,
    )
    finetune_history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.training.finetune_epochs,
        callbacks=cb,
        verbose=2,
    )

    cfg.training.final_model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(cfg.training.final_model_path)

    frozen_metrics = _history_to_serializable(frozen_history)
    finetune_metrics = _history_to_serializable(finetune_history)
    history_output = {
        "frozen_phase": frozen_metrics,
        "finetune_phase": finetune_metrics,
    }
    cfg.training.history_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.training.history_path.write_text(json.dumps(history_output, indent=2), encoding="utf-8")

    best_val_accuracy = max(
        max(frozen_metrics.get("val_accuracy", [0.0])),
        max(finetune_metrics.get("val_accuracy", [0.0])),
    )
    metrics = {
        "best_val_accuracy": float(best_val_accuracy),
        "model_path": str(cfg.training.model_path),
        "final_model_path": str(cfg.training.final_model_path),
    }
    cfg.training.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.training.metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(json.dumps(metrics, indent=2))
    print(f"Training history saved to {cfg.training.history_path}")


if __name__ == "__main__":
    main()


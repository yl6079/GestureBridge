from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from gesturebridge.config import SystemConfig
from gesturebridge.ml.data_pipeline import build_dataset, load_class_names, load_manifest


def _plot_confusion_matrix(cm: np.ndarray, labels: list[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    fig.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Predicted",
        ylabel="True",
        title="ASL29 Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=90, ha="center")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _top_confusions(cm: np.ndarray, labels: list[str], top_n: int = 20) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for true_idx in range(cm.shape[0]):
        for pred_idx in range(cm.shape[1]):
            if true_idx == pred_idx:
                continue
            count = int(cm[true_idx, pred_idx])
            if count > 0:
                records.append(
                    {
                        "true_label": labels[true_idx],
                        "pred_label": labels[pred_idx],
                        "count": count,
                    }
                )
    if not records:
        return pd.DataFrame(columns=["true_label", "pred_label", "count"])
    out = pd.DataFrame(records).sort_values(by="count", ascending=False).head(top_n)
    return out.reset_index(drop=True)


def main() -> None:
    cfg = SystemConfig().asl29
    class_names = load_class_names(cfg.data.labels_path)
    if len(class_names) != cfg.num_classes:
        raise RuntimeError(
            f"Class names count ({len(class_names)}) does not match num_classes ({cfg.num_classes})."
        )

    test_manifest = load_manifest(cfg.data.test_csv)
    test_ds = build_dataset(
        csv_path=cfg.data.test_csv,
        image_size=cfg.data.image_size,
        batch_size=cfg.training.batch_size,
        training=False,
        shuffle_seed=cfg.training.random_seed,
    )

    model = tf.keras.models.load_model(cfg.training.model_path)
    probs = model.predict(test_ds, verbose=0)
    y_pred = np.argmax(probs, axis=1).astype(np.int64)
    y_true = test_manifest["label"].to_numpy(dtype=np.int64)

    cm = confusion_matrix(y_true, y_pred, labels=np.arange(cfg.num_classes))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    accuracy = float((y_true == y_pred).mean())
    report = classification_report(
        y_true,
        y_pred,
        labels=np.arange(cfg.num_classes),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    cfg.evaluation.eval_dir.mkdir(parents=True, exist_ok=True)
    _plot_confusion_matrix(cm, class_names, cfg.evaluation.confusion_matrix_png)
    confusions = _top_confusions(cm, class_names, top_n=30)
    confusions.to_csv(cfg.evaluation.misclassifications_csv, index=False)

    output = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "num_samples": int(len(y_true)),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "high_confusions_csv": str(cfg.evaluation.misclassifications_csv),
        "confusion_matrix_png": str(cfg.evaluation.confusion_matrix_png),
    }
    cfg.evaluation.report_json.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(json.dumps({"accuracy": accuracy, "macro_f1": macro_f1}, indent=2))
    print(f"Saved evaluation report to {cfg.evaluation.report_json}")


if __name__ == "__main__":
    main()


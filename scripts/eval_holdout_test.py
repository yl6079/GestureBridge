"""Evaluate a TFLite ASL29 model on the Kaggle-provided 28-image holdout set.

The Kaggle ASL Alphabet ships a separate `asl_alphabet_test/` directory with
one image per class (28 images, no `del`), captured in a different recording
session than the 87k train images. The deployed model has never seen these,
so this script gives a free honest generalization signal — no retraining
required.

Usage:
    python scripts/eval_holdout_test.py \
        --model artifacts/asl29/tflite/model_fp32.tflite \
        --labels artifacts/asl29/labels.txt \
        --test-dir ~/Desktop/Elen6908/data/asl29_raw/asl_alphabet_test/asl_alphabet_test
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from ai_edge_litert.interpreter import Interpreter


def _preprocess(path: Path, size: int) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    # MobileNetV3Small in tf.keras.applications expects raw [0,255]; the
    # rescale to [-1,1] is baked into the model's first layer, and
    # tf.keras.applications.mobilenet_v3.preprocess_input is a no-op.
    arr = img.astype(np.float32)
    return arr[None, ...]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--test-dir", required=True, type=Path)
    parser.add_argument("--out-json", default=Path("artifacts/asl29/eval/holdout_metrics.json"), type=Path)
    args = parser.parse_args()

    labels = [line.strip() for line in args.labels.read_text().splitlines() if line.strip()]
    interp = Interpreter(model_path=str(args.model))
    interp.allocate_tensors()
    in_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]
    in_size = int(in_det["shape"][1])

    rows = []
    for path in sorted(args.test_dir.iterdir()):
        if not path.is_file() or not path.name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        true_class = path.stem.split("_test")[0]
        x = _preprocess(path, in_size).astype(in_det["dtype"])
        interp.set_tensor(in_det["index"], x)
        interp.invoke()
        out = interp.get_tensor(out_det["index"])[0]
        # softmax if not already
        if out.min() < 0 or out.sum() > 1.5 or out.sum() < 0.5:
            e = np.exp(out - out.max())
            out = e / e.sum()
        top_idx = int(np.argmax(out))
        pred = labels[top_idx]
        conf = float(out[top_idx])
        # top-3 for context
        top3 = sorted(enumerate(out.tolist()), key=lambda x: -x[1])[:3]
        rows.append({
            "file": path.name,
            "true": true_class,
            "pred": pred,
            "confidence": conf,
            "correct": pred == true_class,
            "top3": [{"label": labels[i], "p": float(p)} for i, p in top3],
        })

    correct = sum(1 for r in rows if r["correct"])
    total = len(rows)
    acc = correct / total if total else 0.0

    print(f"Held-out test accuracy: {correct}/{total} = {acc:.4f}")
    print()
    print(f"{'TRUE':<10} {'PRED':<10} {'CONF':<8} CORRECT")
    print("-" * 40)
    for r in rows:
        mark = "OK " if r["correct"] else "XX "
        print(f"{r['true']:<10} {r['pred']:<10} {r['confidence']:.3f}    {mark}")

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps({"accuracy": acc, "correct": correct, "total": total, "rows": rows}, indent=2))
    print(f"\nWrote {args.out_json}")


if __name__ == "__main__":
    main()

"""Evaluate a TFLite ASL29 model on a CSV split (path,label,class_name).

Reports overall accuracy, per-class precision/recall/F1, and the top
confusion pairs.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ai_edge_litert.interpreter import Interpreter


def _preprocess(path: str, size: int) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        return np.zeros((1, size, size, 3), dtype=np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    return img.astype(np.float32)[None, ...]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--split-csv", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=0, help="Optional cap on samples (0 = all)")
    parser.add_argument("--out-json", default=Path("artifacts/asl29/eval/split_metrics.json"), type=Path)
    args = parser.parse_args()

    labels = [line.strip() for line in args.labels.read_text().splitlines() if line.strip()]
    label_to_idx = {n: i for i, n in enumerate(labels)}

    df = pd.read_csv(args.split_csv)
    if args.limit > 0:
        df = df.sample(n=min(args.limit, len(df)), random_state=0).reset_index(drop=True)

    interp = Interpreter(model_path=str(args.model))
    interp.allocate_tensors()
    in_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]
    in_size = int(in_det["shape"][1])

    correct = 0
    confusions: Counter[tuple[str, str]] = Counter()
    per_class_tp: Counter[str] = Counter()
    per_class_fp: Counter[str] = Counter()
    per_class_fn: Counter[str] = Counter()
    per_class_n: Counter[str] = Counter()

    for i, row in enumerate(df.itertuples(index=False), 1):
        x = _preprocess(row.path, in_size).astype(in_det["dtype"])
        interp.set_tensor(in_det["index"], x)
        interp.invoke()
        out = interp.get_tensor(out_det["index"])[0]
        pred_idx = int(np.argmax(out))
        pred = labels[pred_idx]
        true = row.class_name
        per_class_n[true] += 1
        if pred == true:
            correct += 1
            per_class_tp[true] += 1
        else:
            confusions[(true, pred)] += 1
            per_class_fp[pred] += 1
            per_class_fn[true] += 1
        if i % 500 == 0:
            print(f"  {i}/{len(df)}  running acc={correct/i:.3f}", flush=True)

    total = len(df)
    acc = correct / total
    per_class = {}
    for c in labels:
        tp = per_class_tp[c]
        fp = per_class_fp[c]
        fn = per_class_fn[c]
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_class[c] = {"support": per_class_n[c], "precision": prec, "recall": rec, "f1": f1}

    print(f"\nOverall accuracy: {correct}/{total} = {acc:.4f}")
    macro_f1 = sum(v["f1"] for v in per_class.values()) / len(per_class)
    print(f"Macro F1: {macro_f1:.4f}")

    print("\nTop confusions (true -> pred : count):")
    for (t, p), c in confusions.most_common(15):
        print(f"  {t:<8} -> {p:<8} : {c}")

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps({
        "accuracy": acc,
        "macro_f1": macro_f1,
        "total": total,
        "correct": correct,
        "per_class": per_class,
        "top_confusions": [{"true": t, "pred": p, "count": c} for (t, p), c in confusions.most_common(50)],
    }, indent=2))
    print(f"\nWrote {args.out_json}")


if __name__ == "__main__":
    main()

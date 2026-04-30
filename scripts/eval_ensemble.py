"""Evaluate MobileNet, landmark MLP, and the ensemble on a split CSV.

Runs each row through MediaPipe HandLandmarker once, then sends the
cropped image to the MobileNet TFLite and the normalized landmarks to
the landmark-MLP TFLite. Reports accuracy for each head plus the
ensemble (using the same decision rule as MainRuntime._maybe_ensemble).

This is the test for P3 — once we have both trained models, this tells
us whether the ensemble actually beats either head alone.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gesturebridge.pipelines.hand_crop import HandCropper
from gesturebridge.pipelines.asl29_tflite import ASL29TFLiteRuntime
from gesturebridge.pipelines.landmark_classifier import LandmarkClassifier


def _ensemble_decision(mobilenet_label: str, mobilenet_conf: float, lm_label: str, lm_conf: float) -> tuple[str, float]:
    if mobilenet_label == lm_label:
        return mobilenet_label, (mobilenet_conf + lm_conf) / 2.0
    if mobilenet_conf >= 0.85 and lm_conf < 0.95:
        return mobilenet_label, mobilenet_conf
    return lm_label, lm_conf


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mobilenet", required=True, type=Path)
    parser.add_argument("--landmark-mlp", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--split-csv", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out-json", default=Path("artifacts/asl29/eval/ensemble_metrics.json"), type=Path)
    args = parser.parse_args()

    df = pd.read_csv(args.split_csv)
    if args.limit > 0:
        df = df.sample(n=min(args.limit, len(df)), random_state=0).reset_index(drop=True)

    cropper = HandCropper(output_size=224, padding_ratio=0.25, min_confidence=0.05)
    mn = ASL29TFLiteRuntime(
        model_path=args.mobilenet,
        labels_path=args.labels,
        use_hand_crop=False,
    )
    lm = LandmarkClassifier(model_path=args.landmark_mlp, labels_path=args.labels)
    # The MobileNet was trained on un-cropped Kaggle images (already
    # hand-filled at 200x200). For a fair Kaggle-test eval, feed the
    # full image. Use MediaPipe only to extract landmarks for the MLP.
    # (At C270 inference time we DO crop — that's the use_hand_crop path
    # in ASL29TFLiteRuntime; this script targets Kaggle eval.)

    counters = {"mobilenet": 0, "landmark": 0, "ensemble": 0}
    no_hand = 0
    confusion: Counter[tuple[str, str]] = Counter()

    for i, row in enumerate(df.itertuples(index=False), 1):
        img = cv2.imread(str(row.path))
        if img is None:
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        crop = cropper.crop(rgb)
        true = row.class_name

        # MobileNet always sees the full image (matches training).
        mn_res = mn.predict(img)
        mn_pred, mn_conf = mn_res.label, mn_res.confidence

        if not crop.found or crop.landmarks is None:
            # No landmarks available — landmark MLP cannot vote. Fall back
            # to MobileNet alone and count this row's ensemble = MobileNet.
            lm_pred, lm_conf = "nothing", 0.0
            no_hand += 1
        else:
            lm_res = lm.predict(crop.landmarks)
            lm_pred, lm_conf = lm_res.label, lm_res.confidence

        ens_label, _ = _ensemble_decision(mn_pred, mn_conf, lm_pred, lm_conf)

        if mn_pred == true: counters["mobilenet"] += 1
        if lm_pred == true: counters["landmark"] += 1
        if ens_label == true: counters["ensemble"] += 1
        else:
            confusion[(true, ens_label)] += 1

        if i % 500 == 0:
            n = i
            print(f"  {i}/{len(df)}  mn={counters['mobilenet']/n:.3f}  lm={counters['landmark']/n:.3f}  ens={counters['ensemble']/n:.3f}", flush=True)

    n = len(df)
    metrics = {
        "n": n,
        "mobilenet_accuracy": counters["mobilenet"] / n,
        "landmark_accuracy": counters["landmark"] / n,
        "ensemble_accuracy": counters["ensemble"] / n,
        "no_hand_detected": no_hand,
        "top_confusions": [{"true": t, "pred": p, "count": c} for (t, p), c in confusion.most_common(20)],
    }
    print()
    print(json.dumps(metrics, indent=2))
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

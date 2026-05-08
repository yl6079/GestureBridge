"""Per-class letter accuracy on the ASL29 contiguous test split.

Loads the deployed letter ensemble (MobileNetV3-Small TFLite + landmark
MLP), runs every test image, and writes a markdown table sorted by
ensemble top-1 to `artifacts/asl29/eval/per_class_letter_report.md`.

Usage:
    python scripts/per_class_letter_report.py
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gesturebridge.config import SystemConfig
from gesturebridge.pipelines.asl29_tflite import ASL29TFLiteRuntime
from gesturebridge.pipelines.landmark_classifier import LandmarkClassifier


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-csv", type=Path, default=Path("data/asl29/splits/test.csv"))
    ap.add_argument(
        "--out-md", type=Path, default=Path("artifacts/asl29/eval/per_class_letter_report.md")
    )
    ap.add_argument(
        "--out-json", type=Path, default=Path("artifacts/asl29/eval/per_class_letter_report.json")
    )
    ap.add_argument("--max-per-class", type=int, default=300)
    args = ap.parse_args()

    cfg = SystemConfig()
    print("[letter] loading models …")
    # match scripts/extract_landmarks.py — Kaggle's tight 200x200 crops trip
    # MediaPipe at the runtime default 0.3 confidence but pass cleanly at
    # 0.05. We're evaluating accuracy here, not measuring real-world
    # detection rate, so lower the gate.
    infer = ASL29TFLiteRuntime(
        model_path=Path(cfg.asl29.export.fp32_tflite_path),
        labels_path=Path(cfg.asl29.data.labels_path),
        threads=cfg.asl29.runtime.tflite_threads,
        image_size=cfg.asl29.data.image_size,
        top_k=cfg.asl29.runtime.preview_top_k,
        use_hand_crop=cfg.asl29.runtime.use_hand_crop,
        hand_cropper_model_path=Path(cfg.asl29.runtime.hand_landmarker_path),
        hand_min_confidence=0.05,
    )
    lm = LandmarkClassifier(
        model_path=Path("artifacts/asl29/landmark_mlp/landmark_mlp.npz"),
        labels_path=Path(cfg.asl29.data.labels_path),
    )

    counters: dict[str, dict] = defaultdict(
        lambda: {"n": 0, "top1_mn": 0, "top1_lm": 0, "top1_ens": 0, "true_conf_mn": 0.0, "true_conf_ens": 0.0, "no_hand": 0}
    )

    rows = []
    with args.split_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    print(f"[letter] {len(rows)} rows in test split")

    # cap per-class to keep runtime reasonable
    by_class: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_class[r["class_name"]].append(r)
    if args.max_per_class > 0:
        for k in by_class:
            by_class[k] = by_class[k][: args.max_per_class]
    flat_rows = [r for v in by_class.values() for r in v]
    print(f"[letter] capped to {len(flat_rows)} rows for per-class report")

    labels = infer.labels
    label_to_idx = {l: i for i, l in enumerate(labels)}

    for i, r in enumerate(flat_rows):
        path = r["path"]
        true_lbl = r["class_name"]
        true_idx = label_to_idx.get(true_lbl, -1)
        img = cv2.imread(path)
        if img is None:
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        out = infer.predict(rgb)
        # Top-K from MN
        mn_top1 = out.label
        mn_conf = float(out.confidence)
        # Build full softmax: TFLite runtime gives top_k only; for per-class
        # confidence we use top_k probabilities (top-K is already 3); fall
        # back to mn_conf for top-1.
        true_conf_mn = next((pr for (lbl, pr) in out.top_k if lbl == true_lbl), 0.0)

        # Landmark classifier on detected hand
        lm_top1 = mn_top1
        if out.landmarks is not None:
            lp = lm.predict(out.landmarks)
            lm_top1 = lp.label

        # Ensemble (mirror MainRuntime._maybe_ensemble logic)
        if out.landmarks is None:
            ens_top1, ens_conf = mn_top1, mn_conf
        else:
            lm_pred = lm.predict(out.landmarks)
            if mn_top1 == lm_pred.label:
                ens_top1 = mn_top1
                ens_conf = (mn_conf + lm_pred.confidence) / 2.0
            elif mn_conf >= 0.85 and lm_pred.confidence < 0.95:
                ens_top1, ens_conf = mn_top1, mn_conf
            else:
                ens_top1, ens_conf = lm_pred.label, lm_pred.confidence

        c = counters[true_lbl]
        c["n"] += 1
        if mn_top1 == true_lbl:
            c["top1_mn"] += 1
        if lm_top1 == true_lbl:
            c["top1_lm"] += 1
        if ens_top1 == true_lbl:
            c["top1_ens"] += 1
        c["true_conf_mn"] += true_conf_mn
        c["true_conf_ens"] += float(ens_conf) if ens_top1 == true_lbl else 0.0
        if out.landmarks is None:
            c["no_hand"] += 1

        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(flat_rows)} processed", flush=True)

    # build sorted report
    report = []
    for lbl, c in counters.items():
        n = c["n"]
        if n == 0:
            continue
        report.append(
            {
                "letter": lbl,
                "n": n,
                "top1_mn": c["top1_mn"] / n,
                "top1_lm": c["top1_lm"] / n,
                "top1_ens": c["top1_ens"] / n,
                "mean_true_conf_mn": c["true_conf_mn"] / n,
                "mean_true_conf_ens_correct": (c["true_conf_ens"] / max(1, c["top1_ens"])),
                "no_hand_rate": c["no_hand"] / n,
            }
        )

    report.sort(key=lambda r: (-r["top1_ens"], -r["mean_true_conf_ens_correct"]))

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2))

    md = ["# Per-class letter accuracy (ASL29 honest test split)\n"]
    md.append(f"Source: `{args.split_csv}` (capped at {args.max_per_class} clips/class)\n")
    md.append("Sorted by ensemble top-1, ties broken by mean true-class confidence on correct predictions.\n")
    md.append("")
    md.append("| Rank | Letter | n | Ens top-1 | MN top-1 | LM top-1 | Mean conf (correct) | No-hand % |")
    md.append("|---|---|---|---|---|---|---|---|")
    for i, r in enumerate(report, 1):
        md.append(
            f"| {i} | **{r['letter']}** | {r['n']} | {r['top1_ens']*100:.1f}% | "
            f"{r['top1_mn']*100:.1f}% | {r['top1_lm']*100:.1f}% | "
            f"{r['mean_true_conf_ens_correct']*100:.1f}% | {r['no_hand_rate']*100:.1f}% |"
        )

    md.append("\n## Top 5 demo letter shortlist (highest ensemble top-1, low no-hand rate)\n")
    shortlist = sorted(
        [r for r in report if r["no_hand_rate"] < 0.3],
        key=lambda r: (-r["top1_ens"], -r["mean_true_conf_ens_correct"]),
    )[:5]
    for i, r in enumerate(shortlist, 1):
        md.append(
            f"{i}. **{r['letter']}** — top-1 {r['top1_ens']*100:.1f}%, "
            f"mean conf {r['mean_true_conf_ens_correct']*100:.1f}%, "
            f"no-hand {r['no_hand_rate']*100:.1f}%"
        )

    args.out_md.write_text("\n".join(md), encoding="utf-8")
    print(f"[letter] wrote {args.out_md}")
    print(f"[letter] wrote {args.out_json}")
    print()
    print("\n".join(md[-15:]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

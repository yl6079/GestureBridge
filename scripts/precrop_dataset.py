"""Pre-crop the ASL29 dataset to hand-only square images.

Reads the contiguous-split CSVs (path, label, class_name), runs MediaPipe
Hands on each image, writes a 224x224 cropped image to a parallel
directory tree, and emits new CSVs with paths pointing to the crops.

Why pre-crop offline instead of cropping at training time:
- Cropping is the dominant per-image cost; doing it once and caching keeps
  GPU training step times sane.
- Forces the train/val/test all to be cropped consistently.

Output layout:
    <output-root>/<class_name>/<original_filename>   (cropped 224x224 PNG)
    <output-splits-dir>/{train,val,test}.csv         (new manifests)
    <output-root>/crop_report.json                   (counts, miss rate)

Images where MediaPipe fails to detect a hand are kept (resized) but
flagged in the report; you can later filter them out with
--drop-undetected if desired.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gesturebridge.pipelines.hand_crop import HandCropper


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits-dir", type=Path, default=Path("data/asl29/splits"))
    parser.add_argument("--output-root", type=Path, default=Path("data/asl29/processed_cropped"))
    parser.add_argument("--output-splits-dir", type=Path, default=Path("data/asl29/splits_cropped"))
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--padding", type=float, default=0.25)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--drop-undetected", action="store_true")
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    args.output_splits_dir.mkdir(parents=True, exist_ok=True)

    cropper = HandCropper(output_size=args.size, padding_ratio=args.padding)

    miss_per_class: Counter[str] = Counter()
    total_per_class: Counter[str] = Counter()

    for split in ("train", "val", "test"):
        df = pd.read_csv(args.splits_dir / f"{split}.csv")
        if args.limit > 0:
            df = df.sample(n=min(args.limit, len(df)), random_state=0).reset_index(drop=True)
        out_rows: list[dict] = []
        print(f"\n[{split}] {len(df)} samples")
        for i, row in enumerate(df.itertuples(index=False), 1):
            src_path = Path(row.path)
            class_name = row.class_name
            label = int(row.label)
            total_per_class[class_name] += 1

            img = cv2.imread(str(src_path))
            if img is None:
                miss_per_class[class_name] += 1
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            res = cropper.crop(img_rgb)
            if not res.found:
                miss_per_class[class_name] += 1
                if args.drop_undetected:
                    continue

            out_dir = args.output_root / class_name
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = (out_dir / src_path.name).with_suffix(".png")
            cv2.imwrite(str(out_path), cv2.cvtColor(res.image, cv2.COLOR_RGB2BGR))
            out_rows.append({
                "path": str(out_path.resolve()),
                "label": label,
                "class_name": class_name,
                "hand_detected": int(res.found),
            })
            if i % 1000 == 0:
                miss_rate = sum(miss_per_class.values()) / sum(total_per_class.values())
                print(f"  {i}/{len(df)}  cumulative miss-rate {miss_rate:.3f}", flush=True)

        out_df = pd.DataFrame(out_rows)
        out_csv = args.output_splits_dir / f"{split}.csv"
        out_df.to_csv(out_csv, index=False)
        print(f"  wrote {out_csv} ({len(out_df)} rows)")

    cropper.close()

    report = {
        "size": args.size,
        "padding": args.padding,
        "drop_undetected": args.drop_undetected,
        "total_per_class": dict(total_per_class),
        "miss_per_class": dict(miss_per_class),
        "overall_miss_rate": sum(miss_per_class.values()) / max(1, sum(total_per_class.values())),
    }
    (args.output_root / "crop_report.json").write_text(json.dumps(report, indent=2))
    print("\n=== Crop report ===")
    print(f"Overall miss-rate: {report['overall_miss_rate']:.4f}")
    print("Per-class miss counts (top 10):")
    for c, n in miss_per_class.most_common(10):
        print(f"  {c}: {n}/{total_per_class[c]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

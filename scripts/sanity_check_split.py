"""Verify that train/val/test split CSVs have no overlap and report stats.

For contiguous splits (the new default), additionally report the frame-index
range each split covers per class — useful for spotting accidental overlap
in the index ranges.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd


_FRAME_INDEX_RE = re.compile(r"(\d+)")


def _frame_index(filename: str) -> int:
    match = _FRAME_INDEX_RE.search(Path(filename).stem)
    return int(match.group(1)) if match else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits-dir", type=Path, default=Path("data/asl29/splits"))
    args = parser.parse_args()

    train = pd.read_csv(args.splits_dir / "train.csv")
    val = pd.read_csv(args.splits_dir / "val.csv")
    test = pd.read_csv(args.splits_dir / "test.csv")

    paths_t = set(train["path"])
    paths_v = set(val["path"])
    paths_te = set(test["path"])

    overlaps = {
        "train ∩ val": paths_t & paths_v,
        "train ∩ test": paths_t & paths_te,
        "val ∩ test": paths_v & paths_te,
    }
    any_overlap = any(s for s in overlaps.values())

    print(f"split sizes: train={len(train)} val={len(val)} test={len(test)}")
    print()
    for name, items in overlaps.items():
        print(f"{name}: {len(items)} overlapping rows")

    if any_overlap:
        print("\nFAIL: splits overlap on file paths.", file=sys.stderr)
        return 1

    print("\nPer-class frame-index ranges:")
    print(f"{'class':<10} {'train':<24} {'val':<22} {'test':<22}")
    print("-" * 80)
    for class_name in sorted(set(train["class_name"])):
        ranges = []
        for split_df in (train, val, test):
            sub = split_df[split_df["class_name"] == class_name]
            if sub.empty:
                ranges.append("—")
                continue
            indices = sub["path"].map(_frame_index).tolist()
            ranges.append(f"{min(indices):>5}..{max(indices):<5} ({len(indices)})")
        print(f"{class_name:<10} {ranges[0]:<24} {ranges[1]:<22} {ranges[2]:<22}")

    print("\nPASS: no path overlap between splits.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

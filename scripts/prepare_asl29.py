from __future__ import annotations

import argparse
import re
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split

from gesturebridge.config import ASL29Config, SystemConfig


_FRAME_INDEX_RE = re.compile(r"(\d+)")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare ASL29 dataset manifests and cleaned images.")
    parser.add_argument("--force", action="store_true", help="Rebuild processed directory from scratch.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help=(
            "Path to local ASL dataset root. If omitted, script auto-detects from "
            "`asl_alphabet`, `data/asl29/raw/asl_alphabet_train`, or `data/asl29/raw/asl_alphabet`."
        ),
    )
    parser.add_argument(
        "--no-copy",
        action="store_true",
        help="Skip copying/re-saving images to the processed dir; write CSVs that "
             "point at raw paths. Faster for quick split-only iteration.",
    )
    parser.add_argument(
        "--split-mode",
        choices=("random", "contiguous"),
        default="contiguous",
        help=(
            "random: stratified random per image (the original, leaks adjacent frames "
            "across splits). contiguous: per class, sort by trailing frame index, take "
            "leading X%% as train, middle as val, trailing as test. Forces temporal "
            "separation within the recording so val/test are not interleaved with train."
        ),
    )
    return parser.parse_args()


def _copy_and_validate(source: Path, target: Path) -> bool:
    try:
        with Image.open(source) as img:
            rgb = img.convert("RGB")
            target.parent.mkdir(parents=True, exist_ok=True)
            rgb.save(target)
    except (UnidentifiedImageError, OSError):
        return False
    return True


def _gather_samples(
    raw_root: Path,
    processed_root: Path,
    class_names: tuple[str, ...],
    copy_files: bool = True,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for label, class_name in enumerate(class_names):
        class_dir = raw_root / class_name
        if not class_dir.exists():
            raise RuntimeError(f"Missing class folder: {class_dir}")
        image_paths = sorted([p for p in class_dir.iterdir() if p.is_file()])
        for image_path in image_paths:
            if copy_files:
                target = processed_root / class_name / image_path.name
                if not _copy_and_validate(image_path, target):
                    continue
                final_path = target.resolve()
            else:
                final_path = image_path.resolve()
            rows.append(
                {
                    "path": str(final_path),
                    "label": label,
                    "class_name": class_name,
                }
            )
    if not rows:
        raise RuntimeError("No valid images found in dataset.")
    return pd.DataFrame(rows)


def _split_dataframe(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("train/val/test ratios must sum to 1.0")

    train_df, temp_df = train_test_split(
        df,
        test_size=(1.0 - train_ratio),
        random_state=seed,
        stratify=df["label"],
    )
    val_relative_ratio = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - val_relative_ratio),
        random_state=seed,
        stratify=temp_df["label"],
    )
    return train_df, val_df, test_df


def _frame_index(filename: str) -> int:
    # Kaggle ASL Alphabet filenames look like "A1.jpg", "A1234.jpg" (one class
    # per folder, integer suffix is the frame ordinal in a single recording).
    match = _FRAME_INDEX_RE.search(Path(filename).stem)
    return int(match.group(1)) if match else 0


def _split_dataframe_contiguous(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Per class: sort by frame index, slice into leading/middle/trailing
    # contiguous blocks. Adjacent frames stay on the same side of the split,
    # so val/test never see frames captured immediately before/after a train
    # frame from the same recording. This is the closest we can get to honest
    # generalization eval on a single-signer single-session dataset like
    # Kaggle ASL Alphabet.
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("train/val/test ratios must sum to 1.0")

    train_parts: list[pd.DataFrame] = []
    val_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []
    for _, group in df.groupby("label", sort=True):
        ordered = group.assign(_frame=group["path"].map(_frame_index)).sort_values("_frame", kind="stable")
        n = len(ordered)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        # n_test absorbs rounding so partition is exact
        train_parts.append(ordered.iloc[:n_train].drop(columns="_frame"))
        val_parts.append(ordered.iloc[n_train:n_train + n_val].drop(columns="_frame"))
        test_parts.append(ordered.iloc[n_train + n_val:].drop(columns="_frame"))
    return (
        pd.concat(train_parts, ignore_index=True),
        pd.concat(val_parts, ignore_index=True),
        pd.concat(test_parts, ignore_index=True),
    )


def _resolve_local_dataset_root(input_dir: Path | None, cfg: ASL29Config) -> Path:
    if input_dir is not None:
        if not input_dir.exists():
            raise RuntimeError(f"Input dataset directory does not exist: {input_dir}")
        return input_dir

    candidates = [
        ROOT / "asl_alphabet",
        cfg.data.raw_dir / "asl_alphabet_train",
        cfg.data.raw_dir / "asl_alphabet",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise RuntimeError(
        "No local ASL dataset directory found. Expected one of: "
        f"{', '.join(str(path) for path in candidates)}"
    )


def main() -> None:
    args = _parse_args()
    cfg = SystemConfig().asl29
    raw_root = _resolve_local_dataset_root(args.input_dir, cfg)

    if args.force and cfg.data.processed_dir.exists():
        shutil.rmtree(cfg.data.processed_dir)
    if not args.no_copy:
        cfg.data.processed_dir.mkdir(parents=True, exist_ok=True)
    cfg.data.splits_dir.mkdir(parents=True, exist_ok=True)

    df = _gather_samples(raw_root, cfg.data.processed_dir, cfg.class_names, copy_files=not args.no_copy)
    if args.split_mode == "contiguous":
        train_df, val_df, test_df = _split_dataframe_contiguous(
            df,
            train_ratio=cfg.data.train_ratio,
            val_ratio=cfg.data.val_ratio,
            test_ratio=cfg.data.test_ratio,
        )
    else:
        train_df, val_df, test_df = _split_dataframe(
            df,
            train_ratio=cfg.data.train_ratio,
            val_ratio=cfg.data.val_ratio,
            test_ratio=cfg.data.test_ratio,
            seed=cfg.data.split_seed,
        )

    train_df.to_csv(cfg.data.train_csv, index=False)
    val_df.to_csv(cfg.data.val_csv, index=False)
    test_df.to_csv(cfg.data.test_csv, index=False)
    cfg.data.labels_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.data.labels_path.write_text("\n".join(cfg.class_names) + "\n", encoding="utf-8")

    summary = pd.DataFrame(
        {
            "split": ["train", "val", "test"],
            "samples": [len(train_df), len(val_df), len(test_df)],
        }
    )
    summary_path = cfg.data.splits_dir / "split_summary.csv"
    summary.to_csv(summary_path, index=False)

    print(f"Source dataset: {raw_root}")
    print(f"Prepared dataset under: {cfg.data.processed_dir}")
    print(f"Train manifest: {cfg.data.train_csv} ({len(train_df)} samples)")
    print(f"Val manifest: {cfg.data.val_csv} ({len(val_df)} samples)")
    print(f"Test manifest: {cfg.data.test_csv} ({len(test_df)} samples)")
    print(f"Split summary: {summary_path}")
    print(f"Labels file: {cfg.data.labels_path}")


if __name__ == "__main__":
    main()


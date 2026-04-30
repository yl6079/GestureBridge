from __future__ import annotations

import argparse
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


def _gather_samples(raw_root: Path, processed_root: Path, class_names: tuple[str, ...]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for label, class_name in enumerate(class_names):
        class_dir = raw_root / class_name
        if not class_dir.exists():
            raise RuntimeError(f"Missing class folder: {class_dir}")
        image_paths = sorted([p for p in class_dir.iterdir() if p.is_file()])
        for image_path in image_paths:
            target = processed_root / class_name / image_path.name
            if _copy_and_validate(image_path, target):
                rows.append(
                    {
                        "path": str(target.resolve()),
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
    cfg.data.processed_dir.mkdir(parents=True, exist_ok=True)
    cfg.data.splits_dir.mkdir(parents=True, exist_ok=True)

    df = _gather_samples(raw_root, cfg.data.processed_dir, cfg.class_names)
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


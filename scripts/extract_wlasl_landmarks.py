"""Extract MediaPipe HandLandmarker sequences from WLASL clips.

Reads `data/wlasl100/manifest.csv` (produced by prepare_wlasl100.py) and
for each successfully-downloaded clip:
  1. Open with cv2; sample N evenly-spaced frames between frame_start and
     frame_end (or full video if frame_end == -1).
  2. Run MediaPipe HandLandmarker (reusing pipelines.hand_crop.HandCropper).
  3. For each frame, normalize the 21-landmark vector (wrist origin,
     max-abs-distance scale) to a 63-d float32.
  4. Stack into a (T, 63) tensor; pad zero / repeat-last for missing frames.

Output NPZ:
  X        : float32 (N_clips, T, 63)
  y        : int32   (N_clips,)              -- gloss index into labels.txt
  detect   : uint8   (N_clips, T)            -- 1 = hand detected, 0 = miss
  split    : uint8   (N_clips,)              -- 0=train, 1=val, 2=test
  paths    : list[str]                        -- relative video paths

Usage:
    python scripts/extract_wlasl_landmarks.py \
        --manifest data/wlasl100/manifest.csv \
        --labels   data/wlasl100/labels.txt \
        --frames   30 \
        --out      data/wlasl100/landmarks.npz
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gesturebridge.pipelines.hand_crop import HandCropper

SPLIT_MAP = {"train": 0, "val": 1, "test": 2}


def _normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """Translate to wrist origin, scale by max-abs distance from wrist.

    Same convention as scripts/extract_landmarks.py for the letter pipeline.
    """
    wrist = landmarks[0:1]
    rel = landmarks - wrist
    scale = float(np.linalg.norm(rel[:, :2], axis=1).max())
    if scale < 1e-6:
        scale = 1.0
    rel = rel / scale
    return rel.astype(np.float32).reshape(-1)


def sample_frame_indices(total: int, n: int, fs: int = 1, fe: int = -1) -> np.ndarray:
    """N evenly-spaced ints in [fs-1, fe-1], inclusive at both ends.

    fs/fe are 1-indexed (WLASL convention); fe == -1 means "use full video".
    """
    start = max(0, fs - 1)
    end = total - 1 if fe == -1 else min(total - 1, fe - 1)
    if end <= start:
        end = total - 1
    if end <= start:
        return np.zeros(n, dtype=np.int64)
    return np.linspace(start, end, n).round().astype(np.int64)


def extract_clip(
    cropper: HandCropper, video_path: Path, n_frames: int, fs: int, fe: int
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (T,63) landmarks tensor + (T,) detection mask."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return np.zeros((n_frames, 63), dtype=np.float32), np.zeros(n_frames, dtype=np.uint8)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total <= 0:
        cap.release()
        return np.zeros((n_frames, 63), dtype=np.float32), np.zeros(n_frames, dtype=np.uint8)

    indices = sample_frame_indices(total, n_frames, fs, fe)
    out = np.zeros((n_frames, 63), dtype=np.float32)
    detect = np.zeros(n_frames, dtype=np.uint8)

    last_seen_landmarks: np.ndarray | None = None
    for j, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, bgr = cap.read()
        if not ok or bgr is None:
            # Use last seen landmarks if any (so the temporal model still has signal).
            if last_seen_landmarks is not None:
                out[j] = last_seen_landmarks
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        try:
            res = cropper.crop(rgb)
        except Exception:
            res = None
        if res is None or not res.found or res.landmarks is None:
            if last_seen_landmarks is not None:
                out[j] = last_seen_landmarks
            continue
        norm = _normalize_landmarks(res.landmarks)
        out[j] = norm
        detect[j] = 1
        last_seen_landmarks = norm
    cap.release()
    return out, detect


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=Path("data/wlasl100/manifest.csv"))
    ap.add_argument("--labels", type=Path, default=Path("data/wlasl100/labels.txt"))
    ap.add_argument("--frames", type=int, default=30, help="frames per clip after sampling")
    ap.add_argument("--out", type=Path, default=Path("data/wlasl100/landmarks.npz"))
    ap.add_argument("--min-detect-rate", type=float, default=0.3,
                    help="drop clips where MediaPipe detected hands in <X fraction of frames")
    ap.add_argument("--min-confidence", type=float, default=0.3,
                    help="MediaPipe min_confidence for detection/presence/tracking")
    ap.add_argument("--limit", type=int, default=0, help="dev cap on number of clips processed")
    args = ap.parse_args()

    labels = [g.strip() for g in args.labels.read_text(encoding="utf-8").splitlines() if g.strip()]
    label_to_idx = {g: i for i, g in enumerate(labels)}
    print(f"[ext] {len(labels)} labels, frames={args.frames}, min_detect={args.min_detect_rate}")

    rows: list[dict] = []
    with args.manifest.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("ok") != "True":
                continue
            if not row.get("path"):
                continue
            rows.append(row)
    print(f"[ext] manifest has {len(rows)} ok rows")
    if args.limit > 0:
        rows = rows[: args.limit]
        print(f"[ext] limiting to {len(rows)}")

    cropper = HandCropper(output_size=224, padding_ratio=0.25, min_confidence=args.min_confidence)

    Xs: list[np.ndarray] = []
    Ds: list[np.ndarray] = []
    ys: list[int] = []
    splits: list[int] = []
    paths: list[str] = []

    drop_low_detect = 0
    drop_open_fail = 0
    t0 = time.monotonic()
    for i, r in enumerate(rows):
        gloss = r["gloss"]
        if gloss not in label_to_idx:
            continue
        path = Path(r["path"])
        if not path.exists():
            drop_open_fail += 1
            continue
        try:
            fs = int(r.get("frame_start", 1) or 1)
            fe = int(r.get("frame_end", -1) or -1)
        except Exception:
            fs, fe = 1, -1
        clip, detect = extract_clip(cropper, path, args.frames, fs, fe)
        if detect.sum() / max(1, args.frames) < args.min_detect_rate:
            drop_low_detect += 1
            continue
        Xs.append(clip)
        Ds.append(detect)
        ys.append(label_to_idx[gloss])
        splits.append(SPLIT_MAP.get(r.get("split", "train"), 0))
        paths.append(r["path"])
        if (i + 1) % 25 == 0 or i == len(rows) - 1:
            elapsed = time.monotonic() - t0
            print(
                f"[ext] {i+1}/{len(rows)} kept={len(Xs)} drop_low_detect={drop_low_detect} "
                f"drop_open_fail={drop_open_fail} elapsed={elapsed:.1f}s",
                flush=True,
            )

    cropper.close()

    if not Xs:
        print("[ext] no clips kept; aborting", file=sys.stderr)
        return 1

    X = np.stack(Xs).astype(np.float32)
    D = np.stack(Ds).astype(np.uint8)
    y = np.asarray(ys, dtype=np.int32)
    sp = np.asarray(splits, dtype=np.uint8)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, X=X, y=y, detect=D, split=sp, paths=np.array(paths))
    # Per-split summary
    print(f"[ext] wrote {args.out}")
    print(f"[ext] X.shape={X.shape} y.shape={y.shape}")
    for name, code in SPLIT_MAP.items():
        n = int((sp == code).sum())
        print(f"[ext] split {name}: {n} clips")
    cls_counts = np.bincount(y, minlength=len(labels))
    print(f"[ext] per-class min/median/max: {cls_counts.min()}/{int(np.median(cls_counts))}/{cls_counts.max()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Convert Kaggle wlasl-300-landmarks (chinhde, MIT) into our training NPZ.

The Kaggle archive contains MediaPipe Holistic-style landmarks for the
official WLASL-100 subset, with 12,730 train + 240 test clips. Format:

    {<class_index_str>: [
        {"keyframes": int,
         "landmarks": {<frame_idx_str>: {
             "pose": [[x,y,z]*15],
             "right": [[x,y,z]*21],
             "left":  [[x,y,z]*21],
         }, ...}},
        ...]}

We keep ONLY the right hand (21×3) to stay format-compatible with our
existing pipeline (`scripts/extract_wlasl_landmarks.py` produces
single-hand 21×3 tensors). If the right hand is all-zero on a frame
but the left hand is non-zero, we mirror the left hand into right-hand
space (flip x). This preserves chirality reasonably for one-hand signs.

We resample each clip to T=30 frames uniformly, normalize per-frame
(wrist origin, max-abs distance scale) — matching our existing code in
`scripts/extract_wlasl_landmarks._normalize_landmarks`.

Output: `data/wlasl100_kaggle/landmarks.npz` with the same schema as
`data/wlasl100/landmarks.npz` (X, y, detect, split, paths).
The label index space is rebuilt from `top_100_classes.txt`. We also
write a fresh `labels.txt` so it's portable.

License: MIT (Kaggle dataset card). Underlying WLASL data is C-UDA;
this aggregated landmark output is a derivative we keep gitignored.

Usage:
    python scripts/convert_kaggle_wlasl100_landmarks.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray | None:
    """Translate to wrist origin, scale by max-abs distance from wrist.

    Returns 63-d float32, or None if input has NaN/Inf so the caller can
    skip rather than poisoning the buffer through the last-seen carry.
    """
    if not np.isfinite(landmarks).all():
        return None
    wrist = landmarks[0:1]
    rel = landmarks - wrist
    scale = float(np.linalg.norm(rel[:, :2], axis=1).max())
    if scale < 1e-6:
        scale = 1.0
    out = (rel / scale).astype(np.float32).reshape(-1)
    if not np.isfinite(out).all():
        return None
    return out


def pick_hand_array(frame_dict: dict) -> tuple[np.ndarray | None, str]:
    """Return one (21,3) array prefering right hand, with chirality flip
    if only left is present. Returns (None, "miss") if neither is detected.
    """
    right = np.asarray(frame_dict.get("right", []), dtype=np.float32)
    left = np.asarray(frame_dict.get("left", []), dtype=np.float32)
    right_ok = right.shape == (21, 3) and np.abs(right).max() > 1e-6
    left_ok = left.shape == (21, 3) and np.abs(left).max() > 1e-6
    if right_ok:
        return right, "right"
    if left_ok:
        # Mirror x to fake a right hand.
        flipped = left.copy()
        flipped[:, 0] = -flipped[:, 0]
        return flipped, "left_mirrored"
    return None, "miss"


def sample_uniform(t_in: int, t_out: int) -> np.ndarray:
    if t_in <= 0:
        return np.zeros(t_out, dtype=np.int64)
    if t_in == 1:
        return np.zeros(t_out, dtype=np.int64)
    return np.linspace(0, t_in - 1, t_out).round().astype(np.int64)


def convert_clip(clip: dict, t_out: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """Convert one clip to (T, 63) + (T,) detection mask."""
    landmarks = clip.get("landmarks", {})
    frame_keys = sorted(landmarks.keys(), key=lambda k: int(k))
    if not frame_keys:
        return np.zeros((t_out, 63), dtype=np.float32), np.zeros(t_out, dtype=np.uint8)

    indices = sample_uniform(len(frame_keys), t_out)
    out = np.zeros((t_out, 63), dtype=np.float32)
    detect = np.zeros(t_out, dtype=np.uint8)
    last_seen: np.ndarray | None = None

    for j, idx in enumerate(indices):
        frame = landmarks[frame_keys[int(idx)]]
        hand, src = pick_hand_array(frame)
        if hand is None:
            if last_seen is not None:
                out[j] = last_seen
            continue
        norm = normalize_landmarks(hand)
        if norm is None:
            if last_seen is not None:
                out[j] = last_seen
            continue
        out[j] = norm
        detect[j] = 1
        last_seen = norm
    return out, detect


def load_class_map(path: Path) -> dict[str, str]:
    """Read top_100_classes.txt → {class_index_str: gloss}. The file is
    space- or tab-separated with class index then gloss.
    """
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) >= 2:
            out[parts[0]] = parts[1]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=Path("data/wlasl_external"),
                    help="dir containing wasl100_landmarks_*.json + top_100_classes.txt")
    ap.add_argument("--out-dir", type=Path, default=Path("data/wlasl100_kaggle"))
    ap.add_argument("--t-out", type=int, default=30)
    ap.add_argument("--min-detect-rate", type=float, default=0.3)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    class_map = load_class_map(args.src / "top_100_classes.txt")
    glosses = [class_map[k] for k in sorted(class_map.keys(), key=lambda x: int(x))]
    label_to_idx = {g: i for i, g in enumerate(glosses)}
    print(f"[conv] {len(glosses)} classes; first 5: {glosses[:5]}")

    splits = [
        ("train", "wasl100_landmarks_train.json", 0),
        ("test", "wasl100_landmarks_test.json", 2),
    ]

    Xs: list[np.ndarray] = []
    Ds: list[np.ndarray] = []
    ys: list[int] = []
    sps: list[int] = []
    paths: list[str] = []

    for split_name, fname, split_code in splits:
        fp = args.src / fname
        if not fp.exists():
            print(f"[conv] missing {fp}, skipping {split_name}")
            continue
        print(f"[conv] loading {fp.name} ...", flush=True)
        d = json.load(fp.open())
        kept = drop_low = drop_empty = 0
        for cls_str, clips in d.items():
            if cls_str not in class_map:
                continue
            gloss = class_map[cls_str]
            cls_idx = label_to_idx[gloss]
            for ci, clip in enumerate(clips):
                seq, detect = convert_clip(clip, args.t_out)
                if int(detect.sum()) == 0:
                    drop_empty += 1
                    continue
                if detect.sum() / max(1, args.t_out) < args.min_detect_rate:
                    drop_low += 1
                    continue
                Xs.append(seq)
                Ds.append(detect)
                ys.append(cls_idx)
                sps.append(split_code)
                paths.append(f"kaggle:{split_name}/{cls_str}/clip_{ci}")
                kept += 1
        print(f"[conv] {split_name}: kept={kept} drop_low_detect={drop_low} drop_empty={drop_empty}")

    if not Xs:
        print("[conv] nothing kept; aborting", file=sys.stderr)
        return 1

    X = np.stack(Xs).astype(np.float32)
    D = np.stack(Ds).astype(np.uint8)
    y = np.asarray(ys, dtype=np.int32)
    sp = np.asarray(sps, dtype=np.uint8)

    # The archive omits a separate val landmark JSON, so peel ~10% off the
    # train clips into val (deterministic by hash so it's reproducible).
    rng = np.random.default_rng(0)
    train_mask = sp == 0
    n_train = int(train_mask.sum())
    val_count = max(1, n_train // 10)
    train_indices = np.where(train_mask)[0]
    val_pick = rng.choice(train_indices, size=val_count, replace=False)
    sp[val_pick] = 1

    out_npz = args.out_dir / "landmarks.npz"
    np.savez_compressed(out_npz, X=X, y=y, detect=D, split=sp, paths=np.array(paths))
    out_labels = args.out_dir / "labels.txt"
    out_labels.write_text("\n".join(glosses) + "\n", encoding="utf-8")

    print(f"[conv] wrote {out_npz} X={X.shape} y={y.shape}")
    print(f"[conv] wrote {out_labels}")
    for name, code in [("train", 0), ("val", 1), ("test", 2)]:
        n = int((sp == code).sum())
        print(f"[conv] split {name}: {n} clips")
    cls_counts = np.bincount(y, minlength=len(glosses))
    print(f"[conv] per-class min/median/max: {cls_counts.min()}/{int(np.median(cls_counts))}/{cls_counts.max()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

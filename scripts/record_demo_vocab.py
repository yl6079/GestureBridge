"""On-Pi recorder for the few-shot signer-conditioned vocabulary (T1-B).

Wire-up:
    1. Sit in front of the Pi's C270.
    2. Run this script over SSH (or directly on the Pi terminal).
    3. For each (word, take), the script counts down 1.5 s and records
       a clip. SPACE between takes; ESC cancels.
    4. After all takes, it extracts MediaPipe Hands landmarks per frame,
       resamples to 30 frames, normalizes (wrist origin + max-abs scale),
       and writes a single npz at `--out`.

Output schema (drop-in compatible with `data/wlasl100_kaggle/landmarks.npz`):
    X:        (N_words * N_takes, 30, 63) float32
    y:        (N_words * N_takes,)      int32
    take_ids: (N_words * N_takes,)      int32   # 0..N_takes-1, lets us split per-take
    class_names: (N_words,)             unicode

Usage:
    python scripts/record_demo_vocab.py \
        --words hello help yes no water \
        --takes 5 \
        --out data/fewshot/landmarks.npz
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gesturebridge.pipelines.hand_crop import HandCropper


def _normalize_landmarks(landmarks_21x3: np.ndarray) -> np.ndarray:
    """Wrist-origin + max-abs scale (matches scripts/extract_wlasl_landmarks)."""
    wrist = landmarks_21x3[0:1]
    rel = landmarks_21x3 - wrist
    scale = float(np.linalg.norm(rel[:, :2], axis=1).max())
    if not np.isfinite(scale) or scale < 1e-6:
        scale = 1.0
    out = (rel / scale).astype(np.float32).reshape(-1)
    if not np.isfinite(out).all():
        return np.zeros(63, dtype=np.float32)
    return out


def _resample_to_n(seq: np.ndarray, n_out: int) -> np.ndarray:
    """Linear temporal resample (T, D) -> (n_out, D)."""
    T, D = seq.shape
    if T == n_out:
        return seq.astype(np.float32)
    if T == 1:
        return np.repeat(seq, n_out, axis=0).astype(np.float32)
    old_t = np.linspace(0.0, 1.0, T, dtype=np.float32)
    new_t = np.linspace(0.0, 1.0, n_out, dtype=np.float32)
    out = np.empty((n_out, D), dtype=np.float32)
    for d in range(D):
        out[:, d] = np.interp(new_t, old_t, seq[:, d]).astype(np.float32)
    return out


def _interp_nans(seq: np.ndarray) -> np.ndarray:
    seq = seq.copy().astype(np.float32)
    T, D = seq.shape
    x = np.arange(T)
    for d in range(D):
        y = seq[:, d]
        mask = np.isfinite(y) & (np.abs(y) > 1e-9)
        if mask.sum() == 0:
            continue
        if mask.sum() == 1:
            seq[:, d] = y[mask][0]
        else:
            seq[:, d] = np.interp(x, x[mask], y[mask]).astype(np.float32)
    return seq


def record_one_clip(
    cap: cv2.VideoCapture,
    cropper: HandCropper,
    word: str,
    take_idx: int,
    record_secs: float,
    target_len: int,
) -> np.ndarray:
    """Record a single clip and return a (target_len, 63) normalized tensor."""
    print(f"\nReady for '{word}' take {take_idx + 1}. Press SPACE to start, ESC to abort.")
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        disp = frame.copy()
        cv2.putText(
            disp,
            f"{word} take {take_idx + 1}: SPACE=start, ESC=quit",
            (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2,
        )
        cv2.imshow("recorder", disp)
        key = cv2.waitKey(1) & 0xFF
        if key == 32:  # space
            break
        if key == 27:  # esc
            raise KeyboardInterrupt("user aborted")

    seq = []
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < record_secs:
        ok, frame = cap.read()
        if not ok:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = cropper.crop(rgb)
        if result.found and result.landmarks is not None:
            seq.append(_normalize_landmarks(result.landmarks))
        else:
            seq.append(np.full(63, np.nan, dtype=np.float32))
        # countdown overlay
        remaining = max(0.0, record_secs - (time.perf_counter() - t0))
        disp = frame.copy()
        cv2.putText(
            disp,
            f"REC {word} take {take_idx + 1} ({remaining:.1f}s)  hand={'Y' if result.found else 'N'}",
            (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2,
        )
        cv2.imshow("recorder", disp)
        cv2.waitKey(1)

    if not seq:
        return np.zeros((target_len, 63), dtype=np.float32)
    arr = np.stack(seq).astype(np.float32)
    arr = _interp_nans(arr)
    arr = _resample_to_n(arr, target_len)
    return arr


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--words", nargs="+",
                    default=["hello", "help", "yes", "no", "water"],
                    help="vocabulary to record (single-handed words recommended)")
    ap.add_argument("--takes", type=int, default=5)
    ap.add_argument("--record-secs", type=float, default=1.4)
    ap.add_argument("--target-len", type=int, default=30)
    ap.add_argument("--camera-index", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("data/fewshot/landmarks.npz"))
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    print(f"[record] words={args.words}  takes={args.takes}  out={args.out}")

    cropper = HandCropper(output_size=224, padding_ratio=0.25, min_confidence=0.3)
    cap = cv2.VideoCapture(args.camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        print(f"[record] could not open camera index {args.camera_index}", file=sys.stderr)
        return 1

    Xs, ys, take_ids = [], [], []
    try:
        for class_id, word in enumerate(args.words):
            for take_idx in range(args.takes):
                seq = record_one_clip(cap, cropper, word, take_idx, args.record_secs, args.target_len)
                Xs.append(seq)
                ys.append(class_id)
                take_ids.append(take_idx)
                print(f"[record] captured {word} take {take_idx + 1}: shape={seq.shape}")
    except KeyboardInterrupt:
        print("[record] aborted; saving partial")
    finally:
        cap.release()
        cropper.close()
        cv2.destroyAllWindows()

    if not Xs:
        print("[record] nothing recorded; exiting", file=sys.stderr)
        return 1

    X = np.stack(Xs).astype(np.float32)
    y = np.asarray(ys, dtype=np.int32)
    take_ids_a = np.asarray(take_ids, dtype=np.int32)
    np.savez_compressed(
        args.out,
        X=X, y=y, take_ids=take_ids_a, class_names=np.array(args.words),
    )
    print(f"[record] wrote {args.out}: X={X.shape} y={y.shape}")
    print(f"[record] class breakdown: " + ", ".join(
        f"{w}:{int((y == i).sum())}" for i, w in enumerate(args.words)
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

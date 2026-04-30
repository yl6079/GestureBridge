from __future__ import annotations

import argparse
from pathlib import Path
from time import sleep

import cv2

from gesturebridge.config import SystemConfig
from gesturebridge.pipelines.asl29_tflite import ASL29TFLiteRuntime


def _parse_args() -> argparse.Namespace:
    cfg = SystemConfig().asl29
    parser = argparse.ArgumentParser(description="Run realtime ASL29 inference in headless mode (no GUI).")
    parser.add_argument("--model", type=Path, default=cfg.export.fp32_tflite_path, help="Path to TFLite model file.")
    parser.add_argument("--labels", type=Path, default=cfg.data.labels_path, help="Path to labels txt file.")
    parser.add_argument("--camera-index", type=int, default=cfg.runtime.camera_index, help="OpenCV camera index.")
    parser.add_argument("--threads", type=int, default=cfg.runtime.tflite_threads, help="TFLite threads.")
    parser.add_argument("--top-k", type=int, default=cfg.runtime.preview_top_k, help="Top-k predictions to print.")
    parser.add_argument("--interval-ms", type=int, default=250, help="Print interval in milliseconds.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    runtime = ASL29TFLiteRuntime(
        model_path=args.model,
        labels_path=args.labels,
        threads=args.threads,
        top_k=args.top_k,
    )

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open camera index {args.camera_index}")

    print(f"[headless] camera={args.camera_index} model={args.model}")
    print("[headless] press Ctrl+C to stop")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            result = runtime.predict(frame)
            topk = " | ".join([f"{label}:{score:.3f}" for label, score in result.top_k])
            print(
                f"pred={result.label:<8} conf={result.confidence:.3f} "
                f"latency={result.latency_ms:6.1f}ms topk=[{topk}]"
            )
            sleep(max(args.interval_ms, 1) / 1000.0)
    except KeyboardInterrupt:
        print("\n[headless] stopped by user")
    finally:
        cap.release()


if __name__ == "__main__":
    main()


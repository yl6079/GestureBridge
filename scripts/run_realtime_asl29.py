from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import cv2
import numpy as np
import tensorflow as tf

from gesturebridge.config import SystemConfig
from gesturebridge.ml.data_pipeline import load_class_names, preprocess_for_mobilenet


def _parse_args() -> argparse.Namespace:
    cfg = SystemConfig().asl29
    parser = argparse.ArgumentParser(description="Run realtime ASL29 inference with a TFLite model.")
    parser.add_argument(
        "--model",
        type=Path,
        default=cfg.export.fp32_tflite_path,
        help="Path to TFLite model file (default: fp32 export path from config).",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=cfg.data.labels_path,
        help="Path to labels txt file.",
    )
    parser.add_argument("--camera-index", type=int, default=cfg.runtime.camera_index, help="OpenCV camera index.")
    parser.add_argument("--image-size", type=int, default=cfg.data.image_size, help="Square input size.")
    parser.add_argument("--threads", type=int, default=cfg.runtime.tflite_threads, help="TFLite threads.")
    parser.add_argument(
        "--top-k",
        type=int,
        default=cfg.runtime.preview_top_k,
        help="Number of top predictions to overlay.",
    )
    parser.add_argument(
        "--window-title",
        default="ASL29 Realtime",
        help="OpenCV preview window title.",
    )
    return parser.parse_args()


def _load_interpreter(model_path: Path, threads: int):
    def _allocate_with_fallback(interpreter):
        try:
            interpreter.allocate_tensors()
            return interpreter
        except RuntimeError as exc:
            if "XNNPACK" not in str(exc):
                raise
            os.environ["TF_LITE_DISABLE_XNNPACK"] = "1"
            try:
                fallback = tf.lite.Interpreter(
                    model_path=str(model_path),
                    experimental_delegates=[],
                    num_threads=1,
                    experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF,
                )
            except TypeError:
                fallback = tf.lite.Interpreter(
                    model_path=str(model_path),
                    experimental_delegates=[],
                    num_threads=1,
                )
            fallback.allocate_tensors()
            return fallback

    try:
        from tflite_runtime.interpreter import Interpreter

        interpreter = Interpreter(model_path=str(model_path), num_threads=threads)
    except ImportError:
        interpreter = tf.lite.Interpreter(model_path=str(model_path), num_threads=threads)
    return _allocate_with_fallback(interpreter)


def _prepare_input(frame: np.ndarray, image_size: int) -> np.ndarray:
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(frame_rgb, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    image = tf.convert_to_tensor(resized, dtype=tf.float32)
    image = preprocess_for_mobilenet(image)
    image = tf.expand_dims(image, axis=0)
    return image.numpy()


def _invoke(interpreter, input_tensor: np.ndarray) -> np.ndarray:
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    input_scale, input_zero_point = input_details["quantization"]
    output_scale, output_zero_point = output_details["quantization"]

    if input_details["dtype"] == np.int8:
        input_tensor = np.clip(np.round(input_tensor / input_scale + input_zero_point), -128, 127).astype(np.int8)
    elif input_details["dtype"] == np.uint8:
        input_tensor = np.clip(np.round(input_tensor / input_scale + input_zero_point), 0, 255).astype(np.uint8)
    else:
        input_tensor = input_tensor.astype(np.float32)

    interpreter.set_tensor(input_details["index"], input_tensor)
    interpreter.invoke()
    logits = interpreter.get_tensor(output_details["index"])
    if output_details["dtype"] in (np.int8, np.uint8) and output_scale > 0:
        logits = (logits.astype(np.float32) - output_zero_point) * output_scale
    return logits[0]


def _format_topk_text(logits: np.ndarray, labels: list[str], top_k: int, elapsed_ms: float) -> list[str]:
    probs = tf.nn.softmax(tf.convert_to_tensor(logits, dtype=tf.float32)).numpy()
    top_k = max(1, min(top_k, len(probs)))
    top_indices = np.argsort(probs)[::-1][:top_k]
    lines = [f"infer={elapsed_ms:.1f}ms"]
    for rank, idx in enumerate(top_indices, start=1):
        label = labels[idx] if idx < len(labels) else f"class_{idx}"
        lines.append(f"{rank}. {label}: {probs[idx]:.2f}")
    return lines


def _draw_overlay(frame: np.ndarray, lines: list[str]) -> None:
    y = 28
    for line in lines:
        cv2.putText(
            frame,
            line,
            (16, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        y += 26


def main() -> None:
    args = _parse_args()
    if not args.model.exists():
        raise RuntimeError(f"Model file not found: {args.model}")
    if not args.labels.exists():
        raise RuntimeError(f"Labels file not found: {args.labels}")

    labels = load_class_names(args.labels)
    interpreter = _load_interpreter(args.model, args.threads)

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open camera index {args.camera_index}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            t0 = perf_counter()
            input_tensor = _prepare_input(frame, args.image_size)
            logits = _invoke(interpreter, input_tensor)
            elapsed_ms = (perf_counter() - t0) * 1000.0

            overlay_lines = _format_topk_text(logits, labels, top_k=args.top_k, elapsed_ms=elapsed_ms)
            _draw_overlay(frame, overlay_lines)
            cv2.imshow(args.window_title, frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


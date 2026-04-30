from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import tensorflow as tf

from gesturebridge.config import SystemConfig
from gesturebridge.ml.data_pipeline import load_manifest, preprocess_for_mobilenet


def _make_interpreter(model_path: Path, threads: int):
    try:
        from tflite_runtime.interpreter import Interpreter

        return Interpreter(model_path=str(model_path), num_threads=threads)
    except ImportError:
        return tf.lite.Interpreter(model_path=str(model_path), num_threads=threads)


def _load_image(path: str, image_size: int) -> np.ndarray:
    image_bytes = tf.io.read_file(path)
    image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    image = tf.image.resize(image, [image_size, image_size], method=tf.image.ResizeMethod.BILINEAR)
    image = tf.cast(image, tf.float32)
    image = preprocess_for_mobilenet(image)
    return image.numpy()


def _run_benchmark(model_path: Path, manifest_path: Path, image_size: int, threads: int) -> dict[str, float]:
    manifest = load_manifest(manifest_path)
    interpreter = _make_interpreter(model_path, threads=threads)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    input_scale, input_zero_point = input_details["quantization"]
    output_scale, output_zero_point = output_details["quantization"]

    y_true = manifest["label"].to_numpy(dtype=np.int64)
    y_pred = np.zeros_like(y_true)
    timings_ms: list[float] = []

    for idx, row in manifest.iterrows():
        image = _load_image(str(row["path"]), image_size=image_size)
        image = np.expand_dims(image, axis=0)
        if input_details["dtype"] == np.int8:
            image = np.clip(np.round(image / input_scale + input_zero_point), -128, 127).astype(np.int8)
        elif input_details["dtype"] == np.uint8:
            image = np.clip(np.round(image / input_scale + input_zero_point), 0, 255).astype(np.uint8)
        else:
            image = image.astype(np.float32)

        t0 = perf_counter()
        interpreter.set_tensor(input_details["index"], image)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details["index"])
        timings_ms.append((perf_counter() - t0) * 1000.0)

        if output_details["dtype"] in (np.int8, np.uint8) and output_scale > 0:
            output = (output.astype(np.float32) - output_zero_point) * output_scale
        y_pred[idx] = int(np.argmax(output[0]))

    p50 = float(np.percentile(timings_ms, 50))
    p95 = float(np.percentile(timings_ms, 95))
    mean = float(np.mean(timings_ms))
    fps = float(1000.0 / mean) if mean > 0 else 0.0
    accuracy = float((y_pred == y_true).mean())
    return {
        "model_path": str(model_path),
        "samples": int(len(y_true)),
        "accuracy": accuracy,
        "latency_ms_mean": mean,
        "latency_ms_p50": p50,
        "latency_ms_p95": p95,
        "fps": fps,
    }


def main() -> None:
    cfg = SystemConfig().asl29
    fp32 = _run_benchmark(
        model_path=cfg.export.fp32_tflite_path,
        manifest_path=cfg.data.test_csv,
        image_size=cfg.data.image_size,
        threads=cfg.runtime.tflite_threads,
    )
    int8 = _run_benchmark(
        model_path=cfg.export.int8_tflite_path,
        manifest_path=cfg.data.test_csv,
        image_size=cfg.data.image_size,
        threads=cfg.runtime.tflite_threads,
    )
    output = {
        "threads": cfg.runtime.tflite_threads,
        "fp32": fp32,
        "int8": int8,
    }
    cfg.runtime.benchmark_report_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.runtime.benchmark_report_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))
    print(f"Saved benchmark report to {cfg.runtime.benchmark_report_path}")


if __name__ == "__main__":
    main()


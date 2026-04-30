from __future__ import annotations

import json
import inspect
import argparse
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
from gesturebridge.ml.data_pipeline import (
    load_manifest,
    preprocess_for_mobilenet,
    representative_dataset_from_csv,
)


def _patch_flatbuffers_end_vector() -> None:
    """Patch flatbuffers API mismatch seen with some TensorFlow builds."""
    try:
        import flatbuffers
    except ImportError:
        return

    end_vector = getattr(flatbuffers.Builder, "EndVector", None)
    if end_vector is None:
        return

    params = list(inspect.signature(end_vector).parameters)
    # Expected signatures:
    # - old: EndVector(self)
    # - new: EndVector(self, vectorNumElems)
    if len(params) < 2:
        return

    original_end_vector = flatbuffers.Builder.EndVector

    def _compat_end_vector(self, vector_num_elems=None):  # type: ignore[no-untyped-def]
        if vector_num_elems is None:
            vector_num_elems = getattr(self, "vectorNumElems", 0)
        return original_end_vector(self, vector_num_elems)

    flatbuffers.Builder.EndVector = _compat_end_vector  # type: ignore[assignment]


def _patch_flatbuffers_create_numpy_vector() -> None:
    """Provide CreateNumpyVector for older flatbuffers versions."""
    try:
        import flatbuffers
    except ImportError:
        return

    if hasattr(flatbuffers.Builder, "CreateNumpyVector"):
        return

    def _compat_create_numpy_vector(self, ndarray):  # type: ignore[no-untyped-def]
        arr = np.asarray(ndarray)
        if arr.ndim != 1:
            arr = arr.reshape(-1)

        dtype = arr.dtype
        if dtype == np.dtype(np.int32):
            self.StartVector(4, arr.size, 4)
            for value in arr[::-1]:
                self.PrependInt32(int(value))
            return self.EndVector(arr.size)
        if dtype == np.dtype(np.uint32):
            self.StartVector(4, arr.size, 4)
            for value in arr[::-1]:
                self.PrependUint32(int(value))
            return self.EndVector(arr.size)
        if dtype == np.dtype(np.int64):
            self.StartVector(8, arr.size, 8)
            for value in arr[::-1]:
                self.PrependInt64(int(value))
            return self.EndVector(arr.size)
        if dtype == np.dtype(np.uint64):
            self.StartVector(8, arr.size, 8)
            for value in arr[::-1]:
                self.PrependUint64(int(value))
            return self.EndVector(arr.size)
        if dtype == np.dtype(np.float32):
            self.StartVector(4, arr.size, 4)
            for value in arr[::-1]:
                self.PrependFloat32(float(value))
            return self.EndVector(arr.size)
        if dtype == np.dtype(np.float64):
            self.StartVector(8, arr.size, 8)
            for value in arr[::-1]:
                self.PrependFloat64(float(value))
            return self.EndVector(arr.size)

        # Fallback for uncommon dtypes used in schema vectors.
        casted = arr.astype(np.int32, copy=False)
        self.StartVector(4, casted.size, 4)
        for value in casted[::-1]:
            self.PrependInt32(int(value))
        return self.EndVector(casted.size)

    flatbuffers.Builder.CreateNumpyVector = _compat_create_numpy_vector  # type: ignore[attr-defined]


def _patch_flatbuffers_finish_signature() -> None:
    """Accept TensorFlow's file_identifier kwarg on older flatbuffers."""
    try:
        import flatbuffers
    except ImportError:
        return

    finish = getattr(flatbuffers.Builder, "Finish", None)
    if finish is None:
        return

    params = list(inspect.signature(finish).parameters)
    if "file_identifier" in params:
        return

    original_finish = flatbuffers.Builder.Finish
    positional_count = len(params)

    def _compat_finish(self, root_table, file_identifier=None):  # type: ignore[no-untyped-def]
        if file_identifier is None or positional_count <= 2:
            return original_finish(self, root_table)
        try:
            return original_finish(self, root_table, file_identifier)
        except TypeError:
            return original_finish(self, root_table)

    flatbuffers.Builder.Finish = _compat_finish  # type: ignore[assignment]


def _load_image(path: str, image_size: int) -> np.ndarray:
    image_bytes = tf.io.read_file(path)
    image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    image = tf.image.resize(image, [image_size, image_size], method=tf.image.ResizeMethod.BILINEAR)
    image = tf.cast(image, tf.float32)
    image = preprocess_for_mobilenet(image)
    return image.numpy()


def _evaluate_tflite(tflite_path: Path, csv_path: Path, image_size: int) -> dict[str, float]:
    manifest = load_manifest(csv_path)
    interpreter = None
    _fallback_tried = False
    try:
        _interp = tf.lite.Interpreter(model_path=str(tflite_path))
        _interp.allocate_tensors()
        interpreter = _interp
    except RuntimeError as exc:
        if "XNNPACK" not in str(exc) and "failed to prepare" not in str(exc):
            raise
        print(f"[eval] XNNPACK delegate failed; retrying without default delegates.")
        _fallback_tried = True
    if interpreter is None and _fallback_tried:
        try:
            from tensorflow.lite.python.interpreter import OpResolverType as _ORT
            _interp = tf.lite.Interpreter(
                model_path=str(tflite_path),
                experimental_op_resolver_type=_ORT.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
            )
            _interp.allocate_tensors()
            interpreter = _interp
        except Exception as exc2:
            print(f"[eval] Cannot evaluate {tflite_path.name}: {exc2}; skipping.")
            return {"warning": "xnnpack_unavailable", "tflite_path": str(tflite_path)}
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    y_true = manifest["label"].to_numpy(dtype=np.int64)
    y_pred = np.zeros_like(y_true)

    input_scale, input_zero_point = input_details["quantization"]
    output_scale, output_zero_point = output_details["quantization"]

    start = perf_counter()
    for idx, row in manifest.iterrows():
        image = _load_image(str(row["path"]), image_size=image_size)
        image = np.expand_dims(image, axis=0)
        if input_details["dtype"] == np.int8:
            image = np.clip(np.round(image / input_scale + input_zero_point), -128, 127).astype(np.int8)
        elif input_details["dtype"] == np.uint8:
            image = np.clip(np.round(image / input_scale + input_zero_point), 0, 255).astype(np.uint8)
        else:
            image = image.astype(np.float32)

        interpreter.set_tensor(input_details["index"], image)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details["index"])

        if output_details["dtype"] in (np.int8, np.uint8) and output_scale > 0:
            output = (output.astype(np.float32) - output_zero_point) * output_scale

        y_pred[idx] = int(np.argmax(output[0]))

    elapsed = perf_counter() - start
    accuracy = float((y_pred == y_true).mean())
    latency_ms = elapsed / max(len(y_true), 1) * 1000.0
    return {
        "accuracy": accuracy,
        "avg_inference_ms_per_sample": float(latency_ms),
        "num_samples": int(len(y_true)),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export ASL29 TFLite models (FP32 and INT8).")
    parser.add_argument(
        "--only-fp32",
        action="store_true",
        help="Export only FP32 model and skip INT8 quantization/evaluation.",
    )
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=None,
        help=(
            "Directory of representative images organized as <dir>/<class>/<file>.png. "
            "If provided, INT8 calibration uses these instead of train CSV. Pair with "
            "scripts/capture_calibration_set.py to capture from the deployed C270 — "
            "matching the deployment distribution is critical for usable INT8."
        ),
    )
    parser.add_argument(
        "--int8-float-output",
        action="store_true",
        help="Keep softmax output in float32 (uint8 input, float32 output). "
             "Reduces accuracy loss from quantizing the head; recommended.",
    )
    return parser.parse_args()


def _representative_dataset_from_dir(calibration_dir: Path, image_size: int):
    """RepresentativeDataset built from <dir>/<class>/<file> images."""
    paths: list[Path] = []
    for class_dir in sorted(p for p in calibration_dir.iterdir() if p.is_dir()):
        paths.extend(sorted(class_dir.glob("*.png")) + sorted(class_dir.glob("*.jpg")))
    if not paths:
        raise RuntimeError(f"No images under {calibration_dir}")

    def _generator():
        for path in paths:
            image_bytes = tf.io.read_file(str(path))
            image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
            image = tf.image.resize(image, [image_size, image_size], method=tf.image.ResizeMethod.BILINEAR)
            image = tf.cast(image, tf.float32)
            image = preprocess_for_mobilenet(image)
            image = tf.expand_dims(image, axis=0)
            yield [tf.cast(image, tf.float32)]

    print(f"[int8] using {len(paths)} calibration images from {calibration_dir}")
    return _generator


def _convert_tflite(converter: tf.lite.TFLiteConverter, enable_compat: bool) -> bytes:
    if enable_compat:
        _patch_flatbuffers_end_vector()
        _patch_flatbuffers_create_numpy_vector()
        _patch_flatbuffers_finish_signature()
    return converter.convert()


def _validate_tflite_bytes(model_bytes: bytes, label: str) -> None:
    if len(model_bytes) < 1024:
        raise RuntimeError(f"{label} export produced suspiciously small file: {len(model_bytes)} bytes")
    if b"TFL3" not in model_bytes[:64]:
        raise RuntimeError(f"{label} export missing TFL3 header; generated artifact appears invalid.")


def main() -> None:
    args = _parse_args()
    cfg = SystemConfig().asl29
    # If a sweep ran with --tag, the best model lives at a tagged path
    # recorded in best_pointer.json rather than the canonical best.keras.
    _pointer = Path("artifacts/asl29/best_pointer.json")
    if _pointer.exists():
        import json as _json
        _ptr = _json.loads(_pointer.read_text())
        _mp = Path(_ptr["best_model_path"])
        if _mp.exists():
            cfg.training.model_path = _mp
            print(f"[export] Using sweep winner: {_mp}")
    model = tf.keras.models.load_model(cfg.training.model_path)

    cfg.export.fp32_tflite_path.parent.mkdir(parents=True, exist_ok=True)

    fp32_converter = tf.lite.TFLiteConverter.from_keras_model(model)
    try:
        fp32_tflite = _convert_tflite(fp32_converter, enable_compat=False)
    except Exception:
        fp32_tflite = _convert_tflite(fp32_converter, enable_compat=True)
    _validate_tflite_bytes(fp32_tflite, "FP32")
    cfg.export.fp32_tflite_path.write_bytes(fp32_tflite)

    fp32_eval = (
        _evaluate_tflite(cfg.export.fp32_tflite_path, cfg.data.test_csv, cfg.data.image_size)
        if cfg.data.test_csv.exists()
        else {"warning": f"missing test csv: {cfg.data.test_csv}"}
    )

    if args.only_fp32:
        report = {
            "fp32_tflite_path": str(cfg.export.fp32_tflite_path),
            "fp32_size_bytes": cfg.export.fp32_tflite_path.stat().st_size,
            "fp32_eval": fp32_eval,
            "int8_skipped": True,
        }
        cfg.export.quant_report_path.parent.mkdir(parents=True, exist_ok=True)
        cfg.export.quant_report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        print(f"Saved export report to {cfg.export.quant_report_path}")
        return

    int8_converter = tf.lite.TFLiteConverter.from_keras_model(model)
    int8_converter.optimizations = [tf.lite.Optimize.DEFAULT]
    if args.calibration_dir is not None and args.calibration_dir.exists():
        int8_converter.representative_dataset = _representative_dataset_from_dir(
            args.calibration_dir,
            image_size=cfg.data.image_size,
        )
    else:
        int8_converter.representative_dataset = representative_dataset_from_csv(
            csv_path=cfg.data.train_csv,
            image_size=cfg.data.image_size,
            sample_count=cfg.export.representative_samples,
        )
    int8_converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    if args.int8_float_output:
        int8_converter.inference_input_type = tf.uint8
        int8_converter.inference_output_type = tf.float32
    else:
        int8_converter.inference_input_type = tf.int8
        int8_converter.inference_output_type = tf.int8
    try:
        int8_tflite = _convert_tflite(int8_converter, enable_compat=False)
    except Exception:
        int8_tflite = _convert_tflite(int8_converter, enable_compat=True)
    _validate_tflite_bytes(int8_tflite, "INT8")
    cfg.export.int8_tflite_path.write_bytes(int8_tflite)

    int8_eval = (
        _evaluate_tflite(cfg.export.int8_tflite_path, cfg.data.test_csv, cfg.data.image_size)
        if cfg.data.test_csv.exists()
        else {"warning": f"missing test csv: {cfg.data.test_csv}"}
    )

    report = {
        "fp32_tflite_path": str(cfg.export.fp32_tflite_path),
        "int8_tflite_path": str(cfg.export.int8_tflite_path),
        "fp32_size_bytes": cfg.export.fp32_tflite_path.stat().st_size,
        "int8_size_bytes": cfg.export.int8_tflite_path.stat().st_size,
        "fp32_eval": fp32_eval,
        "int8_eval": int8_eval,
    }
    cfg.export.quant_report_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.export.quant_report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"Saved quantization report to {cfg.export.quant_report_path}")


if __name__ == "__main__":
    main()


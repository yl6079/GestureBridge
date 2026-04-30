from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import tensorflow as tf

def _preprocess_for_mobilenet(image: tf.Tensor) -> tf.Tensor:
    return tf.keras.applications.mobilenet_v3.preprocess_input(image)


@dataclass(slots=True)
class InferenceResult:
    label: str
    confidence: float
    latency_ms: float
    top_k: list[tuple[str, float]]


def _prepare_frame(frame: np.ndarray, image_size: int) -> np.ndarray:
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(frame_rgb, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    image = tf.convert_to_tensor(resized, dtype=tf.float32)
    image = _preprocess_for_mobilenet(image)
    image = tf.expand_dims(image, axis=0)
    return image.numpy()


def _load_labels(labels_path: Path) -> list[str]:
    return [line.strip() for line in labels_path.read_text(encoding="utf-8").splitlines() if line.strip()]


@dataclass(slots=True)
class ASL29TFLiteRuntime:
    model_path: Path
    labels_path: Path
    threads: int = 4
    image_size: int = 224
    top_k: int = 3
    labels: list[str] = field(init=False)
    interpreter: object = field(init=False)
    input_details: dict = field(init=False)
    output_details: dict = field(init=False)

    def __post_init__(self) -> None:
        self.labels = _load_labels(self.labels_path)
        self.interpreter = self._load_interpreter()
        self.input_details = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()[0]

    def _load_interpreter(self):
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
                        model_path=str(self.model_path),
                        experimental_delegates=[],
                        num_threads=1,
                        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF,
                    )
                except TypeError:
                    fallback = tf.lite.Interpreter(
                        model_path=str(self.model_path),
                        experimental_delegates=[],
                        num_threads=1,
                    )
                fallback.allocate_tensors()
                return fallback

        try:
            from tflite_runtime.interpreter import Interpreter

            interpreter = Interpreter(model_path=str(self.model_path), num_threads=self.threads)
        except ImportError:
            interpreter = tf.lite.Interpreter(model_path=str(self.model_path), num_threads=self.threads)
        return _allocate_with_fallback(interpreter)

    def predict(self, frame: np.ndarray) -> InferenceResult:
        t0 = perf_counter()
        input_tensor = _prepare_frame(frame, self.image_size)

        input_scale, input_zero_point = self.input_details["quantization"]
        output_scale, output_zero_point = self.output_details["quantization"]

        if self.input_details["dtype"] == np.int8:
            input_tensor = np.clip(np.round(input_tensor / input_scale + input_zero_point), -128, 127).astype(np.int8)
        elif self.input_details["dtype"] == np.uint8:
            input_tensor = np.clip(np.round(input_tensor / input_scale + input_zero_point), 0, 255).astype(np.uint8)
        else:
            input_tensor = input_tensor.astype(np.float32)

        self.interpreter.set_tensor(self.input_details["index"], input_tensor)
        self.interpreter.invoke()
        logits = self.interpreter.get_tensor(self.output_details["index"])
        if self.output_details["dtype"] in (np.int8, np.uint8) and output_scale > 0:
            logits = (logits.astype(np.float32) - output_zero_point) * output_scale

        probs = tf.nn.softmax(tf.convert_to_tensor(logits[0], dtype=tf.float32)).numpy()
        indices = np.argsort(probs)[::-1][: max(1, min(self.top_k, len(probs)))]
        top_k = [(self.labels[idx] if idx < len(self.labels) else f"class_{idx}", float(probs[idx])) for idx in indices]
        latency_ms = (perf_counter() - t0) * 1000.0
        return InferenceResult(
            label=top_k[0][0],
            confidence=top_k[0][1],
            latency_ms=latency_ms,
            top_k=top_k,
        )

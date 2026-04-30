from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

# tensorflow is heavy; only imported lazily (interpreter fallback + softmax).
# MobileNetV3Small in tf.keras builds a model that expects raw [0,255] inputs;
# the rescale to [-1,1] is a layer inside the model. preprocess_input is a
# no-op for MobileNetV3 in modern TF.


@dataclass(slots=True)
class InferenceResult:
    label: str
    confidence: float
    latency_ms: float
    top_k: list[tuple[str, float]]
    hand_detected: bool = True
    hand_bbox: tuple[int, int, int, int] | None = None


def _prepare_frame(frame: np.ndarray, image_size: int) -> np.ndarray:
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(frame_rgb, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    return resized.astype(np.float32)[None, ...]


def _prepare_cropped_rgb(rgb: np.ndarray, image_size: int) -> np.ndarray:
    # Caller has already produced a hand-centered RGB crop at image_size.
    return rgb.astype(np.float32)[None, ...]


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def _load_labels(labels_path: Path) -> list[str]:
    return [line.strip() for line in labels_path.read_text(encoding="utf-8").splitlines() if line.strip()]


@dataclass(slots=True)
class ASL29TFLiteRuntime:
    model_path: Path
    labels_path: Path
    threads: int = 4
    image_size: int = 224
    top_k: int = 3
    use_hand_crop: bool = False
    hand_cropper_model_path: Path | None = None
    labels: list[str] = field(init=False)
    interpreter: object = field(init=False)
    input_details: dict = field(init=False)
    output_details: dict = field(init=False)
    _cropper: object = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.labels = _load_labels(self.labels_path)
        self.interpreter = self._load_interpreter()
        self.input_details = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()[0]
        if self.use_hand_crop:
            from gesturebridge.pipelines.hand_crop import HandCropper, DEFAULT_MODEL_PATH
            try:
                self._cropper = HandCropper(
                    output_size=self.image_size,
                    model_path=self.hand_cropper_model_path or DEFAULT_MODEL_PATH,
                )
            except FileNotFoundError as exc:
                # Non-fatal: log once, fall back to plain resize. Demo-day insurance.
                print(f"[asl29_tflite] hand_crop disabled: {exc}", flush=True)
                self.use_hand_crop = False
                self._cropper = None

    def _load_interpreter(self):
        # Try lightweight runtimes first; fall back to full tensorflow only if
        # neither tflite_runtime nor ai_edge_litert is available.
        Interpreter = None
        try:
            from tflite_runtime.interpreter import Interpreter as _I  # type: ignore[import-not-found]
            Interpreter = _I
        except ImportError:
            try:
                from ai_edge_litert.interpreter import Interpreter as _I  # type: ignore[import-not-found]
                Interpreter = _I
            except ImportError:
                import tensorflow as tf  # heavy fallback
                Interpreter = tf.lite.Interpreter

        try:
            interpreter = Interpreter(model_path=str(self.model_path), num_threads=self.threads)
        except TypeError:
            interpreter = Interpreter(model_path=str(self.model_path))

        try:
            interpreter.allocate_tensors()
            return interpreter
        except RuntimeError as exc:
            if "XNNPACK" not in str(exc):
                raise
            print(f"[asl29_tflite] XNNPACK delegate failed; falling back to plain CPU: {exc}", flush=True)
            os.environ["TF_LITE_DISABLE_XNNPACK"] = "1"
            try:
                fallback = Interpreter(model_path=str(self.model_path), num_threads=1)
            except TypeError:
                fallback = Interpreter(model_path=str(self.model_path))
            fallback.allocate_tensors()
            return fallback

    def predict(self, frame: np.ndarray) -> InferenceResult:
        t0 = perf_counter()
        hand_detected = True
        hand_bbox = None
        if self.use_hand_crop and self._cropper is not None:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            crop_res = self._cropper.crop(frame_rgb)
            hand_detected = crop_res.found
            hand_bbox = crop_res.bbox
            if not crop_res.found:
                # No hand → return "nothing" without spending the classifier.
                latency_ms = (perf_counter() - t0) * 1000.0
                nothing_label = "nothing" if "nothing" in self.labels else self.labels[0]
                return InferenceResult(
                    label=nothing_label,
                    confidence=1.0,
                    latency_ms=latency_ms,
                    top_k=[(nothing_label, 1.0)],
                    hand_detected=False,
                    hand_bbox=None,
                )
            input_tensor = _prepare_cropped_rgb(crop_res.image, self.image_size)
        else:
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

        # Trained model already has softmax in the head; only re-apply if the
        # output looks like raw logits (e.g. INT8-dequantised tensors).
        out = logits[0].astype(np.float32)
        if out.min() < -0.01 or abs(out.sum() - 1.0) > 0.05:
            probs = _softmax(out)
        else:
            probs = out
        indices = np.argsort(probs)[::-1][: max(1, min(self.top_k, len(probs)))]
        top_k = [(self.labels[idx] if idx < len(self.labels) else f"class_{idx}", float(probs[idx])) for idx in indices]
        latency_ms = (perf_counter() - t0) * 1000.0
        return InferenceResult(
            label=top_k[0][0],
            confidence=top_k[0][1],
            latency_ms=latency_ms,
            top_k=top_k,
            hand_detected=hand_detected,
            hand_bbox=hand_bbox,
        )

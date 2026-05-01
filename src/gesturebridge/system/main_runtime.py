from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import os
from time import monotonic, sleep
import re

import cv2
import numpy as np

from gesturebridge.config import SystemConfig
from gesturebridge.pipelines.asl29_tflite import ASL29TFLiteRuntime, InferenceResult
from gesturebridge.pipelines.asr import OfflineASR
from gesturebridge.pipelines.tts import TTSOutput

NATO_TO_LETTER: dict[str, str] = {
    "alpha": "A",
    "bravo": "B",
    "charlie": "C",
    "delta": "D",
    "echo": "E",
    "foxtrot": "F",
    "golf": "G",
    "hotel": "H",
    "india": "I",
    "juliett": "J",
    "juliet": "J",
    "kilo": "K",
    "lima": "L",
    "mike": "M",
    "november": "N",
    "oscar": "O",
    "papa": "P",
    "quebec": "Q",
    "romeo": "R",
    "sierra": "S",
    "tango": "T",
    "uniform": "U",
    "victor": "V",
    "whiskey": "W",
    "whisky": "W",
    "xray": "X",
    "x-ray": "X",
    "yankee": "Y",
    "zulu": "Z",
}


@dataclass(slots=True)
class MainRuntime:
    config: SystemConfig
    infer: ASL29TFLiteRuntime
    asr: OfflineASR
    tts: TTSOutput
    landmark_classifier: object | None = None  # optional LandmarkClassifier for ensemble
    mode: str = "read"
    last_activity_ts: float = field(default_factory=monotonic)
    prediction_window: deque[tuple[str, float]] = field(default_factory=deque)
    latest_result: InferenceResult | None = None
    latest_tts: str = ""
    latest_transcript: str = ""
    learn_target: str = "A"
    learn_target_idx: int = 0
    latest_frame_jpeg: bytes = b""
    last_spoken_label: str = ""
    last_spoken_ts: float = 0.0
    last_infer_ts: float = 0.0
    last_response: dict[str, object] | None = None
    last_learn_feedback: str = ""
    last_learn_feedback_ts: float = 0.0

    def _set_placeholder_frame(self, title: str, subtitle: str = "") -> None:
        canvas = np.zeros((self.config.asl29.runtime.webcam_height, self.config.asl29.runtime.webcam_width, 3), dtype=np.uint8)
        cv2.putText(canvas, title, (22, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
        if subtitle:
            cv2.putText(canvas, subtitle, (22, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 2, cv2.LINE_AA)
        ok_jpg, jpg = cv2.imencode(".jpg", canvas)
        if ok_jpg:
            self.latest_frame_jpeg = jpg.tobytes()

    def touch(self) -> None:
        self.last_activity_ts = monotonic()

    def is_idle(self, idle_timeout: int) -> bool:
        return monotonic() - self.last_activity_ts >= idle_timeout

    def set_mode(self, mode: str) -> str:
        if mode not in {"read", "speech_to_sign", "learn"}:
            return self.mode
        self.mode = mode
        self.touch()
        if mode == "learn":
            if self.infer.labels:
                if self.learn_target not in self.infer.labels:
                    self.learn_target_idx = 0
                else:
                    self.learn_target_idx = self.infer.labels.index(self.learn_target)
                self.learn_target = self.infer.labels[self.learn_target_idx]
        if mode == "speech_to_sign":
            self._set_placeholder_frame("Speech to Sign", "Listening mode (no camera inference)")
        return self.mode

    def shift_learn_target(self, step: int) -> str:
        if not self.infer.labels:
            self.learn_target = "A"
            self.learn_target_idx = 0
            return self.learn_target
        count = len(self.infer.labels)
        self.learn_target_idx = (self.learn_target_idx + step) % count
        self.learn_target = self.infer.labels[self.learn_target_idx]
        self.touch()
        return self.learn_target

    def _maybe_ensemble(self, result: InferenceResult) -> tuple[str, float, dict | None]:
        """Combine MobileNet + landmark MLP if both are present.

        Decision rule (the why: ASL alphabet is highly geometric — landmark
        models tend to be more robust to lighting/skin/background. We trust
        landmark predictions unless MobileNet is very confident, on the
        assumption that high-confidence MobileNet wins on signs that are
        truly visual rather than geometric, e.g. those involving back-of-hand
        orientation that landmarks alone don't capture):

        - If no landmarks (no hand detected) → use MobileNet result as-is.
        - If MobileNet confidence >= 0.85 AND landmark confidence < 0.95 →
          use MobileNet (high-confidence visual override).
        - If both heads agree → use the agreed label, mean confidence.
        - Else → use landmark MLP (default trust).
        """
        if self.landmark_classifier is None or result.landmarks is None:
            return result.label, result.confidence, None
        try:
            lm_pred = self.landmark_classifier.predict(result.landmarks)
        except Exception as exc:
            print(f"[main_runtime] landmark predict failed: {exc}", flush=True)
            return result.label, result.confidence, None
        info = {"landmark_label": lm_pred.label, "landmark_confidence": lm_pred.confidence}
        if result.label == lm_pred.label:
            return result.label, (result.confidence + lm_pred.confidence) / 2.0, info
        if result.confidence >= 0.85 and lm_pred.confidence < 0.95:
            return result.label, result.confidence, info
        return lm_pred.label, lm_pred.confidence, info

    def process_camera_frame(self, frame) -> dict[str, object]:
        result = self.infer.predict(frame)
        self.latest_result = result
        self.touch()
        ensembled_label, ensembled_conf, ensemble_info = self._maybe_ensemble(result)
        self.prediction_window.append((ensembled_label, float(ensembled_conf)))
        if len(self.prediction_window) > self.config.asl29.runtime.stable_prediction_window:
            self.prediction_window.popleft()

        confidence_threshold = self.config.thresholds.prediction_confidence
        votes: dict[str, float] = {}
        for label, conf in self.prediction_window:
            if conf < confidence_threshold:
                continue
            votes[label] = votes.get(label, 0.0) + conf
        stable_label = max(votes.items(), key=lambda item: item[1])[0] if votes else "nothing"

        response: dict[str, object] = {
            "mode": self.mode,
            "label": ensembled_label,
            "stable_label": stable_label,
            "confidence": ensembled_conf,
            "latency_ms": result.latency_ms,
            "top_k": result.top_k,
            "mobilenet_label": result.label,
            "mobilenet_confidence": result.confidence,
            "hand_detected": result.hand_detected,
        }
        if ensemble_info:
            response.update(ensemble_info)
        if self.mode == "read":
            now = monotonic()
            cooldown = self.config.thresholds.tts_repeat_interval_seconds
            should_speak = (
                stable_label != self.last_spoken_label
                or now - self.last_spoken_ts >= cooldown
            )
            if should_speak and stable_label != "nothing":
                self.latest_tts = self.tts.speak(stable_label)
                self.last_spoken_label = stable_label
                self.last_spoken_ts = now
            response["tts"] = self.latest_tts
        elif self.mode == "learn":
            passed = stable_label == self.learn_target
            now = monotonic()
            cooldown = self.config.thresholds.tts_repeat_interval_seconds
            # In learn mode, announce only pass/fail feedback.
            learn_feedback = "true" if passed else "false"
            should_speak_feedback = (
                learn_feedback != self.last_learn_feedback
                or now - self.last_learn_feedback_ts >= cooldown
            )
            if should_speak_feedback:
                self.latest_tts = self.tts.speak(learn_feedback)
                self.last_learn_feedback = learn_feedback
                self.last_learn_feedback_ts = now
            response["target"] = self.learn_target
            response["passed"] = passed
            response["score"] = result.confidence
            response["tts"] = self.latest_tts
        return response

    def get_latest_frame_jpeg(self) -> bytes:
        return self.latest_frame_jpeg

    def run_speech_to_sign(self, utterance: str) -> dict[str, object]:
        transcript = self.asr.transcribe(utterance)
        self.latest_transcript = transcript
        self.touch()
        tokens = [tok for tok in re.split(r"\s+", transcript.strip().lower()) if tok]
        letters: list[str] = []
        for tok in tokens:
            if tok in NATO_TO_LETTER:
                letters.append(NATO_TO_LETTER[tok])
                continue
            letters.extend([ch.upper() for ch in tok if ch.isalpha()])
        if not letters:
            letters = ["NOTHING"]
        sign_assets: list[str] = []
        for letter in letters:
            if letter == "NOTHING":
                sign_assets.append("nothing.jpg")
            elif letter == "SPACE":
                sign_assets.append("space.jpg")
            elif letter == "DEL":
                sign_assets.append("del.jpg")
            else:
                sign_assets.append(f"{letter}.jpg")
        return {
            "mode": "speech_to_sign",
            "transcript": transcript,
            "letters": letters,
            "sign_assets": sign_assets,
        }

    def run_camera_loop(self) -> None:
        cap = None
        has_display = bool(os.environ.get("DISPLAY"))
        if not has_display:
            print("[main] DISPLAY is not set, running in headless camera mode")

        try:
            while True:
                if self.mode == "speech_to_sign":
                    if cap is not None:
                        cap.release()
                        cap = None
                    if not self.latest_frame_jpeg:
                        self._set_placeholder_frame("Speech to Sign", "Listening mode (no camera inference)")
                    sleep(0.05)
                    continue

                if cap is None:
                    cap = cv2.VideoCapture(self.config.asl29.runtime.camera_index)
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.asl29.runtime.webcam_width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.asl29.runtime.webcam_height)
                    if not cap.isOpened():
                        raise RuntimeError(f"Unable to open camera index {self.config.asl29.runtime.camera_index}")
                ok, frame = cap.read()
                if not ok:
                    continue
                now = monotonic()
                infer_interval_s = max(self.config.asl29.runtime.inference_interval_ms, 0) / 1000.0
                should_infer = self.last_response is None or (now - self.last_infer_ts) >= infer_interval_s
                if should_infer:
                    result = self.process_camera_frame(frame)
                    self.last_response = result
                    self.last_infer_ts = now
                else:
                    result = self.last_response
                overlay = frame.copy()
                text = f"{result['mode']} {result['stable_label']} {result['confidence']:.2f}"
                cv2.putText(overlay, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
                ok_jpg, jpg = cv2.imencode(".jpg", overlay)
                if ok_jpg:
                    self.latest_frame_jpeg = jpg.tobytes()
                if has_display:
                    cv2.imshow("GestureBridge Main Runtime", overlay)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        finally:
            if cap is not None:
                cap.release()
            if has_display:
                cv2.destroyAllWindows()

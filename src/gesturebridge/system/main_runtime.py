from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import os
import threading
from time import monotonic, sleep
import re

import cv2
import numpy as np

from gesturebridge.config import SystemConfig
from gesturebridge.system.mic_default import prefer_c270_default_mic
from gesturebridge.pipelines.asl29_tflite import ASL29TFLiteRuntime, InferenceResult
from gesturebridge.pipelines.asr import OfflineASR
from gesturebridge.pipelines.tts import TTSOutput


def _resample_pcm16_mono(pcm: bytes, src_sr: int, dst_sr: int) -> bytes:
    """Linear resample of int16 mono PCM (numpy only; avoids scipy)."""
    if src_sr == dst_sr or not pcm:
        return pcm
    x = np.frombuffer(pcm, dtype=np.int16).astype(np.float64)
    n_src = int(x.shape[0])
    n_dst = max(1, int(round(n_src * (dst_sr / src_sr))))
    if n_src <= 1:
        return pcm
    t_end = n_src / float(src_sr)
    t_src = np.linspace(0.0, t_end, num=n_src, endpoint=False)
    t_dst = np.linspace(0.0, t_end, num=n_dst, endpoint=False)
    y = np.interp(t_dst, t_src, x)
    y = np.clip(np.round(y), -32768, 32767).astype(np.int16)
    return y.tobytes()


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


# Phase 2: word-level speech-to-sign. Map spoken-word tokens to a video clip
# under assets/word_clips/. When a token hits this map AND the clip file
# exists at runtime, the response uses the clip instead of letter-spelling.
# Aliases (e.g. "thanks" -> thank_you) live here so we don't have to invent
# multiple clip files for synonyms.
WORD_CLIP_MAP: dict[str, str] = {
    "hello": "hello.mp4",
    "hi": "hello.mp4",
    "hey": "hello.mp4",
    "thanks": "thank_you.mp4",
    "thank": "thank_you.mp4",
    "thank_you": "thank_you.mp4",
    "thankyou": "thank_you.mp4",
    "yes": "yes.mp4",
    "yeah": "yes.mp4",
    "yep": "yes.mp4",
    "no": "no.mp4",
    "nope": "no.mp4",
    "help": "help.mp4",
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
    latest_speech_letters: list[str] = field(default_factory=list)
    latest_sign_assets: list[str] = field(default_factory=list)
    learn_target: str = "A"
    learn_target_idx: int = 0
    latest_frame_jpeg: bytes = b""
    last_spoken_label: str = ""
    last_spoken_ts: float = 0.0
    last_infer_ts: float = 0.0
    last_response: dict[str, object] | None = None
    last_learn_feedback: str = ""
    last_learn_feedback_ts: float = 0.0
    _vosk_stt: object | None = field(default=None, init=False, repr=False)
    _vosk_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _vosk_recording: bool = field(default=False, init=False, repr=False)
    _vosk_buffer: bytearray = field(default_factory=bytearray, init=False, repr=False)
    _vosk_stream: object | None = field(default=None, init=False, repr=False)
    _vosk_max_timer: threading.Timer | None = field(default=None, init=False, repr=False)
    _vosk_notification: dict[str, object] | None = field(default=None, init=False, repr=False)
    _vosk_capture_sr: int = field(default=16000, init=False, repr=False)

    @staticmethod
    def _prefer_real_mic_device_index(sd) -> int | None:
        """Pick capture device. Prefer PulseAudio virtual device so routing matches `pactl` default.

        Direct ALSA `hw:` / USB names can yield all-zero PCM when PipeWire/Pulse owns the device,
        while `pactl set-default-source` only affects apps that record through Pulse.
        """
        import os

        devices = sd.query_devices()

        def input_pairs():
            for i, d in enumerate(devices):
                if int(d.get("max_input_channels", 0) or 0) < 1:
                    continue
                yield i, str(d.get("name", ""))

        if os.environ.get("GESTUREBRIDGE_VOSK_SKIP_PULSE", "").strip() not in ("1", "true", "yes"):
            for i, name in input_pairs():
                nl = name.lower()
                if nl == "pulse" or nl.startswith("pulse:") or "pulseaudio" in nl:
                    return i
            for i, name in input_pairs():
                nl = name.lower()
                if "pipewire" in nl and "midi" not in nl:
                    return i

        # Prefer specific names before generic "usb" — logs showed index 0 "USB2.0 Device" matched
        # substring "usb" in "usb2.0" before Logitech C270 appeared later in the list (all-zero PCM).
        keywords_ordered = (
            "c270",
            "046d",
            "c925",
            "logitech",
            "logi",
            "hd webcam",
            "webcam",
            "headset",
            "usb audio",
            "microphone",
            "mic",
            "usb",
        )
        for kw in keywords_ordered:
            for i, name in input_pairs():
                nl = name.lower()
                if "hdmi" in nl and "monitor" not in nl:
                    continue
                if kw in nl:
                    return i
        for i, name in input_pairs():
            nl = name.lower()
            if "hdmi" in nl:
                continue
            return i
        return None

    @staticmethod
    def _find_pulse_or_pipewire_input_index(sd) -> int | None:
        """PortAudio may omit the substring 'pulse' in device names; match host API too."""
        try:
            apis = sd.query_hostapis()
        except Exception:
            apis = ()
        for i, d in enumerate(sd.query_devices()):
            if int(d.get("max_input_channels", 0) or 0) < 1:
                continue
            name_l = str(d.get("name", "")).lower()
            if "pulse" in name_l or "pulseaudio" in name_l:
                return i
            try:
                hi = int(d.get("hostapi", -1))
                if 0 <= hi < len(apis):
                    aname = str(apis[hi].get("name", "")).lower()
                    if "pulse" in aname or "pipewire" in aname:
                        return i
            except (TypeError, ValueError, KeyError):
                continue
        return None

    def _resolve_vosk_input_device(self):
        """PortAudio device index or None for library default."""
        import os

        import sounddevice as sd

        raw_env = os.environ.get("GESTUREBRIDGE_VOSK_INPUT_DEVICE", "").strip()
        choice: str | int | None
        if raw_env:
            choice = raw_env
        else:
            choice = self.config.vosk.input_device

        if choice is None or (isinstance(choice, str) and not str(choice).strip()):
            return self._prefer_real_mic_device_index(sd)

        if isinstance(choice, int):
            return choice
        s = str(choice).strip()
        if s.isdigit():
            return int(s)
        needle = s.lower()
        for i, d in enumerate(sd.query_devices()):
            if int(d.get("max_input_channels", 0) or 0) < 1:
                continue
            if needle in str(d.get("name", "")).lower():
                return i
        if needle == "pulse":
            idx = self._find_pulse_or_pipewire_input_index(sd)
            if idx is not None:
                return idx
            print(
                "[gesturebridge] GESTUREBRIDGE_VOSK_INPUT_DEVICE=pulse: no Pulse/PipeWire capture device "
                "in PortAudio's list; falling back to automatic device selection. "
                "Unset this env if you did not mean to override.",
                flush=True,
            )
            return self._prefer_real_mic_device_index(sd)
        raise RuntimeError(
            f"No input device name contains {choice!r}. "
            'Try: python -c "import sounddevice as sd; print(sd.query_devices())"'
        )

    def _pick_input_samplerate(self, device: int | str | None) -> int:
        """Use 16 kHz when supported; else device default / common rates (USB mics often lack 16 kHz)."""
        import sounddevice as sd

        try:
            if device is None:
                dev = sd.query_devices(kind="input")
            else:
                dev = sd.query_devices(device)
        except Exception:
            dev = sd.query_devices(sd.default.device[0])
        native = int(round(float(dev.get("default_samplerate", 48000))))
        candidates: list[int] = []
        for sr in (16000, native, 48000, 44100, 32000, 22050):
            if sr not in candidates:
                candidates.append(sr)
        for sr in candidates:
            try:
                sd.check_input_settings(device=device, channels=1, dtype="int16", samplerate=sr)
                return sr
            except Exception:
                continue
        return native

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
        if self.mode == "speech_to_sign" and mode != "speech_to_sign":
            self.abort_vosk_recording()
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
            prefer_c270_default_mic()
            self._set_placeholder_frame(
                "Speech to Sign",
                "Device microphone — use Start recording in the UI (offline Vosk)",
            )
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
        try:
            transcript = self.asr.transcribe(utterance)
        except ValueError as exc:
            # OfflineASR rejects empty/whitespace-only input; Vosk often returns "" for silence/no match.
            if "ASR_FAILURE" not in str(exc):
                raise
            transcript = ""
        self.latest_transcript = transcript
        self.touch()
        tokens = [tok for tok in re.split(r"\s+", transcript.strip().lower()) if tok]
        word_clips_dir = self.config.web.word_clips_dir
        # Per-token lookup: prefer a word clip (if one exists on disk); else letter-spell.
        # `letters` and `sign_assets` are kept aligned 1:1 so the UI can map indices.
        letters: list[str] = []
        sign_assets: list[str] = []
        for tok in tokens:
            if tok in NATO_TO_LETTER:
                letter = NATO_TO_LETTER[tok]
                letters.append(letter)
                sign_assets.append(f"{letter}.jpg")
                continue
            clip_name = WORD_CLIP_MAP.get(tok)
            if clip_name and (word_clips_dir / clip_name).exists():
                letters.append(tok.upper())
                sign_assets.append(clip_name)
                continue
            for ch in tok:
                if ch.isalpha():
                    letters.append(ch.upper())
                    sign_assets.append(f"{ch.upper()}.jpg")
        if not letters:
            letters = ["NOTHING"]
            sign_assets = ["nothing.jpg"]
        self.latest_speech_letters = list(letters)
        self.latest_sign_assets = list(sign_assets)
        return {
            "mode": "speech_to_sign",
            "transcript": transcript,
            "letters": letters,
            "sign_assets": sign_assets,
        }

    def _get_or_create_vosk_stt(self):
        if self._vosk_stt is None:
            from gesturebridge.pipelines.vosk_stt import VoskSTT

            self._vosk_stt = VoskSTT(self.config.vosk.model_dir.resolve())
        return self._vosk_stt

    def is_vosk_recording(self) -> bool:
        with self._vosk_lock:
            return self._vosk_recording

    def take_vosk_notification(self) -> dict[str, object] | None:
        with self._vosk_lock:
            n = self._vosk_notification
            self._vosk_notification = None
            return n

    def abort_vosk_recording(self) -> None:
        """Stop the microphone stream without transcription (e.g. mode switch)."""
        with self._vosk_lock:
            if not self._vosk_recording:
                return
            if self._vosk_max_timer is not None:
                self._vosk_max_timer.cancel()
                self._vosk_max_timer = None
            stream = self._vosk_stream
            self._vosk_stream = None
        if stream is not None:
            stream.stop()
            stream.close()
        with self._vosk_lock:
            self._vosk_recording = False
        self._vosk_buffer.clear()

    def start_vosk_recording(self) -> None:
        prefer_c270_default_mic()
        self._get_or_create_vosk_stt()
        import sounddevice as sd

        device = self._resolve_vosk_input_device()
        capture_sr = self._pick_input_samplerate(device)

        def callback(indata, _frames, _time_info, status) -> None:
            chunk = memoryview(indata).tobytes()
            with self._vosk_lock:
                if not self._vosk_recording:
                    return
                self._vosk_buffer.extend(chunk)

        with self._vosk_lock:
            if self._vosk_recording:
                raise RuntimeError("Already recording.")
            self._vosk_buffer.clear()
            self._vosk_capture_sr = capture_sr
            self._vosk_recording = True
            try:
                stream = sd.InputStream(
                    device=device,
                    samplerate=capture_sr,
                    channels=1,
                    dtype="int16",
                    callback=callback,
                    blocksize=4096,
                )
                stream.start()
                self._vosk_stream = stream
            except Exception:
                self._vosk_recording = False
                self._vosk_capture_sr = int(self.config.vosk.sample_rate)
                raise

        max_sec = max(0.0, float(self.config.vosk.max_record_sec))
        if max_sec > 0:
            self._vosk_max_timer = threading.Timer(max_sec, self._vosk_auto_stop)
            self._vosk_max_timer.daemon = True
            self._vosk_max_timer.start()

    def _vosk_auto_stop(self) -> None:
        with self._vosk_lock:
            if not self._vosk_recording:
                return
        try:
            result = self.stop_vosk_recording_and_run_speech_to_sign()
            with self._vosk_lock:
                self._vosk_notification = {"ok": True, "autostop": True, "result": result}
        except RuntimeError:
            pass
        except Exception as exc:
            with self._vosk_lock:
                self._vosk_notification = {"ok": False, "error": str(exc), "autostop": True}

    def stop_vosk_recording_and_run_speech_to_sign(self) -> dict[str, object]:
        with self._vosk_lock:
            if not self._vosk_recording:
                raise RuntimeError("Not recording.")
            if self._vosk_max_timer is not None:
                self._vosk_max_timer.cancel()
                self._vosk_max_timer = None
            stream = self._vosk_stream
            self._vosk_stream = None
        if stream is not None:
            stream.stop()
            stream.close()
        with self._vosk_lock:
            self._vosk_recording = False
        pcm = bytes(self._vosk_buffer)
        self._vosk_buffer.clear()
        if not pcm:
            raise RuntimeError("No audio captured.")
        arr = np.frombuffer(pcm, dtype=np.int16)
        peak_capture = int(np.max(np.abs(arr.astype(np.int64)))) if arr.size else 0
        if peak_capture == 0:
            print(
                "[gesturebridge] Vosk capture was silent (all samples zero). "
                "If `pactl` already shows the right mic, PortAudio may be using ALSA directly — "
                "try GESTUREBRIDGE_VOSK_INPUT_DEVICE=pulse or pick index from "
                '`python -c "import sounddevice as sd; print(sd.query_devices())"`.',
                flush=True,
            )
        target_sr = int(self.config.vosk.sample_rate)
        pcm16k = _resample_pcm16_mono(pcm, self._vosk_capture_sr, target_sr)
        stt = self._get_or_create_vosk_stt()
        text = stt.transcribe_pcm16_mono(pcm16k, target_sr)
        return self.run_speech_to_sign(text)

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
                        self._set_placeholder_frame(
                            "Speech to Sign",
                            "Device microphone — use Start recording in the UI (offline Vosk)",
                        )
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

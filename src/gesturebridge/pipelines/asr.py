from __future__ import annotations

import json
import os
from pathlib import Path


class OfflineASR:
    """Real offline ASR using Vosk + C270 microphone.

    Behavior:
    - If audio_text_proxy is a non-empty string: pass it through (test/demo mode).
    - If audio_text_proxy is empty/None: record from the real microphone and
      transcribe using Vosk.

    Falls back to stub passthrough when:
    - vosk is not installed
    - sounddevice is not installed
    - Vosk model directory does not exist
    - GESTUREBRIDGE_MOCK_ASR=1 is set

    The C270 microphone is ALSA card 3 on the Pi.
    Model path default: models/vosk-model-small-en-us-0.15
    """

    def __init__(
        self,
        min_chars: int = 1,
        model_path: str = "models/vosk-model-small-en-us-0.15",
        sample_rate: int = 16000,
        record_seconds: int = 4,
    ) -> None:
        self.min_chars = min_chars
        self.model_path = model_path
        self.sample_rate = sample_rate
        self.record_seconds = record_seconds

        self._use_mock = os.environ.get("GESTUREBRIDGE_MOCK_ASR", "0") == "1"
        self._model = None

        if not self._use_mock:
            try:
                from vosk import Model
                import sounddevice  # noqa: F401 — ensure available
                model_dir = Path(self.model_path)
                if model_dir.exists():
                    self._model = Model(str(model_dir))
                else:
                    self._use_mock = True
            except ImportError:
                self._use_mock = True

    def transcribe(self, audio_text_proxy: str = "") -> str:
        text = (audio_text_proxy or "").strip().lower()

        # Proxy string provided (tests / demo) — use it directly
        if text:
            if len(text) < self.min_chars:
                raise ValueError("ASR_FAILURE: empty or invalid speech input")
            return text

        # No proxy — record from mic and transcribe
        if self._use_mock or self._model is None:
            raise ValueError("ASR_FAILURE: vosk model not loaded and no proxy provided")

        return self._transcribe_from_mic()

    def _transcribe_from_mic(self) -> str:
        import sounddevice as sd
        from vosk import KaldiRecognizer

        rec = KaldiRecognizer(self._model, self.sample_rate)
        audio = sd.rec(
            int(self.record_seconds * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1,
            dtype="int16",
            device=None,  # system default; on Pi this resolves to C270 mic (card 3)
        )
        sd.wait()

        chunk = audio.tobytes()
        if rec.AcceptWaveform(chunk):
            result = json.loads(rec.Result())
        else:
            result = json.loads(rec.FinalResult())

        transcript = result.get("text", "").strip().lower()
        if len(transcript) < self.min_chars:
            raise ValueError("ASR_FAILURE: empty transcription")
        return transcript

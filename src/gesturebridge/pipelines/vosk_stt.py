"""Vosk offline speech-to-text (small English model, 16 kHz mono PCM)."""

from __future__ import annotations

import json
import threading
from pathlib import Path


class VoskSTT:
    """Thread-safe wrapper: one Model, new KaldiRecognizer per utterance."""

    def __init__(self, model_dir: Path) -> None:
        self._model_dir = Path(model_dir)
        self._model = None
        self._lock = threading.Lock()

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            from vosk import Model
        except ImportError as exc:
            raise RuntimeError(
                'Vosk is not installed. Run: pip install -e ".[speech]"'
            ) from exc
        if not self._model_dir.is_dir():
            raise RuntimeError(
                f"Vosk model not found at {self._model_dir}. Run: bash scripts/fetch_vosk_small.sh"
            )
        self._model = Model(str(self._model_dir))

    def transcribe_pcm16_mono(self, pcm: bytes, sample_rate: int = 16000) -> str:
        """Decode 16-bit little-endian mono PCM. Only 16000 Hz is supported."""
        if sample_rate != 16000:
            raise ValueError("Only sample_rate=16000 is supported for this Vosk model.")
        if not pcm:
            return ""
        self._ensure_model()
        try:
            from vosk import KaldiRecognizer
        except ImportError as exc:
            raise RuntimeError(
                'Vosk is not installed. Run: pip install -e ".[speech]"'
            ) from exc
        with self._lock:
            rec = KaldiRecognizer(self._model, 16000)
            chunk = 8000
            for i in range(0, len(pcm), chunk):
                rec.AcceptWaveform(pcm[i : i + chunk])
            out = json.loads(rec.FinalResult())
            return str(out.get("text", "") or "").strip()

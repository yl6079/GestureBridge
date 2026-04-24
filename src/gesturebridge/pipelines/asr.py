from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class OfflineASR:
    """A lightweight offline ASR placeholder.

    In production, this can be replaced with Vosk/Whisper.cpp, but we keep
    the same interface for integration and tests.
    """

    min_chars: int = 1

    def transcribe(self, audio_text_proxy: str) -> str:
        text = audio_text_proxy.strip().lower()
        if len(text) < self.min_chars:
            raise ValueError("ASR_FAILURE: empty or invalid speech input")
        return text

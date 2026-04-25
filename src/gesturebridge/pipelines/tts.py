from __future__ import annotations

import os


class TTSOutput:
    """Real TTS output using pyttsx3 + espeak-ng.

    Falls back to text-only stub when:
    - pyttsx3 is not installed
    - espeak-ng is not present on the system
    - GESTUREBRIDGE_MOCK_TTS=1 is set

    Always returns "[TTS] {text}" string regardless of mode,
    so existing tests continue to pass unchanged.
    """

    def __init__(self) -> None:
        self._use_mock = os.environ.get("GESTUREBRIDGE_MOCK_TTS", "0") == "1"
        self._engine = None

        if not self._use_mock:
            try:
                import pyttsx3
                engine = pyttsx3.init()
                engine.setProperty("rate", 150)
                engine.setProperty("volume", 1.0)
                self._engine = engine
            except Exception:
                # pyttsx3 not installed or espeak-ng not present — fall back silently
                self._use_mock = True

    def speak(self, text: str) -> str:
        if not self._use_mock and self._engine is not None:
            try:
                self._engine.say(text)
                self._engine.runAndWait()
            except Exception:
                pass  # Audio failure is non-fatal; text output still returned
        return f"[TTS] {text}"

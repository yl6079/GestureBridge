from __future__ import annotations


class TTSOutput:
    """Simple TTS abstraction.

    The demo returns text while production can route to a real speaker engine.
    """

    def speak(self, text: str) -> str:
        return f"[TTS] {text}"

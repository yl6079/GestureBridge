from __future__ import annotations

from dataclasses import dataclass
import shutil
import subprocess


@dataclass(slots=True)
class TTSOutput:
    """Simple TTS abstraction.

    The demo returns text while production can route to a real speaker engine.
    """

    speaker_cmd: str = ""

    def __post_init__(self) -> None:
        if self.speaker_cmd:
            return
        # Prefer espeak-ng when available; fall back to legacy espeak.
        self.speaker_cmd = shutil.which("espeak-ng") or shutil.which("espeak") or ""

    def speak(self, text: str) -> str:
        if self.speaker_cmd:
            try:
                subprocess.run([self.speaker_cmd, text], check=False, capture_output=True)
            except OSError:
                # Keep behavior non-fatal on devices without audio output.
                pass
        return f"[TTS] {text}"

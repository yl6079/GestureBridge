from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gesturebridge.pipelines.asr import OfflineASR
from gesturebridge.pipelines.classifier import QuantizedCentroidClassifier
from gesturebridge.pipelines.landmarks import LandmarkExtractor
from gesturebridge.pipelines.tts import TTSOutput
from gesturebridge.vocabulary import VocabularyItem


@dataclass(slots=True)
class TranslateMode:
    landmark_extractor: LandmarkExtractor
    classifier: QuantizedCentroidClassifier
    asr: OfflineASR
    tts: TTSOutput
    vocabulary: dict[int, VocabularyItem]
    prediction_threshold: float = 0.65
    labels_path: Path | None = None

    def _load_labels(self) -> list[str]:
        if not self.labels_path or not self.labels_path.exists():
            return []
        return [line.strip() for line in self.labels_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def _label_for_sign(self, sign_id: int) -> str:
        if self.labels_path and self.labels_path.exists():
            labels = self._load_labels()
            if sign_id < len(labels):
                return labels[sign_id]
        if sign_id in self.vocabulary:
            return self.vocabulary[sign_id].meaning
        return f"class_{sign_id}"

    def sign_to_speech(self, frame: np.ndarray) -> dict[str, str | float]:
        features = self.landmark_extractor.extract(frame).reshape(1, -1)
        probs = self.classifier.predict_proba(features)[0]
        sign_id = int(np.argmax(probs))
        confidence = float(probs[sign_id])
        if confidence < self.prediction_threshold:
            return {"status": "LOW_CONFIDENCE", "confidence": confidence}
        label = self._label_for_sign(sign_id)
        item = self.vocabulary.get(sign_id)
        spoken = label if item is None else item.tts_text
        return {
            "status": "OK",
            "meaning": label,
            "speech": self.tts.speak(spoken),
            "confidence": confidence,
        }

    def speech_to_sign(self, audio_text_proxy: str) -> dict[str, str | list[str]]:
        text = self.asr.transcribe(audio_text_proxy)
        # First try full-phrase lookup in existing vocabulary.
        for item in self.vocabulary.values():
            if item.meaning == text.lower():
                return {"status": "OK", "transcript": text, "image_key": item.image_key, "meaning": item.meaning}
        # Fallback for ASL alphabet flow: split into letters and map to sign assets.
        letters = [ch.upper() for ch in text if ch.isalpha()]
        if letters:
            return {
                "status": "OK",
                "transcript": text,
                "letters": letters,
                "image_keys": [f"{letter}.png" for letter in letters],
                "meaning": "".join(letters),
            }
        return {
            "status": "NO_MATCH",
            "transcript": text,
            "image_key": "unknown.png",
            "meaning": "unknown",
        }

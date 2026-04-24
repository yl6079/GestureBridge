from __future__ import annotations

from dataclasses import dataclass

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

    def sign_to_speech(self, frame: np.ndarray) -> dict[str, str | float]:
        features = self.landmark_extractor.extract(frame).reshape(1, -1)
        probs = self.classifier.predict_proba(features)[0]
        sign_id = int(np.argmax(probs))
        confidence = float(probs[sign_id])
        if confidence < self.prediction_threshold:
            return {"status": "LOW_CONFIDENCE", "confidence": confidence}
        item = self.vocabulary[sign_id]
        return {
            "status": "OK",
            "meaning": item.meaning,
            "speech": self.tts.speak(item.tts_text),
            "confidence": confidence,
        }

    def speech_to_sign(self, audio_text_proxy: str) -> dict[str, str]:
        text = self.asr.transcribe(audio_text_proxy)
        # Baseline keyword lookup: maps closest exact meaning in vocabulary.
        for item in self.vocabulary.values():
            if item.meaning == text:
                return {
                    "status": "OK",
                    "transcript": text,
                    "image_key": item.image_key,
                    "meaning": item.meaning,
                }
        return {
            "status": "NO_MATCH",
            "transcript": text,
            "image_key": "unknown.png",
            "meaning": "unknown",
        }

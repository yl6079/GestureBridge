from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from gesturebridge.pipelines.classifier import QuantizedCentroidClassifier
from gesturebridge.pipelines.landmarks import LandmarkExtractor
from gesturebridge.vocabulary import VocabularyItem


@dataclass(slots=True)
class LearnStats:
    attempts: int = 0
    correct: int = 0
    current_streak: int = 0
    best_streak: int = 0

    def register(self, is_correct: bool) -> None:
        self.attempts += 1
        if is_correct:
            self.correct += 1
            self.current_streak += 1
            self.best_streak = max(self.best_streak, self.current_streak)
        else:
            self.current_streak = 0

    @property
    def accuracy(self) -> float:
        if self.attempts == 0:
            return 0.0
        return self.correct / self.attempts


@dataclass(slots=True)
class LearnMode:
    landmark_extractor: LandmarkExtractor
    classifier: QuantizedCentroidClassifier
    vocabulary: dict[int, VocabularyItem]
    pass_threshold: float = 0.75
    stats: LearnStats = field(default_factory=LearnStats)

    def _evaluate_attempt(self, frame: np.ndarray, expected_sign_id: int) -> dict[str, str | float | bool]:
        features = self.landmark_extractor.extract(frame).reshape(1, -1)
        probs = self.classifier.predict_proba(features)[0]
        pred_sign_id = int(np.argmax(probs))
        confidence = float(probs[pred_sign_id])
        is_correct = pred_sign_id == expected_sign_id and confidence >= self.pass_threshold
        self.stats.register(is_correct)
        return {
            "is_correct": is_correct,
            "pred_sign_id": pred_sign_id,
            "confidence": confidence,
            "expected_sign_id": expected_sign_id,
        }

    def teaching_stage(self, frame: np.ndarray, target_sign_id: int) -> dict[str, str | float | bool]:
        result = self._evaluate_attempt(frame, target_sign_id)
        target = self.vocabulary[target_sign_id]
        result["prompt"] = f"Please perform sign: {target.meaning}"
        return result

    def practice_stage(self, frame: np.ndarray, target_sign_id: int) -> dict[str, str | float | bool]:
        result = self._evaluate_attempt(frame, target_sign_id)
        target = self.vocabulary[target_sign_id]
        result["prompt"] = f"Meaning only: {target.meaning}"
        return result

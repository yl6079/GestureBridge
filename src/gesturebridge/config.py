from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class RuntimeThresholds:
    activity_trigger: float = 0.5
    prediction_confidence: float = 0.65
    learn_pass_confidence: float = 0.75
    inactivity_seconds: int = 20


@dataclass(slots=True)
class ModelConfig:
    feature_dim: int = 63
    num_classes: int = 20
    model_path: Path = Path("artifacts/model_baseline.npz")
    quantized_model_path: Path = Path("artifacts/model_baseline_int8.npz")


@dataclass(slots=True)
class DataConfig:
    vocabulary_path: Path = Path("docs/vocabulary.csv")
    dataset_dir: Path = Path("data")
    train_path: Path = Path("data/train.npz")
    val_path: Path = Path("data/val.npz")
    test_path: Path = Path("data/test.npz")


@dataclass(slots=True)
class SystemConfig:
    thresholds: RuntimeThresholds = field(default_factory=RuntimeThresholds)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)

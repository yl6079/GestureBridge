from __future__ import annotations

from pathlib import Path

from gesturebridge.config import SystemConfig
import numpy as np

from gesturebridge.data import generate_synthetic_dataset
from gesturebridge.devices.rpi import RPiRuntime
from gesturebridge.devices.xiao import XIAODetector
from gesturebridge.modes.learn import LearnMode
from gesturebridge.modes.translate import TranslateMode
from gesturebridge.pipelines.asr import OfflineASR
from gesturebridge.pipelines.classifier import NearestCentroidClassifier, QuantizedCentroidClassifier
from gesturebridge.pipelines.landmarks import LandmarkExtractor
from gesturebridge.pipelines.tts import TTSOutput
from gesturebridge.state_machine import SystemStateMachine
from gesturebridge.system.controller import GestureBridgeController
from gesturebridge.vocabulary import load_vocabulary


def ensure_default_artifacts(config: SystemConfig) -> None:
    paths = [
        config.data.train_path,
        config.data.val_path,
        config.data.test_path,
        config.model.model_path,
        config.model.quantized_model_path,
    ]
    for p in paths:
        p.parent.mkdir(parents=True, exist_ok=True)

    centers_path = config.data.dataset_dir / "class_centers.npy"
    if centers_path.exists():
        class_centers = np.load(centers_path).astype(np.float32)
    else:
        rng = np.random.default_rng(5)
        class_centers = rng.normal(
            0,
            2.0,
            size=(config.model.num_classes, config.model.feature_dim),
        ).astype(np.float32)
        centers_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(centers_path, class_centers)

    if not config.data.train_path.exists():
        generate_synthetic_dataset(
            output_path=config.data.train_path,
            num_samples=1200,
            feature_dim=config.model.feature_dim,
            num_classes=config.model.num_classes,
            seed=7,
            class_centers=class_centers,
        )
    if not config.data.val_path.exists():
        generate_synthetic_dataset(
            output_path=config.data.val_path,
            num_samples=300,
            feature_dim=config.model.feature_dim,
            num_classes=config.model.num_classes,
            seed=17,
            class_centers=class_centers,
        )
    if not config.data.test_path.exists():
        generate_synthetic_dataset(
            output_path=config.data.test_path,
            num_samples=300,
            feature_dim=config.model.feature_dim,
            num_classes=config.model.num_classes,
            seed=23,
            class_centers=class_centers,
        )

    if not config.model.model_path.exists() or not config.model.quantized_model_path.exists():
        from gesturebridge.data import load_dataset

        x_train, y_train = load_dataset(config.data.train_path)
        float_model = NearestCentroidClassifier.fit(x_train, y_train, config.model.num_classes)
        float_model.save(config.model.model_path)
        quantized = QuantizedCentroidClassifier.from_float(float_model)
        quantized.save(config.model.quantized_model_path)


def build_controller(config: SystemConfig | None = None) -> GestureBridgeController:
    cfg = config or SystemConfig()
    ensure_default_artifacts(cfg)
    vocabulary = load_vocabulary(Path(cfg.data.vocabulary_path))
    quantized_classifier = QuantizedCentroidClassifier.load(cfg.model.quantized_model_path)
    extractor = LandmarkExtractor(output_dim=cfg.model.feature_dim)
    translate = TranslateMode(
        landmark_extractor=extractor,
        classifier=quantized_classifier,
        asr=OfflineASR(),
        tts=TTSOutput(),
        vocabulary=vocabulary,
        prediction_threshold=cfg.thresholds.prediction_confidence,
    )
    learn = LearnMode(
        landmark_extractor=extractor,
        classifier=quantized_classifier,
        vocabulary=vocabulary,
        pass_threshold=cfg.thresholds.learn_pass_confidence,
    )
    return GestureBridgeController(
        config=cfg,
        xiao=XIAODetector(threshold=cfg.thresholds.activity_trigger),
        rpi=RPiRuntime(),
        state_machine=SystemStateMachine(),
        translate_mode=translate,
        learn_mode=learn,
    )

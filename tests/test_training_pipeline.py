from pathlib import Path

from gesturebridge.bootstrap import ensure_default_artifacts
from gesturebridge.config import SystemConfig
from gesturebridge.data import load_dataset
from gesturebridge.pipelines.classifier import QuantizedCentroidClassifier


def test_artifacts_exist_and_model_predicts() -> None:
    cfg = SystemConfig()
    ensure_default_artifacts(cfg)
    assert cfg.model.quantized_model_path.exists()

    x_test, _ = load_dataset(cfg.data.test_path)
    model = QuantizedCentroidClassifier.load(Path(cfg.model.quantized_model_path))
    preds = model.predict(x_test[:5])
    assert preds.shape[0] == 5

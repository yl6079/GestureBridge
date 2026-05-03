from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class RuntimeThresholds:
    activity_trigger: float = 0.5
    # 0.08 was a near-zero filter (any prediction passed). With softmax over
    # 29 classes, uniform-random is 1/29 ≈ 0.034; a usable signal needs to
    # beat that handily. 0.4 is a calibrated starting point — revisit once
    # the new model's calibration plot is in (P5.2).
    prediction_confidence: float = 0.4
    learn_pass_confidence: float = 0.75
    inactivity_seconds: int = 20
    tts_repeat_interval_seconds: float = 1.5


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
class ASL29DataConfig:
    root_dir: Path = Path("data/asl29")
    raw_dir: Path = Path("data/asl29/raw")
    processed_dir: Path = Path("data/asl29/processed")
    splits_dir: Path = Path("data/asl29/splits")
    train_csv: Path = Path("data/asl29/splits/train.csv")
    val_csv: Path = Path("data/asl29/splits/val.csv")
    test_csv: Path = Path("data/asl29/splits/test.csv")
    labels_path: Path = Path("artifacts/asl29/labels.txt")
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    split_seed: int = 42
    image_size: int = 224


@dataclass(slots=True)
class ASL29TrainingConfig:
    batch_size: int = 32
    frozen_epochs: int = 10
    finetune_epochs: int = 20
    frozen_learning_rate: float = 1e-3
    finetune_learning_rate: float = 1e-4
    dropout: float = 0.2
    weight_decay: float = 1e-5
    patience: int = 5
    unfreeze_layers: int = 30
    random_seed: int = 42
    model_path: Path = Path("artifacts/asl29/checkpoints/best.keras")
    final_model_path: Path = Path("artifacts/asl29/checkpoints/final.keras")
    history_path: Path = Path("artifacts/asl29/training_history.json")
    metrics_path: Path = Path("artifacts/asl29/train_metrics.json")


@dataclass(slots=True)
class ASL29ExportConfig:
    fp32_tflite_path: Path = Path("artifacts/asl29/tflite/model_fp32.tflite")
    int8_tflite_path: Path = Path("artifacts/asl29/tflite/model_int8.tflite")
    quant_report_path: Path = Path("artifacts/asl29/tflite/quantization_report.json")
    representative_samples: int = 400


@dataclass(slots=True)
class ASL29EvalConfig:
    eval_dir: Path = Path("artifacts/asl29/eval")
    report_json: Path = Path("artifacts/asl29/eval/metrics.json")
    confusion_matrix_png: Path = Path("artifacts/asl29/eval/confusion_matrix.png")
    misclassifications_csv: Path = Path("artifacts/asl29/eval/high_confusions.csv")


@dataclass(slots=True)
class ASL29RuntimeConfig:
    benchmark_report_path: Path = Path("artifacts/asl29/benchmarks/rpi_tflite_report.json")
    benchmark_iterations: int = 200
    benchmark_warmup: int = 20
    tflite_threads: int = 4
    camera_index: int = 0
    preview_top_k: int = 3
    stable_prediction_window: int = 8
    inference_interval_ms: int = 300
    webcam_width: int = 640
    webcam_height: int = 480
    use_hand_crop: bool = True
    hand_landmarker_path: Path = Path("artifacts/mediapipe/hand_landmarker.task")


@dataclass(slots=True)
class SerialConfig:
    port: str = "/dev/ttyACM0"
    baudrate: int = 115200
    timeout_seconds: float = 0.25
    reconnect_seconds: float = 1.0
    heartbeat_timeout_seconds: int = 60
    human_on_token: str = "HUMAN_ON"
    human_off_token: str = "HUMAN_OFF"
    ping_token: str = "PING"
    err_prefix: str = "ERR"
    hand_label: str = "Hand"
    empty_label: str = "Empty"
    # Edge Impulse Hand/Empty scores: higher hand_on = stricter HUMAN_ON (less sensitive).
    hand_on_threshold: float = 0.55
    hand_off_threshold: float = 0.15
    # Higher empty_off = only very confident "empty" triggers HUMAN_OFF (fewer false offs).
    empty_off_threshold: float = 0.90


@dataclass(slots=True)
class DaemonConfig:
    poll_interval_seconds: float = 0.1
    idle_timeout_seconds: int = 300
    min_active_seconds: int = 8
    debounce_human_on_seconds: float = 0.8
    debounce_human_off_seconds: float = 2.0
    main_command: tuple[str, ...] = ("python", "-m", "gesturebridge.app", "--run-main")


@dataclass(slots=True)
class WebUIConfig:
    host: str = "127.0.0.1"
    port: int = 8080
    kiosk_url: str = "http://127.0.0.1:8080"
    assets_dir: Path = Path("assets/signs")
    auto_open_browser: bool = True
    kiosk_mode: bool = True


@dataclass(slots=True)
class VoskConfig:
    """Offline speech-to-text (Vosk small English). Download model to model_dir (see scripts/fetch_vosk_small.sh)."""

    model_dir: Path = Path("artifacts/vosk/vosk-model-small-en-us-0.15")
    sample_rate: int = 16000
    max_record_sec: float = 60.0
    #: PortAudio input: None = auto (prefer USB/webcam mic over silent defaults). Int = device index.
    #: Str = substring match on device name. Env GESTUREBRIDGE_VOSK_INPUT_DEVICE overrides when set.
    input_device: str | int | None = None


@dataclass(slots=True)
class ASL29Config:
    num_classes: int = 29
    class_names: tuple[str, ...] = tuple(
        [*(chr(ord("A") + idx) for idx in range(26)), "del", "nothing", "space"]
    )
    data: ASL29DataConfig = field(default_factory=ASL29DataConfig)
    training: ASL29TrainingConfig = field(default_factory=ASL29TrainingConfig)
    export: ASL29ExportConfig = field(default_factory=ASL29ExportConfig)
    evaluation: ASL29EvalConfig = field(default_factory=ASL29EvalConfig)
    runtime: ASL29RuntimeConfig = field(default_factory=ASL29RuntimeConfig)


@dataclass(slots=True)
class SystemConfig:
    thresholds: RuntimeThresholds = field(default_factory=RuntimeThresholds)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    asl29: ASL29Config = field(default_factory=ASL29Config)
    serial: SerialConfig = field(default_factory=SerialConfig)
    daemon: DaemonConfig = field(default_factory=DaemonConfig)
    web: WebUIConfig = field(default_factory=WebUIConfig)
    vosk: VoskConfig = field(default_factory=VoskConfig)

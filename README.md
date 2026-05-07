# GestureBridge

Real-time American Sign Language system on Raspberry Pi 5. Camera in, speech out, fully offline.

Yizheng Lin, Shufeng Chen. ELEN 6908, Spring 2026.

## Project Overview

GestureBridge is an edge-first ASL interaction system that runs entirely on-device (no cloud dependency). It supports:

- **ASL29 letter recognition** (A-Z + `del` + `nothing` + `space`) from webcam frames.
- **WLASL-100 word recognition** from 30-frame hand-landmark sequences.
- **Three UI modes** in a local web app:
  - **Read:** gesture to spoken output.
  - **Speech-to-sign:** offline speech input to sign assets (word clips first, then letter fallback).
  - **Trainer:** interactive letter practice with immediate feedback.
- **Low-power standby orchestration** via serial wake signals (`Hand` / `Empty`) from an external ESP32 sensor node.

## Results

### Letter recognition (29-class ASL alphabet)

| Model | Test accuracy | Notes |
|---|---|---|
| Original FP32 (leaky split) | 1.000 | Memorized; not useful |
| Honest FP32 MobileNetV3-Small | 0.802 | First credible baseline |
| Landmark MLP (detected hands) | 0.820 | 63-d MediaPipe vector |
| **Ensemble** | **0.829** | +2.7 pp over MobileNet alone |
| INT8 (Kaggle calibrated) | 0.218 | Distribution mismatch; do not deploy |

On Raspberry Pi 5, full hand-present inference is ~37.6 ms mean (10 runs). Frames with no detected hand short-circuit at ~9 ms.

### Word recognition (WLASL-100, dynamic signs)

| Model | Test top-1 | Test top-5 | Notes |
|---|---|---|---|
| Conv1D-Small (50K params) | 0.527 | 0.862 | 30-frame x 63-d landmark sequence |
| GRU-Small (80K params) | 0.502 | 0.837 | Same input |
| **Conv1D + GRU ensemble** | **0.577** | **0.870** | 0.5 / 0.5 softmax average; deployed |

Trained on WLASL-100 canonical split (12,888 clips, Kaggle `chinhde/wlasl-300-landmarks`, MIT). On Mac CPU the ensemble head is ~1 ms/clip; on Pi 5 it adds ~5 ms on top of landmark extraction.

## Three Bugs We Found and Fixed

1. **Frame leakage** from naive random split on sequential Kaggle frames.  
   Fix: contiguous split per class block.
2. **Invalid left-right augmentation** for chirality-sensitive letters.  
   Fix: remove random flip; keep mild crop-pad/hue jitter.
3. **Train/inference distribution mismatch** (tight 200x200 crops vs webcam full frame).  
   Fix: MediaPipe hand ROI crop with padding before MobileNet.

## Runtime Architecture

```
C270 camera  ->  MediaPipe HandLandmarker
                     |
                     +-- no hand  ->  "nothing"           [~9 ms]
                     |
                     +-- hand     ->  crop+resize 224     [~38 ms]
                                          |
                                          +-- MobileNetV3-Small (TFLite FP32)
                                          +-- Landmark MLP (optional, npz/tflite)
                                          +-- 30-frame landmark buffer
                                                   |
                                                   +-- Conv1D-Small (npz, numpy)
                                                   +-- GRU-Small    (npz, numpy)
```

Main entrypoint: `python -m gesturebridge.app`

Supported flags:

- `--run-main`: run camera loop + web UI.
- `--run-daemon`: run standby daemon (serial wake-trigger).
- `--benchmark-asl29`: benchmark letter runtime.
- `--demo`: synthetic controller demo path.
- `--camera-index N`: override camera index.
- `--speech "..."`: inject one utterance into speech-to-sign flow at startup.

## Quick Start

### 1) Development environment (training/evaluation workstation)

### Letter pipeline (29-class alphabet)

```bash
python3 -m venv .venv311
source .venv311/bin/activate
pip install -e ".[dev,ml,speech]"
```

### 2) Pi runtime environment (inference-focused)

`pyproject.toml` requires Python `>=3.10`; this project is validated around Python 3.11.  
Python 3.13 is explicitly blocked in runtime and redirected to `.venv311` if present.

```bash
python3 -m venv .venv311
source .venv311/bin/activate
pip install -e ".[speech]"
pip install opencv-python ai_edge_litert==2.1.4 mediapipe==0.10.18
```

### 3) Download required runtime assets

```bash
# Offline speech model
bash scripts/fetch_vosk_small.sh

# Optional word-level clip assets used by speech-to-sign mode
bash scripts/fetch_word_clips.sh
```

### 4) Start the app

```bash
# Auto-selects daemon mode when serial device exists; otherwise run-main
bash start_gesturebridge.sh

# Or run explicitly
python -m gesturebridge.app --run-main --camera-index 0
```

Open `http://127.0.0.1:8080`.

## Reproduction Pipelines

### ASL29 (letters)

```bash
# Download dataset
kaggle datasets download grassknoted/asl-alphabet

# Build contiguous split to avoid leakage
python scripts/prepare_asl29.py --split-mode contiguous

# Train MobileNetV3 sweep (typically CUDA box / Vast.ai)
bash scripts/vastai_train.sh

# Train landmark MLP head
python scripts/train_landmark_mlp.py

# Evaluate ensemble
python scripts/eval_ensemble.py \
  --mobilenet artifacts/asl29/tflite/model_fp32.tflite \
  --landmark-mlp artifacts/asl29/landmark_mlp/landmark_mlp.npz \
  --split-csv data/asl29/splits/test.csv
```

Useful companion scripts:

- `scripts/precrop_dataset.py`
- `scripts/extract_landmarks.py`
- `scripts/train_mobilenetv3_asl29.py`
- `scripts/evaluate_mobilenetv3_asl29.py`
- `scripts/eval_split.py`
- `scripts/eval_holdout_test.py`
- `scripts/export_tflite_int8_asl29.py`
- `scripts/benchmark_tflite_rpi.py`
- `scripts/run_realtime_asl29.py`

### WLASL-100 (word-level)

```bash
# 1) Pull pre-extracted holistic landmarks
kaggle datasets download chinhde/wlasl-300-landmarks -p data/wlasl_external/
unzip data/wlasl_external/wlasl-300-landmarks.zip -d data/wlasl_external/

# 2) Convert to GestureBridge format (N, 30, 63)
python scripts/convert_kaggle_wlasl100_landmarks.py

# 3) Train both heads
python scripts/train_wlasl100_pose.py --arch conv1d_small \
  --data data/wlasl100_kaggle/landmarks.npz \
  --labels data/wlasl100_kaggle/labels.txt \
  --out-dir artifacts/wlasl100

python scripts/train_wlasl100_pose.py --arch gru_small \
  --data data/wlasl100_kaggle/landmarks.npz \
  --labels data/wlasl100_kaggle/labels.txt \
  --out-dir artifacts/wlasl100

# 4) Single-clip sanity check
python scripts/predict_word_clip.py path/to/clip.mp4
```

Notes:

- `scripts/train_wlasl100_pose.py` defaults to `data/wlasl100/landmarks.npz` and `data/wlasl100/labels.txt` if arguments are omitted.
- Runtime auto-attaches word models only when files exist under `artifacts/wlasl100/` (Conv1D mandatory; GRU optional for ensemble).

## Runtime Configuration

Most defaults live in `src/gesturebridge/config.py` (`SystemConfig` dataclasses).

Key paths:

- Letter model: `artifacts/asl29/tflite/model_fp32.tflite`
- Letter labels: `artifacts/asl29/labels.txt`
- Hand landmarker: `artifacts/mediapipe/hand_landmarker.task`
- Word models: `artifacts/wlasl100/conv1d_small.npz` and `artifacts/wlasl100/gru_small.npz`
- Vosk model: `artifacts/vosk/vosk-model-small-en-us-0.15`
- UI sign assets: `assets/signs`
- UI word clips: `assets/word_clips`

Environment variables:

- `GESTUREBRIDGE_FORCE_DAEMON=1`: force daemon mode in `start_gesturebridge.sh`.
- `GESTUREBRIDGE_VOSK_INPUT_DEVICE`: override audio input device for Vosk capture.
- `GESTUREBRIDGE_VOSK_SKIP_PULSE=1`: skip Pulse/PipeWire preference logic.
- `GESTUREBRIDGE_DISABLE_TTS=1`: disable speech output.
- `FETCH_VOSK_FORCE_PROXY=1`: keep proxy env while downloading Vosk model.

## Deployment

### Rsync deployment helper

```bash
PI_PASS=<password> bash scripts/deploy_to_pi.sh
```

Supported env knobs in deploy script: `PI_USER`, `PI_HOST`, `PI_PATH`, `PI_PASS`.

### systemd units

Templates are in `deploy/systemd/`:

- `gesturebridge-daemon.service`
- `gesturebridge-main.service`

Adjust `WorkingDirectory` and virtualenv path before installing on target.

### Kiosk launcher

```bash
bash deploy/kiosk/open_kiosk.sh
# windowed mode:
KIOSK=0 bash deploy/kiosk/open_kiosk.sh
```

## Repository Walkthrough

```text
src/gesturebridge/
  app.py                    CLI entrypoint and runtime assembly
  config.py                 system-wide dataclass config defaults
  bootstrap.py              synthetic/demo controller bootstrap
  devices/
    xiao.py                 serial event parsing (Hand/Empty)
    rpi.py
  modes/
    translate.py
    learn.py
  pipelines/
    asl29_tflite.py         letter inference runtime + hand crop integration
    hand_crop.py
    landmark_classifier.py  landmark MLP head
    word_classifier.py      Conv1D word model inference (numpy)
    word_ensemble.py        Conv1D + GRU ensemble inference
    asr.py                  simple ASR interface
    vosk_stt.py             offline Vosk capture/transcribe utility
    tts.py                  text-to-speech output
  system/
    main_runtime.py         camera loop, mode logic, speech/sign bridging
    daemon.py               standby/wake process manager
    mic_default.py          C270 microphone preference helpers
  ui/
    web.py                  HTTP server, UI page, JSON API

scripts/
  data prep, training, evaluation, export, deployment, diagnostics

deploy/
  systemd/                  service templates
  kiosk/                    Chromium launcher

tests/
  test_daemon_serial.py
  test_state_machine.py
  test_translate_mode.py
  test_learn_mode.py
  test_training_pipeline.py
```

## Data and Artifact Policy

This repository intentionally ignores most large/generated outputs:

- `data/` is gitignored.
- `artifacts/` is gitignored.
- `assets/word_clips/*.mp4` is gitignored.

Expect to generate or download these locally before running full functionality.

## Hardware

Typical deployment hardware:

- Raspberry Pi 5 (8GB)
- Logitech C270 webcam (video + microphone)
- USB speaker
- ESP32 serial wake-trigger node (emits `Hand:` / `Empty:` style events)
- HDMI display (kiosk optional)

## Testing

```bash
pytest
```

Test suite covers daemon state transitions, mode logic, and parts of the training pipeline.

## Citations

This project builds on and uses the following resources. Please cite the
upstream datasets and models if you reuse this work.

### Datasets

- **WLASL** (word-level ASL recognition):
  Li, D., Rodriguez, C., Yu, X., & Li, H. (2020). *Word-level Deep Sign
  Language Recognition from Video: A New Large-scale Dataset and Methods
  Comparison.* WACV 2020.
  Repo: <https://github.com/dxli94/WLASL>. License: C-UDA (computational
  use only).
- **WLASL-100 pre-extracted MediaPipe Holistic landmarks**:
  Kaggle dataset *chinhde/wlasl-300-landmarks* (despite the name, ships
  the canonical WLASL-100 split). License: MIT.
  <https://www.kaggle.com/datasets/chinhde/wlasl-300-landmarks>
- **ASL Alphabet (29-class)**: Kaggle dataset
  *grassknoted/asl-alphabet*.
  <https://www.kaggle.com/datasets/grassknoted/asl-alphabet>
- **Word reference clips** (`assets/word_clips/`): aslbricks.org direct
  MP4s, signbsl.com Start ASL mirror, plus selected clips mined from the
  WLASL pool. Educational / academic use only — not redistributed.
  Per-clip provenance: `assets/word_clips/SOURCES.md`.

### Models and libraries

- **MediaPipe Hands** (21-landmark hand tracking):
  Zhang, F., Bazarevsky, V., Vakunov, A., et al. (2020). *MediaPipe
  Hands: On-device Real-time Hand Tracking.* CVPR Workshop.
  <https://google.github.io/mediapipe/solutions/hands>
- **MobileNetV3-Small** (letter image classifier backbone):
  Howard, A., Sandler, M., Chu, G., et al. (2019). *Searching for
  MobileNetV3.* ICCV 2019. arXiv:1905.02244.
- **Vosk** (offline speech recognition):
  Alpha Cephei. <https://alphacephei.com/vosk/>. Model used:
  `vosk-model-small-en-us-0.15`.
- **TensorFlow Lite + XNNPACK** (FP32 letter inference on Pi 5).
  <https://www.tensorflow.org/lite> · <https://github.com/google/XNNPACK>
- **pyttsx3** (cross-platform offline TTS).
  <https://github.com/nateshmbhat/pyttsx3>
- **NumPy / OpenCV / TensorFlow / scikit-learn** for training-time work
  on Mac; **numpy / ai_edge_litert / mediapipe / opencv-python** for the
  Pi inference path.

### Phase 2 design notes

- The Conv1D-Small / GRU-Small temporal heads on landmark sequences are
  small custom architectures. Pose-only WLASL baselines in the
  literature (e.g. Pose-TGCN, Li et al. 2020, and ASL Citizen baselines,
  Desai et al. NeurIPS 2023) report 60–70 % top-1 / 85–90 % top-5 on
  WLASL-100; our ensemble lands in the lower end of that band
  (test top-1 57.7 %, top-5 87.0 %).
- ASL Citizen: Desai, A., Berger, L., Minakov, F. P., et al. (2023).
  *ASL Citizen: A Community-Sourced Dataset for Advancing Isolated Sign
  Language Recognition.* NeurIPS 2023 Datasets & Benchmarks.
  <https://www.microsoft.com/en-us/research/project/asl-citizen/>

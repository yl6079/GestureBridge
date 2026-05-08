# GestureBridge

Real-time American Sign Language system on Raspberry Pi 5. Camera in, speech out, fully offline.

Yizheng Lin, Shufeng Chen. ELEN 6908, Spring 2026.

## Project Overview

GestureBridge runs the entire vision and speech pipeline on a single Raspberry Pi 5 with a Logitech C270, no cloud, no network. The system recognizes the 29-class ASL alphabet from webcam frames at 37.6 ms per frame with 82.9 % test accuracy on a leakage-free split, after fixing three latent training-pipeline bugs (frame leakage, ASL-incorrect mirror augmentation, train/inference distribution mismatch). It also plays word-level reference clips for spoken English via offline Vosk speech recognition with letter-spelling fallback for out-of-vocabulary words. An external ESP32 with a coarse hand classifier gates the heavy pipeline so the camera and ML stack only run when someone is signing.

A pose-only WLASL-100 word classifier is included as an optional extension; see the "Word recognition" section for results and limitations.

UI is a local web app on `localhost:8080` with three modes: Read (letter recognition + TTS, plus a Capture Word button for the extension), Speech-to-sign (offline speech to sign clips), Trainer (letter practice with feedback).

## Demo

Full system walkthrough on YouTube (click to play):

[![Full demo](https://img.youtube.com/vi/uCTLwS_5yoM/maxresdefault.jpg)](https://www.youtube.com/watch?v=uCTLwS_5yoM)

## Results

### Letter recognition (29-class)

The headline pipeline. Ensemble of MobileNetV3-Small (TFLite FP32 on Pi) and a 21-landmark MLP head, both fed by a single MediaPipe HandLandmarker pass. Honest 80 / 10 / 10 contiguous split, no frame leakage.

| Model | Test accuracy | Notes |
|---|---|---|
| Original FP32 (leaky split) | 1.000 | Memorized; not useful |
| Honest FP32 MobileNetV3-Small | 0.802 | First credible baseline |
| Landmark MLP (detected hands) | 0.820 | 63-d MediaPipe vector |
| **Ensemble (deployed)** | **0.829** | +2.7 pp over MobileNet alone |
| INT8 (Kaggle calibrated) | 0.218 | Distribution mismatch; do not deploy |

**On Raspberry Pi 5: ~37.6 ms / frame mean** (10-run benchmark) for hand-present inference, ~9 ms when no hand is detected and the pipeline short-circuits to `"nothing"`. Letter MobileNet alone at the synthetic-input benchmark on Pi: 4.5 ms avg, 6.1 ms p95 (i.e. the ensemble compute is dominated by MediaPipe, not the classifier).

### Training-pipeline bugs and fixes

These are the engineering meat of the project; each was a multi-percentage-point silent error in the prior baseline.

1. **Frame leakage** from naive random split on sequential Kaggle frames. Adjacent frames from the same recording session ended up on both sides of the split, so the model "memorized" rather than generalized. Fix: contiguous-block split per class (`scripts/prepare_asl29.py`).
2. **Invalid left-right augmentation** for chirality-sensitive letters. ASL fingerspelling depends on which hand the orientation references; mirror flips silently corrupt training signal for several glyphs. Fix: remove random flip; keep mild crop-pad / hue jitter only.
3. **Train / inference distribution mismatch.** Kaggle frames are tight 200×200 crops of an isolated hand; deployment frames are full-resolution webcam captures. Fix: front-load a MediaPipe HandLandmarker crop with 1.2× padding so deployment images match training distribution.

### Hardware trade-offs

We deliberately ship FP32 letter inference because INT8 post-training quantization collapsed accuracy from 0.802 to 0.218 (table above); calibration on the production distribution would require a longer effort than the project allowed. To keep FP32 inference inside the Pi 5's per-frame compute budget, the camera capture pipeline is run at:

- **640 × 480 resolution** (not the C270's 1280 × 720 max), which limits Pi-side decode and downscale work.
- **`inference_interval_ms = 300`** (≈ 3 fps inference cadence) by default. The camera grabber spins faster, but the heavy stack only runs every ~300 ms, leaving thermal headroom on the Cortex-A76.
- **`use_hand_crop = true`** so MediaPipe is the only source of full-frame compute; downstream MobileNet sees a 224 × 224 hand crop.

This is a real precision-vs-latency tradeoff: a higher-resolution camera path with INT8 inference would extract finer hand detail, but accuracy was unacceptable. We optimize for honest FP32 accuracy at lower capture resolution.

### Speech-to-sign

The reverse direction: spoken English in (offline Vosk small-en model on the C270 mic), visual sign output. **80 reference clips covering ~115 spoken-word tokens** including verb-conjugation and family-noun aliases (e.g. `going / went → go`, `told / telling → tell`, `mom → mother`, `done → finish`). Out-of-vocabulary words fall back to letter-by-letter fingerspelling. End-to-end ~1-2 s + audio length.

## Word recognition (WLASL-100)

Pose-only sequence classification on top of the existing MediaPipe stream, with no new hardware and no new accelerator. Ships as **pure numpy** on the Pi.

This is the project's stretch contribution, not its primary deliverable. The ensemble plus calibrated gating illustrate where landmark-only ASL recognition tops out under our hardware constraints, rather than a finished consumer-grade word recognizer.

### Models and accuracy

| Model | Test top-1 | Test top-5 | Notes |
|---|---|---|---|
| Conv1D-Small (50K params, Keras) | 0.527 | 0.862 | 30-frame × 63-d landmark sequence |
| GRU-Small (80K params, Keras) | 0.502 | 0.837 | Same input |
| Conv1D + GRU ensemble (prior) | 0.577 | 0.870 | 0.5 / 0.5 softmax average |
| BigConv1D (best single, A100, PyTorch) | 0.657 | 0.875 | 400 K params, attention pool, mixup |
| 3-seed BigConv1D mean | 0.657 | 0.908 | seeds 42/43/1337 |
| **5-way ensemble (deployed)** | **0.674** | **0.921** | 0.7 × (3 BigConv1D mean) + 0.3 × (Conv1D + GRU)/2 |

Trained on the canonical WLASL-100 split (12,888 clips, Kaggle `chinhde/wlasl-300-landmarks`, MIT). The deployed 5-way ensemble runs as **pure numpy** on the Pi (no PyTorch / TF on Pi at runtime); the three BigConv1D seeds were trained on an external A100 and exported to npz with bit-exact numpy forward (max diff 2 × 10⁻⁶ vs PyTorch).

### Latency

- **17 ms / clip** for the 5-way ensemble inference on Pi 5 (well under the 50 ms / frame budget).
- **1.3 s capture latency** from button press to result on Pi 5: the camera loop temporarily bypasses its 300 ms inference throttle while `_word_capturing` is true, so the 30-frame window fills at the natural pipeline rate (~22 fps) instead of the throttled 3 fps.

### Confidence gating

The deployed UI applies a calibrated probability threshold so users see honest "ambiguous" fallbacks instead of bad top-1 guesses:

| Setting | Value |
|---|---|
| Global threshold (calibrated on the 239-clip held-out test) | 0.48 |
| Coverage (fraction of captures shown as confident) | 66.5 % |
| **Precision when shown confidently** | **81.1 %** |
| When below threshold | UI renders top-3 + amber "Did you mean…?" banner |

`scripts/calibrate_word_ensemble.py` reproduces the threshold; the runtime auto-loads `artifacts/wlasl100/calibration.npz` if present.

### Limitations

- **Cross-signer drop is real.** WLASL-100's test split shares signers with train. Deployment-time accuracy on a new signer with the C270 will be lower than the held-out 67.4 %, and anecdotally we observed a substantial gap when testing with different signers under different lighting. We did not run a formal signer-disjoint evaluation in time for this report; that remains future work.
- **Camera-precision floor.** The 640 × 480 capture (mandated by the FP32 letter pipeline above) yields slightly degraded MediaPipe landmark precision compared to a 1280 × 720 capture. Higher resolution would likely improve word recognition but at the cost of letter inference latency.
- **Pose-only ceiling.** Published WLASL-100 baselines using full-body Holistic (543 keypoints) or graph-based skeleton models (ST-GCN family) report 65-70 % top-1, so our 67.4 % is in the upper half of pose-only baselines, not a state-of-the-art number.

### Dynamic gesture capture

The clip below shows a dynamic gesture performed in the Read tab. The user signs a letter, the word-mode UI buffers 30 frames over ~1.3 s, and the 5-way ensemble emits a top-5 list with calibrated confidence bars. The letter shown is not in the WLASL-100 vocabulary, so the clip does not measure word-recognition accuracy. It illustrates the dynamic path end to end: capture buffering, top-5 with confidence, and the gating threshold deciding between a confident top-1 and an "ambiguous" top-3.

<video src="https://github.com/user-attachments/assets/ad9de90b-48d2-4eaa-b123-0e8039870dfa" poster="docs/demo_poster.png" controls></video>

Other demos (letter recognition, speech-to-sign, Learner mode) are included in the project's presentation video.

## Runtime Architecture

```
C270 camera  ->  MediaPipe HandLandmarker
                     |
                     +-- no hand  ->  "nothing"           [~9 ms]
                     |
                     +-- hand     ->  crop+resize 224
                                          |
                                          +-- letter pipeline                       [~38 ms / frame]
                                          |    +-- MobileNetV3-Small (TFLite FP32)
                                          |    +-- Landmark MLP ensemble
                                          |
                                          +-- 30-frame rolling landmark buffer
                                                   |
                                                   +-- word pipeline (5-way ens.)   [~17 ms / clip]
                                                        +-- Conv1D-Small  (npz, numpy)
                                                        +-- GRU-Small     (npz, numpy)
                                                        +-- BigConv1D × 3 (npz, numpy, A100-trained)
                                                        +-- confidence gate (T=0.48)
```

Word capture latency (button press → top-5 + gating): **~1.3 s** on Pi 5.

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

The Kaggle ASL Alphabet dataset is split with `--split-mode contiguous` to avoid frame-level leakage. The MobileNetV3 sweep is GPU-bound (we used a Vast.ai RTX 4090); everything else runs on CPU.

```bash
kaggle datasets download grassknoted/asl-alphabet
python scripts/prepare_asl29.py --split-mode contiguous
bash scripts/vastai_train.sh
python scripts/train_landmark_mlp.py
python scripts/eval_ensemble.py \
  --mobilenet artifacts/asl29/tflite/model_fp32.tflite \
  --landmark-mlp artifacts/asl29/landmark_mlp/landmark_mlp.npz \
  --split-csv data/asl29/splits/test.csv
```

See `scripts/` for the data-prep, training, export, and evaluation helpers.

### WLASL-100 (word-level)

Pre-extracted Holistic landmarks are pulled from Kaggle and converted to the GestureBridge `(N, 30, 63)` format. The Keras Conv1D and GRU baselines train on a Mac CPU in ~30 min combined; the BigConv1D swarm is GPU-bound (~90 s per seed on an A100). Each PyTorch checkpoint is exported to `.npz` so the Pi runtime stays NumPy-only.

```bash
kaggle datasets download chinhde/wlasl-300-landmarks -p data/wlasl_external/
unzip data/wlasl_external/wlasl-300-landmarks.zip -d data/wlasl_external/
python scripts/convert_kaggle_wlasl100_landmarks.py

python scripts/train_wlasl100_pose.py --arch conv1d_small \
  --data data/wlasl100_kaggle/landmarks.npz \
  --labels data/wlasl100_kaggle/labels.txt \
  --out-dir artifacts/wlasl100
python scripts/train_wlasl100_pose.py --arch gru_small \
  --data data/wlasl100_kaggle/landmarks.npz \
  --labels data/wlasl100_kaggle/labels.txt \
  --out-dir artifacts/wlasl100

for seed in 42 43 1337; do
  python scripts/train_conv1d_a100.py --epochs 120 --seed $seed \
    --out-dir artifacts/wlasl100_a100_conv1d_s$seed
  python scripts/export_bigconv1d_to_npz.py \
    --ckpt artifacts/wlasl100_a100_conv1d_s$seed/ckpts/best.pt \
    --out  artifacts/wlasl100/bigconv1d_s$seed.npz
done

python scripts/calibrate_word_ensemble.py
python scripts/eval_a100_ensemble.py
python scripts/predict_word_clip.py path/to/clip.mp4
```

`scripts/calibrate_word_ensemble.py` writes the 0.48 global threshold to `artifacts/wlasl100/calibration.npz`. The Pi runtime auto-attaches whichever word models exist under `artifacts/wlasl100/`: with all three BigConv1D `.npz` files plus the Conv1D, GRU, and calibration files present, it assembles the deployed 5-way ensemble; otherwise it falls back to fewer heads. The Keras and PyTorch pipelines coexist: BigConv1D is the preferred backbone, the Keras heads remain in the ensemble for diversity.

#### Optional: signer-conditioned fine-tune

Adapts the word backbone to a single signer's recordings on a small custom vocabulary. The resulting checkpoint coexists with, and does not replace, the 100-class ensemble.

```bash
.venv311/bin/python scripts/record_demo_vocab.py --takes 5
python scripts/finetune_demo_words.py \
  --backbone artifacts/wlasl100_a100_conv1d/ckpts/best.pt \
  --out-dir artifacts/wlasl5
```

## Runtime Configuration

Default model paths and runtime parameters are declared in `src/gesturebridge/config.py` (`SystemConfig` dataclasses). Runtime environment-variable overrides are documented at the top of `start_gesturebridge.sh`.

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
  WLASL pool. Educational / academic use only, not redistributed.
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

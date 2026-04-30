# GestureBridge

Real-time ASL letter recognition on a Raspberry Pi 5 — camera in, speech out.

Yizheng Lin · Shufeng Chen — ELEN 6908, Spring 2026

---

## Results

| Model | Test acc | Notes |
|---|---|---|
| Original FP32 (leaky split) | 1.000 | Memorized — useless |
| New FP32 MobileNetV3-Small | 0.802 | Honest baseline after fix |
| Landmark MLP (detected hands) | 0.820 | 63-d MediaPipe vector |
| **Ensemble** | **0.829** | +2.7 pp over MobileNet alone |
| INT8 (Kaggle-calibrated) | 0.218 | Distribution mismatch — do not deploy |

On-device: **37.6 ms / frame** mean on Raspberry Pi 5 (C270 webcam, FP32, MediaPipe + MobileNet).  
Short-circuit when no hand detected: **9 ms** (skips classifier entirely).

Branch `shufeng` contains all improvements. Branch `yizheng` is the original baseline.

---

## What We Fixed

Three latent bugs masked a 20-point accuracy gap between reported (100%) and real-camera (~0.08 confidence) performance:

1. **Frame leakage** — `train_test_split(stratify=y)` on 3,000 sequential frames per class puts adjacent frames in train and test. Fixed with a contiguous-block split (frames 1–2400 train / 2401–2700 val / 2701–3000 test).
2. **ASL-incorrect flip** — `tf.image.random_flip_left_right` breaks chirality-sensitive signs (J/L, J/Z trajectory). Removed; replaced with small crop-pad jitter and hue jitter.
3. **Distribution mismatch at inference** — Kaggle images are 200×200 hand-filled squares; C270 frames are 640×480 with the hand at ~20% of area. Fixed by inserting MediaPipe HandLandmarker → crop ROI → 224×224 before MobileNet.

---

## System

```
C270 → MediaPipe HandLandmarker
        ├── (no hand) → "nothing"     [9 ms]
        └── (hand)    → crop+resize → MobileNetV3-Small  [38 ms]
                                   ↓
                                   ├── label, confidence
                                   └── 21×3 landmarks → 63-d MLP
                                                       ↓
                                                       ensemble
```

Web UI (kiosk browser on Pi) has three modes:
- **Read** — camera → predicted letter → TTS
- **Speech-to-sign** — browser STT → reference image
- **Trainer** — target letter + true/false feedback

An ESP32 with a PIR sensor wakes/sleeps the app when someone walks up.

---

## Reproduction

```bash
# 1. Dataset (Kaggle ASL Alphabet, ~87k images)
kaggle datasets download grassknoted/asl-alphabet

# 2. Prepare with contiguous split
python scripts/prepare_asl29.py --split-mode contiguous

# 3. Train (GPU recommended — see scripts/vastai_train.sh for cloud setup)
bash scripts/vastai_train.sh          # 4-config sweep, ~25 min on RTX 4090

# 4. Train landmark MLP (CPU, <5 min)
python scripts/train_landmark_mlp.py

# 5. Evaluate ensemble
python scripts/eval_ensemble.py \
  --mobilenet artifacts/asl29/tflite/model_fp32.tflite \
  --landmark-mlp artifacts/asl29/landmark_mlp/landmark_mlp.npz \
  --split-csv data/asl29/splits/test.csv

# 6. Deploy to Pi
PI_PASS=<password> bash scripts/deploy_to_pi.sh
```

---

## Directory Layout

```
src/gesturebridge/      core app (pipelines, web UI, daemon, config)
scripts/                training, eval, export, deploy, diagnostics
artifacts/asl29/        model outputs (tflite, landmark MLP, eval JSONs)
data/asl29/             dataset splits (CSV manifests)
report/                 ACM paper draft, Marp slides, demo video script
notes/                  improvement run log, Pi validation photos, conventions
tests/                  pytest suite
```

---

## Environment

Python 3.11. On the Pi, **do not install TensorFlow** — mediapipe and TF have an irreconcilable protobuf version conflict. The runtime uses `ai_edge_litert` instead (already handled by the lazy-import chain in `pipelines/asl29_tflite.py`).

```bash
python3 -m venv .venv311
source .venv311/bin/activate
pip install -e ".[dev,ml]"
```

Pi-specific:

```bash
pip install ai_edge_litert==2.1.4 mediapipe==0.10.18
# do NOT pip install tensorflow on the Pi
```

---

## Hardware

Raspberry Pi 5 (8 GB) · Logitech C270 USB webcam · generic USB speaker  
ESP32-WROOM-32 with HC-SR501 PIR · 7" HDMI LCD · all powered from Pi USB-C PSU

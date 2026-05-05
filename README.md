# GestureBridge

Real-time American Sign Language interpreter on a Raspberry Pi 5. Camera in, speech out. No cloud, no internet.

Yizheng Lin, Shufeng Chen. ELEN 6908, Spring 2026.

## Background

Most production sign-language interpreters today depend on cloud inference, which introduces latency, privacy exposure, and a hard requirement for an internet uplink that is not always available in classrooms, clinics, or assistive contexts. GestureBridge runs the entire vision and speech pipeline on a single Raspberry Pi 5 paired with an ESP32 motion sensor, demonstrating that a complete bidirectional ASL interaction system fits comfortably inside the compute budget of a fanless edge device. The system covers two granularities of recognition — the **29-letter ASL alphabet** for fingerspelling and **WLASL-100 word-level signs** for natural ASL — across three modes: gesture to speech, speech to reference image (with word-level video clips), and an interactive trainer with true / false feedback.

## Results

### Letter recognition (29-class ASL alphabet)

![Honest evaluation and on-device latency](docs/results_summary.png)

| Model | Test accuracy | Notes |
|---|---|---|
| Original FP32 (leaky split) | 1.000 | Memorized; not useful |
| Honest FP32 MobileNetV3-Small | 0.802 | First credible baseline |
| Landmark MLP (detected hands) | 0.820 | 63-d MediaPipe vector |
| **Ensemble** | **0.829** | +2.7 pp over MobileNet alone |
| INT8 (Kaggle calibrated) | 0.218 | Distribution mismatch; do not deploy |

On the Raspberry Pi 5, one full inference takes 37.6 ms mean (median 37.7 ms) over 10 runs. Frames with no detected hand short-circuit at 9 ms.

### Word recognition (WLASL-100, dynamic signs)

Letters by themselves are unnatural for fluent ASL; native signers use word-level glosses and reserve fingerspelling for proper nouns. We train a lightweight pose-only sequence classifier on top of the existing MediaPipe HandLandmarker stream so word recognition reuses the same per-frame detection (no new hardware, no new accelerator).

| Model | Test top-1 | Test top-5 | Notes |
|---|---|---|---|
| Conv1D-Small (50K params) | 0.527 | 0.862 | 30-frame × 63-d landmark sequence |
| GRU-Small (80K params) | 0.502 | 0.837 | Same input |
| **Conv1D + GRU ensemble** | **0.577** | **0.870** | 0.5 / 0.5 softmax average; deployed |

Trained on the canonical WLASL-100 split, 12,888 clips total (Kaggle `chinhde/wlasl-300-landmarks`, MIT). On Mac CPU the ensemble runs at ~1 ms / clip; on the Pi 5 it adds an estimated 5 ms on top of the existing landmark extraction.

**30 of 100 classes hit 100 % test top-1** with mean confidence ≥ 67 %. The 20 strongest of those are curated as a demo vocabulary at `artifacts/wlasl100/demo_vocab.txt`: water, tea, chair, table, bed, shirt, pencil, orange, dance, work, travel, finish, enjoy, have, new, wrong, many, problem, class, snow.

The remaining classes are honest: noisy 2-3-clip per-class test buckets and class confusions consistent with WLASL pose-only baselines (~60-70 % top-1 in the literature). For arbitrary in-the-wild signing the model is best read top-3 with on-screen confidence.

## Three latent bugs we found and fixed

The starting point was a system reporting 100 % held-out accuracy that delivered 0.08 confidence on the actual webcam. Three bugs combined to produce that gap.

1. **Frame leakage.** `train_test_split(stratify=y)` on the Kaggle ASL Alphabet (3,000 sequential frames per class) puts adjacent video frames into both train and test. The model memorized the recording. Fix: contiguous-block split, frames 1 to 2400 train, 2401 to 2700 val, 2701 to 3000 test.
2. **ASL-incorrect flip augmentation.** `tf.image.random_flip_left_right` breaks chirality-sensitive signs (J, L, J/Z trajectory). Fix: removed the flip, added small crop-pad and hue jitter.
3. **Distribution mismatch at inference.** Kaggle images are 200×200 hand-filled squares; C270 frames are 640×480 with the hand at roughly 20 % of area. Fix: insert MediaPipe HandLandmarker, crop a 25 % padded ROI, resize to 224×224 before MobileNet.

## System

```
C270 camera  ->  MediaPipe HandLandmarker
                     |
                     +-- no hand  ->  "nothing"           [ 9 ms]
                     |
                     +-- hand     ->  crop+resize 224     [38 ms]
                                          |
                                          +--  MobileNetV3-Small (TFLite FP32)        }
                                          +--  Landmark MLP (63-d, sklearn -> npz)    }  letter ensemble
                                          |                                                  -> letter label
                                          +--  30-frame landmark buffer                }
                                                  |                                       word ensemble (button)
                                                  +-- Conv1D-Small  (npz, numpy)       }       -> word label
                                                  +-- GRU-Small     (npz, numpy)       }
```

Web UI on `localhost:8080` with three modes:

- **Read.** Live camera, letter ensemble label, TTS playback. Includes a "Capture Word (1 s)" button that buffers 30 landmark frames and reports top-5 WLASL-100 predictions with confidence.
- **Speech-to-sign.** Two-step record button, offline Vosk on the C270 mic. Returns word-level video clips for known glosses (`hello`, `thanks`, `yes`, `no`, `help`, plus aliases) and falls back to letter spelling for any out-of-vocabulary word.
- **Trainer.** Random target letter, spoken true or false feedback.

An on-board ML model on the ESP32 (XIAO ESP32S3 + Edge Impulse hand classifier) emits `Hand: …` / `Empty: …` over USB serial; the Pi gates the heavy pipeline on this signal so power stays low when no one is signing.

## Reproduction

### Letter pipeline (29-class alphabet)

```bash
# Dataset
kaggle datasets download grassknoted/asl-alphabet

# Contiguous split (fixes frame-leakage bug from the Kaggle naive split)
python scripts/prepare_asl29.py --split-mode contiguous

# Train sweep on a CUDA box (~25 min on RTX 4090)
bash scripts/vastai_train.sh

# Train the landmark MLP (CPU, under 5 min)
python scripts/train_landmark_mlp.py

# Evaluate the ensemble
python scripts/eval_ensemble.py \
  --mobilenet artifacts/asl29/tflite/model_fp32.tflite \
  --landmark-mlp artifacts/asl29/landmark_mlp/landmark_mlp.npz \
  --split-csv data/asl29/splits/test.csv
```

### Word pipeline (WLASL-100, dynamic signs)

```bash
# 1. Pre-extracted MediaPipe Holistic landmarks, MIT licensed
kaggle datasets download chinhde/wlasl-300-landmarks -p data/wlasl_external/
unzip data/wlasl_external/wlasl-300-landmarks.zip -d data/wlasl_external/

# 2. Convert to our (T, 63) tensor format with right-hand selection
#    + chirality flip when only the left hand is detected.
python scripts/convert_kaggle_wlasl100_landmarks.py
# → data/wlasl100_kaggle/landmarks.npz, shape (12888, 30, 63)

# 3. Train Conv1D-Small + GRU-Small on the same data (CPU is fine)
python scripts/train_wlasl100_pose.py --arch conv1d_small \
  --data data/wlasl100_kaggle/landmarks.npz \
  --labels data/wlasl100_kaggle/labels.txt \
  --out-dir artifacts/wlasl100
python scripts/train_wlasl100_pose.py --arch gru_small \
  --data data/wlasl100_kaggle/landmarks.npz \
  --labels data/wlasl100_kaggle/labels.txt \
  --out-dir artifacts/wlasl100

# 4. Sanity-check on a single clip (CLI)
python scripts/predict_word_clip.py path/to/some_clip.mp4

# 5. Run the full app with both pipelines wired in
python -m gesturebridge.app --run-main --camera-index 0
# Boot log shows: "WLASL-100 ensemble attached (Conv1D+GRU, 100 classes)"
# → http://127.0.0.1:8080, Read tab, "Capture Word (1s)" button
```

### Deploy to the Pi

```bash
PI_PASS=<password> bash scripts/deploy_to_pi.sh
```

## Repository layout

```
src/gesturebridge/         core app (pipelines, web UI, daemon, config)
  pipelines/
    asl29_tflite.py        letter MobileNet runtime + MediaPipe crop
    landmark_classifier.py letter landmark MLP head (sklearn → npz)
    word_classifier.py     WLASL-100 Conv1D, pure-numpy Pi inference
    word_ensemble.py       Conv1D + GRU ensemble + numpy GRU forward
scripts/                   training, eval, export, deploy, diagnostics
  prepare_wlasl100.py             WLASL video downloader (direct + yt-dlp)
  extract_wlasl_landmarks.py      video → (N, 30, 63) MediaPipe tensor
  convert_kaggle_wlasl100_landmarks.py  pre-extracted Holistic → our format
  train_wlasl100_pose.py          Conv1D / GRU training + npz export
  predict_word_clip.py            CLI predictor for a single mp4
artifacts/asl29/           letter outputs (tflite, landmark_mlp, eval JSONs)
artifacts/wlasl100/        word outputs (conv1d_small.npz, gru_small.npz, demo_vocab.txt)
data/asl29/                letter dataset split manifests
data/wlasl100_kaggle/      word landmark tensor (gitignored)
tests/                     pytest suite
```

## Environment

Python 3.11 on the Mac for development, training, and export. On the Pi the runtime uses `ai_edge_litert` and `mediapipe`; do not install full TensorFlow on the Pi because its protobuf dependency conflicts with mediapipe.

```bash
# Mac side (development)
python3 -m venv .venv311
source .venv311/bin/activate
pip install -e ".[dev,ml]"

# Pi side (runtime only)
pip install ai_edge_litert==2.1.4 mediapipe==0.10.18
```

## Hardware

Raspberry Pi 5 (8 GB), Logitech C270 USB webcam, USB speaker, ESP32-WROOM-32 with HC-SR501 PIR, 7-inch HDMI LCD. Powered from the Pi USB-C PSU.

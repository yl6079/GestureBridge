# GestureBridge: Real-Time ASL Recognition on Raspberry Pi 5

*Yizheng Lin, Shufeng Chen — ELEN 6908, Spring 2026*

> **Note for the writer:** This is a content draft (~1700 words plus
> tables/figures). Convert to ACM `acmart` two-column at submission time.
> All numbers come from `notes/improvement_run_log.md` and the eval JSONs
> under `artifacts/asl29/eval/`. Figure suggestions inline as `[FIG: ...]`.

---

## Abstract (~120 words)

GestureBridge is an interactive American Sign Language (ASL) interpreter
deployed on a Raspberry Pi 5 with a USB webcam, USB speaker, and an
ESP32-based ambient sensor for state management. The system recognizes
the 29-class ASL alphabet in real time, speaks recognized letters
through TTS, and supports a "speech-to-sign" learning mode driven by
**fully offline Vosk speech recognition** running locally on the C270's
microphone (no cloud, no internet dependency). Our key contribution is a
*measurement* contribution:
we identify and fix three latent bugs in the project's training
pipeline (frame-leakage in the train/test split, ASL-incorrect
horizontal flip augmentation, and missing inference-time hand
preprocessing) that masked a 20-point gap between reported accuracy
(100%) and honest test accuracy (80%). After the fixes, an ensemble
of a fine-tuned MobileNetV3-Small and a 63-d MediaPipe-landmark MLP
achieves **82.9 %** on a leakage-free test split at **38 ms per
frame** on a Pi 5.

## 1. Introduction (~250 words)

[FIG 1: System hardware photo — Pi 5 + LCD + C270 + ESP32 + speaker.]

ASL recognition for educational and accessibility use cases is
appealing on the Pi because the hardware is cheap, fanless, and
HDMI-out makes deployment trivial. Off-the-shelf datasets (notably the
Kaggle "ASL Alphabet" set with ~87 k images across 29 classes) make a
working classifier achievable within a course timeline. **The
pernicious failure mode**, which our investigation surfaces, is that
naive use of these datasets produces a model that scores ~100 % on its
own test split and yet fails completely on a real webcam.

Our contribution is **a credibility audit** of one such pipeline. We
identify three latent bugs that combine to produce the reported-vs-real
gap, fix each, retrain, and report the first honest generalization
number (80.2 % MobileNet alone; 82.9 % ensemble) for this dataset on a
contiguous-frame split. We deploy the resulting model on the actual Pi
hardware in the lab and validate end-to-end inference at 37.6 ms /
frame. We argue that *splitting protocols and inference-time
preprocessing* deserve at least as much attention as architecture
choice in coursework projects, and that demonstrating real-camera
robustness must be a default, not an afterthought.

## 2. System overview (~250 words)

[FIG 2: System block diagram — C270 → MediaPipe HandLandmarker →
{MobileNetV3, Landmark-MLP} → ensemble decision → smoothing → TTS /
web UI; ESP32 → USB-serial → daemon (idle/active state machine).]

GestureBridge runs three coordinating processes on the Pi 5:

1. **Daemon** — listens on USB serial for `HUMAN_ON` / `HUMAN_OFF`
   tokens emitted by the ESP32 (PIR sensor). Spawns the main app on
   `HUMAN_ON`, kills it after `idle_timeout_seconds` of `HUMAN_OFF`.
2. **Main app** (`gesturebridge.app --run-main`) — opens the C270 at
   `/dev/video0`, runs the inference pipeline at 300 ms cadence,
   maintains a stable-prediction window, and exposes a FastAPI web UI
   on `localhost:8080` (kiosk-mode browser) for read mode (camera +
   live label + TTS), speech-to-sign mode (offline Vosk STT on C270 mic → reference
   image), and trainer mode (target letter + true/false feedback).
3. **ESP32 firmware** (`esp32_camera.ino`) — drives a PIR + onboard
   LED + USB-serial heartbeat; debounced human-on / human-off events.

The inference pipeline (§4) is the focus of this paper.

## 3. Problem: a 20-point gap (~250 words)

The starting point of our work was a deployed system on the Pi
reporting **100 % held-out test accuracy** in `metrics.json`, while
the developer's anecdotal experience with the C270 was that "almost
nothing crosses the 0.65 confidence threshold; even the gestures that
land at top-1 only have confidence 0.08–0.09." The threshold had been
manually lowered to 0.08 to make the demo flow visible.

Three bugs combine to produce this gap:

- **B1. Train/test leakage from per-image stratified split.** The
  Kaggle ASL Alphabet contains 3,000 sequential frames per class from
  a single recording session. `sklearn.train_test_split` selects rows
  uniformly at random with stratification by label, which places
  adjacent frames into both `train` and `test`. The model effectively
  memorizes the recording. On a contiguous-block split (frames 1–2400
  train, 2401–2700 val, 2701–3000 test, never adjacent) the same
  trained model still scores 100 % because every frame was seen in
  training; only the *split methodology* differs.

- **B2. Horizontal flip in augmentation.** The original
  `tf.image.random_flip_left_right` is silently wrong for ASL.
  Several signs are chirality-dependent (J vs L, the J/Z trajectory),
  and the alphabet contains direction-sensitive shapes. Flipping
  effectively trains the model to view "left" and "right" as
  identical.

- **B3. Distribution mismatch at inference.** Kaggle ASL Alphabet
  images are pre-cropped to ~200×200 hand-filled squares with a
  uniform black background. C270 frames are 640×480 with the hand
  occupying ~20 % of the area against arbitrary lab background. The
  deployed runtime fed the raw C270 frame directly to the
  224×224-trained MobileNet, yielding the reported 0.08–0.09
  confidences regardless of the gesture.

## 4. Methods (~600 words)

### 4.1 Honest splitting and a re-evaluation harness (P0)

We add `--split-mode {random,contiguous}` to `prepare_asl29.py` and a
`sanity_check_split.py` that asserts zero filename overlap between
splits and prints per-class frame-index ranges. With the new
contiguous split (69,600 / 8,700 / 8,700), the existing FP32 model
still scores 100 % — confirming B1 alone is not sufficient, and
re-training is required.

### 4.2 Inference-time hand crop with MediaPipe (P1)

We insert a MediaPipe HandLandmarker (Tasks API,
`hand_landmarker.task`, 7.5 MB) before the MobileNet input. When a
hand is detected, we crop a square ROI around the landmarks bounding
box with 25 % padding and resize to 224×224. When no hand is detected,
the runtime short-circuits to `nothing` and skips the classifier
entirely (saving ~30 ms per frame and avoiding spurious labels).

We deliberately do **not** pre-crop the *training* set. Kaggle ASL
Alphabet images are already hand-filled, so cropping at training time
is approximately the identity. The crop's value is exclusively at
inference time, to bring real C270 frames closer to the trained
distribution.

### 4.3 Augmentation correction and re-training sweep (P2)

We removed `random_flip_left_right` and added small random crop+pad
(~8 % jitter) plus random hue (±0.05). On a rented vast.ai RTX 4090
24 GB box, we ran a 4-config sweep over backbone unfreeze depth and
dropout. Each run completed in ~6 minutes; the full sweep in ~25 min.

| tag | unfreeze | dropout | best val acc |
|---|---|---|---|
| c_u15_d20 | 15 | 0.2 | 0.799 |
| c_u30_d20 | 30 | 0.2 | 0.874 |
| **c_u50_d20** | **50** | **0.2** | **0.886** |
| c_u30_d30 | 30 | 0.3 | 0.870 |

The winner (`c_u50_d20`) was exported to FP32 TFLite (3.7 MB) and
INT8 TFLite (1.2 MB).

### 4.4 Landmark MLP and ensemble (P3)

ASL is intrinsically geometric, so we train a second classifier on
the 21×3 MediaPipe landmark vector (wrist-centered, scale-normalized →
63-d). A scikit-learn MLP with hidden layers (256, 128) and ReLU
activations trains in <2 minutes on a CPU. Weights are exported to a
199 KB `.npz` so the Pi runtime requires only NumPy (no extra TF).

Ensemble decision rule (in `MainRuntime._maybe_ensemble`): if both
heads agree, return the agreed label with the mean confidence; if
MobileNet is highly confident (≥0.85) while the landmark MLP is
unsure (<0.95), trust MobileNet; otherwise trust the landmark MLP
because of its geometric prior.

### 4.5 INT8 quantization attempt (P4)

We exported a fully INT8-quantized MobileNet (1.2 MB), calibrated on
the Kaggle training distribution. Because Kaggle and C270 distributions
are markedly different (§3 B3), the resulting INT8 scale factors
collapse predictions to a small set of always-on classes; test accuracy
drops to **21.8 %**. Lacking on-device calibration data at the time of
writing, **we ship FP32**. Future work: capture ~500 frames from the
deployed C270 and re-quantize.

## 5. Results (~400 words)

[FIG 3: top-confusion bar chart from `ensemble_metrics.json`.]

All numbers below are on the contiguous test split (8,700 samples,
frames 2701–3000 per class, never adjacent to any training frame).

| Model | Test acc | Macro-F1 | Notes |
|---|---|---|---|
| Original FP32 (Yizheng's) on leaky split | 1.000 | 1.000 | Reproducible but **uninformative** |
| Original FP32 on contiguous split | 1.000 | 1.000 | Memorized the recording |
| New FP32 (c_u50_d20) on contiguous test | **0.802** | **0.783** | **Honest baseline** |
| Landmark-MLP only (detected hands, n=4,672) | 0.820 | — | Top-3: 0.871 |
| Landmark-MLP scored over full test (n=8,700) | 0.471 | — | Penalised by no-hand frames |
| **MN + LM ensemble (full test)** | **0.829** | — | **+2.7 pp over MN alone** |
| INT8 (Kaggle-calibrated) | 0.218 | — | Calibration distribution mismatch; do not deploy |

Top remaining confusions in the ensemble are (true → pred): V→K,
Q→P, M↔N, S↔E, Y→L, X/T/J/Z→nothing. The "→nothing" failures are
expected for J and Z (single-frame trajectory signs) and indicate
MediaPipe's hand detector occasionally misses unusual finger
configurations.

[FIG 4: Pi-side empty-room C270 frame and pipeline output, showing
short-circuit to `nothing` at conf 1.0 in 37 ms.]

**On-device measurement.** On the Raspberry Pi 5 in the lab, a single
inference (MediaPipe hand check + conditional MobileNet) takes
**37.6 ms mean / 37.7 ms median over 10 runs** on a 640×480 C270
frame. This is well below the 300 ms scheduling cadence. With the
short-circuit, no-hand frames cost ~9 ms and skip the MobileNet
entirely.

**Confidence threshold.** With the leaky model, the deployed
`prediction_confidence` was 0.08 (random over 29 classes is 1/29 ≈
0.034). With the new model + hand crop, the calibrated default is
0.4. A final calibration sweep on real C270 frames is left for the
deployment week.

## 6. Discussion (~200 words)

The single most consequential change in this project was **none of
the model code**: it was changing how the dataset is split. A
4-line addition to `prepare_asl29.py` (sort by frame index, take
first/middle/last block) flipped the credibility of every downstream
number. This is the pattern coursework projects most often miss, in
our experience: there is no friction to writing
`train_test_split(stratify=y)`, and it produces beautiful loss
curves.

The ASL-flip bug is similarly low-friction: every TF tutorial uses
`random_flip_left_right` as its default augmentation. For domains
where chirality or directionality matter, the default is wrong and
no error message appears.

The deployed-vs-trained distribution gap (B3) was solvable by adding
a hand-cropper at inference, but **only** because we were willing to
accept a small first-stage cost (MediaPipe at ~10 ms). A pure
end-to-end model would have required collecting C270 training data —
a much bigger lift.

A late but consequential design decision was the speech-recognition path.
Our initial implementation used the browser's Web Speech API, which is
trivial to wire up but fails the moment the Pi has no internet — exactly
the demo condition. We replaced it with a fully offline Vosk small-model
pipeline running on the C270's built-in microphone, with a two-step
record-button UI so the user controls when capture starts. The system is
now end-to-end offline with no network dependency at any stage, which
matches the privacy and edge-deployment motivation of the project rather
than merely paying lip service to it.

## 7. Limitations and future work (~120 words)

- **No on-device INT8.** P4 (capture C270 calibration set, re-export)
  is well-defined and would yield ~3× model size reduction and ~2×
  inference speedup, but is left for the demo week.
- **Single recording session.** All 87 k training images come from
  one signer. Multi-signer generalization is not measured here, and
  is the next obvious threat to validity.
- **Trajectory signs (J, Z).** Frame-level classification cannot
  capture motion; we treat these as confusable with `nothing` for
  now. A small temporal model on a sliding window would address them.

## 8. Conclusion (~80 words)

Three latent bugs in the original GestureBridge pipeline produced a
reported 100 % accuracy that did not survive contact with a real
webcam. By fixing the train/test leakage, removing ASL-incorrect
horizontal flip augmentation, and adding inference-time hand
cropping plus a landmark-MLP ensemble, we achieve an honest 82.9 %
on a leakage-free test split at 37.6 ms per frame on a Raspberry
Pi 5. The fixes are small, but the credibility delta is 18 points.

---

## Appendix A — Reproduction

- Code: <github URL on push>; branch `shufeng`.
- Dataset: `kaggle datasets download grassknoted/asl-alphabet`.
- Splits: `python scripts/prepare_asl29.py --split-mode contiguous`.
- Train: `bash scripts/vastai_train.sh` (4-config sweep on a CUDA box).
- Eval: `python scripts/eval_ensemble.py --mobilenet ... --landmark-mlp
  ... --split-csv data/asl29/splits/test.csv`.
- Deploy: `PI_PASS=... bash scripts/deploy_to_pi.sh`.

## Appendix B — Hardware

Raspberry Pi 5 (8 GB), Logitech C270 USB webcam, generic USB speaker,
ESP32-WROOM-32 with HC-SR501 PIR, 7" HDMI LCD, all powered from the
Pi's USB-C PSU.

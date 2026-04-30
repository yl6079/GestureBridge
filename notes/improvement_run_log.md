# Improvement Run Log — `shufeng` branch

## 2026-04-30 — port-then-improve merge

- Merged `origin/yizheng@81c5226` tree into `shufeng` (commit `820a541`).
- Pre-merge state preserved as tag `shufeng-pre-merge-2026-04-30`.
- Both histories reachable via `git log --all --graph`.

## 2026-04-30 — root cause confirmed

- `prepare_asl29.py` original splitter used `train_test_split` per-image.
  Kaggle ASL Alphabet has 3000 sequential frames per class from one
  recording session, so adjacent video frames leak across train/val/test.
- Yizheng's reported accuracy 1.0 is real on his split — the model
  memorized each recording.
- Current deployed `model_fp32.tflite` (3.7MB, on Pi 2026-04-30 07:11):
  - Kaggle 28-image holdout: **28/28 = 100%** with confidence ~1.000
    on every sample (the test images are from the same session as train,
    not actually OOD — well-known dataset quirk).
  - Contiguous-block test split (frames 2701-3000 per class, never
    adjacent to train frames 1-2400): **8700/8700 = 100%**.
  - Conclusion: cannot measure honest generalization with this model;
    it has seen every frame in the dataset. Must retrain.

## 2026-04-30 — P0 deliverables (this commit)

- `scripts/prepare_asl29.py` gains `--split-mode {random,contiguous}`
  (default contiguous) and `--no-copy` for fast iteration.
- `scripts/sanity_check_split.py` verifies zero overlap and prints
  per-class frame-index ranges.
- `scripts/eval_holdout_test.py` runs a TFLite model on the Kaggle
  28-image holdout dir (free OOD-ish signal).
- `scripts/eval_split.py` runs a TFLite model on any split CSV with
  per-class F1 + top confusions.
- New splits generated under `data/asl29/splits/{train,val,test}.csv`
  (69600 / 8700 / 8700 samples; frames 1-2400 / 2401-2700 / 2701-3000
  per class).

## 2026-04-30 — P1: hand-crop preprocessing wired into inference

- `pipelines/hand_crop.py` (MediaPipe HandLandmarker; downloaded
  `hand_landmarker.task` to `artifacts/mediapipe/`).
- `pipelines/asl29_tflite.py` runs the cropper before MobileNet when
  `use_hand_crop=True`; on no-hand short-circuits to "nothing"
  without spending the classifier (~15 ms total).
- TF made a lazy import; runtime now happy with just
  `tflite_runtime` or `ai_edge_litert` plus `mediapipe`. Lighter on
  the Pi.
- Decision (revised): not pre-cropping the train set. Kaggle ASL
  Alphabet train images are already hand-filled (crop ≈ identity).
  The crop's value is at *inference* time on the C270 to bring real
  frames closer to the trained distribution. Train-side, strong
  augmentation (P2) is the lever.

## 2026-04-30 — P2: ASL-correct augmentation; bug found

- Removed `tf.image.random_flip_left_right` from data pipeline — it
  was a real bug. Horizontal flip changes ASL meaning (J vs L,
  J/Z trajectory, every chirality-sensitive sign). Yizheng's previous
  training relied on this so the model was effectively told "left
  and right are the same" which is wrong.
- Added small random crop+pad (~8% jitter) and random hue (±0.05)
  to broaden train-time distribution.
- Sweep runner: `scripts/vastai_train.sh` runs 4 configs in box;
  `scripts/vastai_remote_sync_and_train.sh` does end-to-end from Mac.

## 2026-04-30 — first honest training number

Sweep run #1 (`c_u15_d20`: unfreeze 15 layers, dropout 0.2,
ASL-correct augmentation, no horizontal flip) on the rented
vast.ai 4090 24GB, contiguous split:

- Train accuracy: ~0.97
- **Val accuracy on contiguous split: 0.72–0.73** (top-3: ~0.90)
- Wall-clock: ~5 min

Compared to Yizheng's reported 1.0 / 1.0 on the leaky split, this is
the first credible generalization number for our pipeline. The
train→val gap (~0.25) confirms substantial overfitting; the rest of
the sweep + ensemble + further augmentation targets that gap.

## 2026-04-30 — P3: landmark MLP + ensemble

- 63-d wrist-centered, scale-normalized hand landmark vector → 2-layer
  MLP (~30K params).
- Extraction running on Mac CPU (M4 Pro, ~110 img/s). Miss rate at
  default MediaPipe confidence 0.3 was ~15%; lowered to 0.05 → ~5%.
- Ensemble rule (`MainRuntime._maybe_ensemble`): trust landmark MLP
  by default for ASL alphabet's geometric structure; high-confidence
  MobileNet (≥0.85, while landmark <0.95) overrides; agreement →
  mean confidence.

## 2026-04-30 — P4 prep + P5 first round

- `scripts/capture_calibration_set.py`: countdown-based C270 frame
  capture for INT8 calibration (run on Pi when devices free).
- `scripts/export_tflite_int8_asl29.py` gains `--calibration-dir`
  and `--int8-float-output`. Critical fix: keep the softmax in
  float32; calibrate from C270 frames not Kaggle train images.
- `prediction_confidence` floor 0.08 → 0.4 (uniform-random over 29
  classes is 0.034; old value was effectively no filter).
- `pipelines/asl29_tflite.py`: log XNNPACK fallback instead of
  swallowing silently.

## Pending

- Pull best-of-sweep TFLite from 4090, eval on contiguous test.
- Train landmark MLP locally once extraction finishes (~10 min after).
- `eval_ensemble.py` to compare MobileNet-only / landmark-only /
  ensemble on the test split — picks the headline.
- P4 capture + INT8 retry — needs Yizheng coordination on the Pi.
- P5.4: demo recording once headline model is settled.

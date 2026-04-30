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

## Next steps

- P1: hand-crop preprocessing (MediaPipe palm) before MobileNet input,
  applied symmetrically to training data and runtime.
- P2: retrain MobileNetV3-Small on 5090 with the contiguous split, with
  and without hand-crop, to isolate the crop's contribution.
- P4: real OOD test — capture a small set of C270 frames in the lab
  (needs Yizheng coordination) so we can finally measure real-world
  accuracy.

# Hardware-induced trade-offs in GestureBridge

Real-time on-device ASL recognition on a Raspberry Pi 5 + Logitech C270
hits a small set of binding constraints. We made deliberate, documented
choices at each one. This note records them so the result table in the
README is interpretable.

## The Pi 5 + C270 compute envelope

- **CPU**: quad-core ARM Cortex-A76 @ 2.4 GHz. No GPU, no NPU.
- **Inference runtime**: TensorFlow Lite (`ai_edge_litert`) with the
  XNNPACK delegate. FP32 single-thread fallback in practice.
- **Per-frame compute budget for "real-time"**: ≈ 50 ms. With
  MediaPipe HandLandmarker eating ~12 ms and a Conv2D classifier
  another 18-25 ms, we have very little slack.
- **Camera**: Logitech C270 USB UVC, 640 × 480 / 1280 × 720, 30 fps max.
- **Memory**: 8 GB Pi 5; not the binding constraint.

## Trade #1 — FP32 vs INT8 letter inference

The natural attempt: post-training INT8 quantization of MobileNetV3-Small,
calibrated on a 100-image sample. Should fit ~3× more arithmetic into
the same time budget.

**Outcome.** Catastrophic accuracy collapse:

| Model | Honest test accuracy |
|---|---|
| FP32 MobileNetV3-Small | 0.802 |
| INT8 (Kaggle calibrated) | **0.218** |

Top-confusion analysis showed many classes mapping to a small set of
"attractor" labels (`F`, `H`, `V`, `space`) — the classic sign of
calibration distribution mismatch between the activation statistics
seen at calibration time and those at deployment time.

**Decision.** Ship FP32. Document the INT8 number as a "do not deploy"
data point. Re-running the calibration with C270-captured images
post-`HandCropper` is plausible future work; we did not have time to
do it before the demo, and our latency budget under FP32 was already
acceptable.

**Consequence.** The letter pipeline is heavier than it would be at
INT8, which forces the next two trade-offs.

## Trade #2 — Camera resolution at capture

The C270 supports 1280 × 720 at 30 fps natively. But each high-res
frame costs:

- More USB bandwidth and decode work on the Pi.
- A larger `cv2.cvtColor` and resize before MediaPipe.
- A longer MediaPipe HandLandmarker pass (proportional to source
  resolution because of the palm detection grid).

Empirical result: at 1280 × 720 we exceeded the 50 ms / frame budget
on hand-present frames after MobileNet inference, with thermal
throttling under sustained use.

**Decision.** Capture at **640 × 480**. After the MediaPipe
`HandCropper` produces a 224 × 224 hand-centered crop, the
classifier doesn't see the original resolution anyway, so the
sacrifice is bounded — what we lose is the marginal MediaPipe
landmark precision on small or distant hands.

**Consequence on the word extension.** Lower MediaPipe landmark
precision propagates into the 21-keypoint sequence the word model
receives. We did not measure the resolution-on-word-accuracy curve
because the extension was already operating below the published
pose-only state-of-the-art for WLASL-100 (60-70 % top-1) and a
small marginal lift wasn't worth the latency hit on the main
letter pipeline.

## Trade #3 — Inference cadence (`inference_interval_ms`)

Even at 640 × 480 + FP32, sustained 30 fps inference exceeds the
sustained CPU budget; 100 % CPU drives temperature up and triggers
clock throttling, which ironically slows inference further.

**Decision.** Default to `inference_interval_ms = 300` (≈ 3 fps
heavy-pipeline inference). The camera grabber thread runs at a
higher rate; the **letter classifier** consumes whatever the latest
frame is, every 300 ms.

**Override for word capture.** When the user presses "Capture Word",
`_word_capturing` is set to true and the camera loop bypasses the
300 ms throttle so the 30-frame buffer fills at the pipeline's
natural rate (~22 fps on Pi). After the buffer fills (~1.3 s), the
classifier runs and the throttle returns. This is the only place
the system temporarily approaches its compute ceiling — and the
total burst is bounded to ~1.5 s.

## Trade #4 — Pose-only word model (extension)

The natural alternative for higher word accuracy is the MediaPipe
Holistic stack (543 keypoints: face + body + hands), or graph-based
models like ST-GCN that explicitly model hand-shoulder-body
relations.

**Holistic latency on Pi 5.** Adds an estimated 40-120 ms per frame
on top of Hands. Doesn't fit our budget while letter inference is
also running.

**ST-GCN.** We attempted training this with two configurations on
A100 (a large variant overfit, train 0.99 / test 0.25; a small
variant under-fit, train and test both 0.19) and abandoned in
favor of the BigConv1D ensemble that hit 0.674 / 0.921. ST-GCN
remains plausible future work with more training time and
hyperparameter search; it doesn't fit the hours we had.

**Decision.** Stay on the existing 21-landmark Hand-only stream.
Accept the cross-signer accuracy floor.

## Where each trade shows up in the deployed numbers

| Number in README | Trade-off responsible |
|---|---|
| Letter test accuracy 0.829 (vs hypothetical INT8 0.218) | Trade #1 |
| Letter inference 37.6 ms / frame on Pi | Trade #1 + #2 |
| Letter capture cadence 3 fps | Trade #3 |
| Word capture latency 1.3 s | Trade #3 (override) |
| Word ensemble inference 17 ms / clip | (no trade — model is small) |
| Word test top-1 0.674 (not pushing higher) | Trade #2 + #4 |

## Summary

Three constraints set the project's working envelope: the absence of
an accelerator, the C270's bandwidth ceiling, and our refusal to ship
an INT8 model that fails on real frames. Within that envelope we
optimize the letter pipeline aggressively (Trade #1, #2, #3) and
treat the word pipeline as an extension that lives within the same
budget rather than as a competitor for it (Trade #4). The trade-offs
are explicit; the reader should evaluate the result table accordingly.

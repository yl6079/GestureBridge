# GestureBridge Specification

## 1. Scope

GestureBridge provides two modes on an edge platform:

1. `Translate Mode`
   - Sign-to-speech: camera hand gestures -> spoken output.
   - Speech-to-sign: spoken input -> text + reference sign label/image key.
2. `Learn Mode`
   - Teaching: show target sign and evaluate learner attempt.
   - Practice: show meaning only and evaluate learner attempt.

## 2. Target Metrics

- Gesture vocabulary size: 20 signs.
- Gesture classification accuracy: >= 85% on held-out test split.
- End-to-end sign-to-speech latency: a few seconds or less on Raspberry Pi.
- Learn mode feedback: interactive frame updates and immediate pass/fail signal.
- Memory budget: runtime comfortably within 4GB RAM.

## 3. Hardware and Device Roles

- XIAO ESP32S3: always-on low-power activity detector.
- Raspberry Pi: on-demand compute node for computer vision, inference, ASR/TTS, and UI.

## 4. Control and State Machine

### System States

- `IDLE_LOW_POWER`: XIAO active, Raspberry Pi sleeping.
- `WAKE_REQUESTED`: activity detected by XIAO.
- `ACTIVE`: Raspberry Pi awake; selected mode runs.
- `COOLDOWN`: no activity timeout reached; prepare sleep transition.

### Mode States

- `MODE_SELECT`
- `TRANSLATE_SIGN_TO_SPEECH`
- `TRANSLATE_SPEECH_TO_SIGN`
- `LEARN_TEACHING`
- `LEARN_PRACTICE`

### Error and Recovery

- `ASR_FAILURE`: ask user to retry speech input.
- `CAMERA_FAILURE`: mode pauses and prompts reconnection.
- `LOW_CONFIDENCE`: no output action, request clearer input.

## 5. Protocol and Interfaces

### XIAO -> Raspberry Pi Wake Signal

- Signal includes:
  - `timestamp_ms`
  - `activity_level` in `[0, 1]`
  - `event_type` (`hand_activity`, future extensible)

### Raspberry Pi Response

- `wake_ack`: wake accepted.
- `busy_reject`: already active and locked.

## 6. Evaluation Matrix

- Functional: all mode transitions and error handling.
- Performance: latency, FPS, CPU/RAM footprint.
- Reliability: long-run stability and repeated wake cycles.
- Power: idle vs wake-on-demand comparison.

# GestureBridge demo video — 3-min script

**Hard cap: 3:00.** Pace is tight; aim for 2:50 cut, 10s buffer for fades.

Recording setup: screen-record the Pi's HDMI output (kiosk browser at
`http://127.0.0.1:8080`) **plus** a phone camera over the shoulder
showing the hand in front of the C270. Picture-in-picture or split
screen in editing.

---

## Beat sheet

| t | Visual | Voiceover |
|---|---|---|
| 0:00 – 0:15 | Logo + "GestureBridge" title card; Pi hardware photo from `notes/pi_validation/` | *"GestureBridge is a real-time American Sign Language interpreter that runs entirely on a Raspberry Pi 5. Camera in, speech out."* |
| 0:15 – 0:35 | Wide shot: Pi + LCD + C270 + ESP32 + speaker on the desk. Walk on the PIR sensor → idle screen → web app launches | *"An ESP32 with a motion sensor wakes the system when you walk up. The Pi streams from the C270 webcam, runs a hand detector and a small classifier, and speaks each letter through the USB speaker."* |
| 0:35 – 1:05 | **Read mode** screen-cap. Sign A, B, C, L, Y in front of camera. Each letter appears on screen; TTS audio plays. Show the confidence number (this is the **ensemble** confidence, not raw MobileNet) | *"In read mode, MediaPipe finds the hand, crops it, and a fine-tuned MobileNetV3 predicts the letter. We added a landmark-MLP ensemble — when the two heads disagree we trust the geometric one, because ASL is mostly hand shape. The number you see on screen is the ensemble's confidence."* |
| 1:05 – 1:25 | **Speech-to-sign** mode. User speaks "L" → reference image of the L sign appears | *"Speech-to-sign uses the browser's speech recognition to look up reference images — useful for someone learning the alphabet."* |
| 1:25 – 1:50 | **Trainer** mode. Random target letter; user signs it; system shows "✓" or "✗" and the speaker says **"true"** or **"false"** | *"Trainer mode picks a target letter and speaks 'true' or 'false' while you practice."* |
| 1:50 – 2:25 | Cut to **the credibility story** — slide showing the 100% → 80% → 82.9% number plus three-bug list (or just the bullets over a blackboard background) | *"The most interesting part of this project wasn't the model — it was finding three bugs that hid a 20-point accuracy gap. The dataset's frames were leaking across train and test, the augmentation was flipping ASL signs left-right which breaks chirality, and we never cropped the hand at inference. Once we fixed those, honest accuracy on a leakage-free split is 82.9 %."* |
| 2:25 – 2:50 | Pi-side latency screenshot or a stopwatch overlay: "37.6 ms / frame on Pi 5" | *"On the Pi, inference takes 37.6 milliseconds per frame, well within real-time. We ship FP32 — INT8 quantized cleanly but the Kaggle calibration didn't survive the C270 distribution, which is something for next iteration."* |
| 2:50 – 3:00 | End card: "Code: github.com/yl6079/GestureBridge — branch `shufeng`" + names | *(silence or short outro)* |

---

## Shot list (so we can record once and edit)

1. Hardware wide shot (10 s, static).
2. PIR walk-up trigger (8 s, one take).
3. Read mode — five letters in sequence: A, B, C, L, Y. ~5 s each.
4. Speech-to-sign mode — speak "L" then "B". 15 s total.
5. Trainer mode — three rounds. 20 s total.
6. Hand close-up (overhead) for B-roll over the credibility narration.

## Editing notes

- Mute the Pi's built-in beeps; voice-over recorded separately and laid
  on top.
- Burn the predicted letter + confidence number into the screen-capture
  if the kiosk UI doesn't already show it large enough.
- The credibility section needs **a single static frame** behind the
  voice-over — don't try to film it. Use slide 4 from `slides.md`
  exported as PNG.

## Q&A prep (likely questions)

- **"Why FP32 not INT8?"** — INT8 calibrated on Kaggle dropped to 21.8 %.
  Kaggle and C270 distributions differ; we need on-device calibration
  data to fix it. FP32 fits and runs at 37 ms, so we shipped it.
- **"How did you find the leakage?"** — The reported 100 % was suspicious
  given Yizheng's experience that real C270 confidence was 0.08. We
  re-ran the same model on a frame-contiguous split and *still* got
  100 % — every test frame had been seen in training. That confirmed
  the per-image stratified split was the bug.
- **"Why a separate landmark MLP, not just MobileNet?"** — ASL is
  geometric; for shapes like B vs D vs H, the landmarks alone tell
  you the answer. The ensemble adds 2.7 pp absolute on the test split.
- **"Will it work for other signers?"** — Single-signer Kaggle is the
  next obvious threat to validity. We didn't measure cross-signer
  generalization; that's the headline limitation.
- **"What does the ESP32 do?"** — PIR-based wake/sleep so the Pi isn't
  spinning the camera 24/7. Not the inference path.

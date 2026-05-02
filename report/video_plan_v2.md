# GestureBridge — 3-min demo video, v2

> **Source documents fused here:**
> - Yizheng's `notes/pre.md` (presentation outline, May 1)
> - Our existing `report/demo_video_script.md` (beat sheet)
> - Course PDF `SP26_Embedded_AI_Final_Project_Guidelines.pdf` (page 3 video spec)
>
> **Course requirement (verbatim):** the video must (a) demonstrate the system,
> (b) highlight key challenges, (c) explain important design decisions, in **≤ 3:00**.
> Q&A is separate (in-class). So this video is **presentation + demo combined**.

## Format decision

- **Voice-over + edited footage**, not live narration.  Quality control is much
  easier and we can hit the 3:00 cap exactly.
- Two visual layers, intercut:
  - **A.** Pi screen capture (kiosk web UI at `http://127.0.0.1:8080`).
  - **B.** Side-on phone shot of the hand in front of the C270 + the screen.
- Two static slide frames as B-roll for the "design decisions" segment (slide 4 + slide 11 from `report/slides.md`, exported as PNGs).

## Beat sheet (target 2:55 + 5 s buffer)

| t | Layer | Visual | Voice-over |
|---|---|---|---|
| 0:00 – 0:15 | Slide | Title card "GestureBridge — Real-time ASL on a Pi 5" + names; cut to hardware photo `notes/pi_validation/c270_empty_2026-04-30.jpg` | *"GestureBridge is a fully offline ASL interpreter on a Raspberry Pi 5. Camera in, speech out — no cloud, no internet, privacy-friendly. The whole system runs on edge hardware that costs under \$120."* |
| 0:15 – 0:35 | B + A | Wide shot of hardware (Pi + LCD + C270 + ESP32 + speaker). Walk past PIR → idle screen → app auto-launches | *"An ESP32 with a motion sensor wakes the system on approach so the Pi isn't running the camera 24/7. The Pi handles inference, speech, and the web UI."* |
| 0:35 – 1:05 | A + B PiP | **Read mode.** Sign A, B, C, L, Y; on-screen ensemble label + confidence each time, TTS audible | *"Read mode finds the hand with MediaPipe, crops it, and runs a fine-tuned MobileNetV3. A second model on hand landmarks votes in parallel — when the two disagree, we trust the geometric one because ASL is mostly hand shape. The number on screen is the ensemble's confidence."* |
| 1:05 – 1:25 | A + B PiP | **Speech-to-sign.** Press the record button, speak "L", release. Reference image of L appears | *"Speech-to-sign uses offline Vosk speech recognition on the C270's microphone — no internet, no cloud round-trip. Useful for someone learning the alphabet."* |
| 1:25 – 1:50 | A + B PiP | **Trainer mode.** Random target letter; sign it; speaker announces "true" | *"Trainer mode picks a target letter and just speaks 'true' or 'false' as you practice. Three rounds in 20 seconds."* |
| 1:50 – 2:25 | Slide | **Slide 4 (3-bug story).** Static, voice-over only | *"The most interesting part of this project wasn't the model — it was finding three bugs that hid a 20-point accuracy gap. The dataset's frames were leaking across train and test, the augmentation was flipping ASL signs which breaks chirality, and at deployment we never cropped the hand. Once we fixed those, honest accuracy on a leakage-free split is **82.9 percent**."* |
| 2:25 – 2:50 | Slide | **Slide 11 (results table).** Static; overlay "37.6 ms / frame on Pi 5" | *"On the Pi, one inference takes 37.6 milliseconds, well within real-time. We ship FP32 — INT8 quantized cleanly but the Kaggle calibration didn't survive the C270 distribution, so that's the next iteration."* |
| 2:50 – 3:00 | Slide | End card: "github.com/yl6079/GestureBridge — branch shufeng" + names | *(short outro music or silence)* |

## How this maps to the course PDF requirements

- **Demonstrate the system** → 0:35–1:50 (three modes, on real hardware).
- **Highlight key challenges** → 1:50–2:25 (the 3-bug story = our headline challenge).
- **Explain important design decisions** → throughout: ESP32 wake/sleep (0:15–0:35),
  ensemble rule (0:35–1:05), offline Vosk over cloud STT (1:05–1:25), FP32 over INT8 (2:25–2:50).
- **≤ 3 minutes** → ends at 2:50 with 10 s of slack.

## How this maps to Yizheng's `notes/pre.md` outline

| `pre.md` section | Where it appears in the video |
|---|---|
| 一、整体目标（offline / privacy / low-cost）| 0:00–0:15 voice-over |
| 二、系统架构（ESP32 + Pi 双设备协同）| 0:15–0:35 |
| 三 · 模式 1（Sign-to-Speech）| 0:35–1:05 |
| 三 · 模式 2（Speech-to-Sign）| 1:05–1:25 |
| 三 · 模式 3（Learning）| 1:25–1:50 |
| 四 · 1 双设备协同（state machine）| 0:15–0:35 voice-over |
| 四 · 2 实时性 vs 准确率（FP32 vs INT8）| 2:25–2:50 voice-over |
| 四 · 3 单模型鲁棒性（landmark ensemble）| 0:35–1:05 voice-over |
| 四 · 4 云端 vs 本地（Vosk）| 1:05–1:25 voice-over |
| 五 · 最终成果 | 2:50–3:00 end card |

Every item in Yizheng's outline is covered.  Our credibility-story segment
(1:50–2:25) is **additive** — it's the strongest "key challenge" content we
have and it's not in his outline yet, but it directly satisfies the course
PDF's "highlight key challenges" requirement.

## Shot list (record once, edit once)

1. **Hardware wide shot** — 10 s static (B-roll for 0:15–0:35).
2. **PIR walk-up** — one take, 8 s.
3. **Read mode** — five letters in sequence: A, B, C, L, Y. ~5 s each.
4. **Speech-to-sign** — record-button workflow: press, say "L", release. Then again with "B". 20 s total.
5. **Trainer mode** — three rounds.  20 s total.
6. **Hand close-up** — overhead, no UI; B-roll for the credibility narration if needed.

## Editing notes

- **Mute the Pi's built-in beeps**; voice-over recorded separately.
- **Burn the predicted letter + confidence number** into the screen-capture if
  the kiosk UI text is too small at 1080p output.
- **Static slide frames** for 1:50–2:25 and 2:25–2:50: export slides 4 and 11
  from `report/slides.md` (or use the rendered `report/GestureBridge_slides.pdf`
  pages 4 and 11 directly).
- Keep transitions minimal (cut, not dissolve) — fancy transitions burn time.

## Q&A prep (in-class, after the video — same as v1)

- **Why FP32 not INT8?** Calibrated INT8 on Kaggle dropped to 21.8 %; Kaggle
  and C270 distributions differ; we'd need on-device calibration. FP32 fits
  and runs at 37 ms, so we shipped it.
- **How did you find the leakage?** The reported 100 % was suspicious given
  Yizheng's experience that real C270 confidence was 0.08. We re-ran the same
  model on a frame-contiguous split and *still* got 100 % — every test frame
  had been seen in training. That confirmed the per-image stratified split
  was the bug.
- **Why a separate landmark MLP, not just MobileNet?** ASL is geometric; for
  shapes like B vs D vs H, the landmarks alone tell you the answer. The
  ensemble adds 2.7 pp absolute on the test split.
- **Will it work for other signers?** Single-signer Kaggle is the next obvious
  threat to validity. We didn't measure cross-signer generalization; that's
  the headline limitation.
- **Why offline Vosk instead of browser STT?** The Pi is offline at the demo;
  the browser Web Speech API needs the internet. Vosk runs locally on the
  C270's microphone, so the system stays end-to-end offline.
- **What does the ESP32 do?** PIR-based wake/sleep so the Pi isn't spinning
  the camera 24/7. Not on the inference path.

## Open items for Yizheng before recording

- Confirm slide 4 + slide 11 (PNG exports from `report/slides.md`) match what
  he wants visually for the credibility-story segment, OR substitute frames
  from his Gamma deck if he prefers.
- Confirm the record-button label / wording in the Speech-to-sign UI so the
  voice-over matches what's on screen.
- Decide who voices the narration — single voice is cleaner; cap total
  recording time at one afternoon.

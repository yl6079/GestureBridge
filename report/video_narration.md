# GestureBridge — 3-min video voice-over (English)

> Word-for-word narration script aligned to the beat sheet in
> [video_plan_v2.md](video_plan_v2.md).
>
> **Target delivery: 2:50** at ~145 words/min. Total budget: ~411 words.
> Recorded **separately** from footage in a quiet room, then dropped on
> top in editing. **Duck the kiosk's TTS audio** under each voice-over
> line.
>
> Pacing tips:
> - Read at conversational pace, not radio-DJ. Pause briefly between
>   sentences (a comma's worth).
> - **Bolded words** are the ones to lean on slightly.
> - Take a breath at every `[breath]` marker — easier to edit than a
>   gasp mid-sentence.

---

## 0:00–0:15 — Title + opener  *(~38 words / 15 s)*

> **GestureBridge** is a real-time American Sign Language interpreter
> that runs **entirely on a Raspberry Pi 5**. [breath] Camera in,
> speech out — **no cloud, no internet**. The whole system costs
> under one hundred and twenty dollars.

*Visual: title card → hardware photo cross-fade.*

---

## 0:15–0:35 — Hardware + ESP32  *(~47 words / 20 s)*

> An **ESP32** with a motion sensor wakes the Pi when you walk up,
> so the camera isn't running twenty-four-seven. [breath] The Pi
> itself handles inference, speech recognition, text-to-speech, and a
> kiosk web interface — **three modes** that all share the same
> hand-recognition pipeline.

*Visual: hardware wide → PIR walk-up → kiosk launches.*

---

## 0:35–1:05 — Read mode + ensemble  *(~71 words / 30 s)*

> In **Read mode**, MediaPipe finds the hand, crops it, and a
> fine-tuned MobileNetV3 predicts the letter. [breath] We added a
> second model — a small MLP on the **hand landmarks** — and we
> ensemble the two. When they disagree, we trust the geometric model,
> because A-S-L is mostly hand shape. The number on screen is the
> **ensemble's confidence**. Watch the predictions track A, B, C, L,
> Y in real time.

*Visual: kiosk Read mode + side-on hand. TTS audio ducked.*

---

## 1:05–1:25 — Speech-to-Sign + Vosk  *(~47 words / 20 s)*

> **Speech-to-Sign** uses fully **offline Vosk** speech recognition on
> the camera's built-in microphone. [breath] No cloud round-trip, no
> internet — which matters because the Pi has **no network at the
> demo**. Press record, say a letter, and the matching reference
> image appears.

*Visual: press the record button, speak "L", reference image of L.*

---

## 1:25–1:50 — Trainer mode  *(~51 words / 25 s)*

> **Trainer mode** flips the script. The system picks a random target
> letter and **speaks 'true' or 'false'** as you practice. [breath]
> Useful for a beginner who doesn't yet know if they're forming the
> shape correctly — three rounds in twenty seconds, with audio
> feedback so you don't even have to look at the screen.

*Visual: 3 trainer rounds. Pi speaker audio ducked under voice-over.*

---

## 1:50–2:25 — Three-bug credibility story  *(~82 words / 35 s)*

> The most interesting part of this project wasn't actually the model.
> It was finding **three latent bugs** that hid a **twenty-point
> accuracy gap**. [breath] The dataset's frames were leaking across
> train and test, the augmentation was flipping signs left-to-right
> which breaks A-S-L chirality, and at deployment we **never cropped
> the hand at all**. [breath] The original code reported one hundred
> percent test accuracy. Once we fixed those three bugs and ran on a
> leakage-free split, honest accuracy is **eighty-two point nine
> percent**.

*Visual: STATIC — slide 4 (three-bugs slide PNG) full-frame.*

---

## 2:25–2:50 — Pi latency + INT8 honest note  *(~63 words / 25 s)*

> On the Raspberry Pi itself, one inference takes **thirty-seven point
> six milliseconds** end-to-end, well under our real-time budget.
> [breath] We ship FP32. We tried INT8 — it quantized cleanly, but
> the Kaggle calibration didn't survive the C-two-seventy camera
> distribution and accuracy collapsed to **twenty-two percent**. The
> honest fix is on-device calibration data, which is the next
> iteration.

*Visual: STATIC — slide 11 (results table PNG); overlay "37.6 ms /
frame on Pi 5" lower-third.*

---

## 2:50–3:00 — End card  *(~12 words / 10 s)*

> Code is on GitHub at **y-l-six-zero-seven-nine slash GestureBridge**.
> Thanks for watching.

*Visual: end card with GitHub URL + names. Outro silence or a single
short note.*

---

## Total time check

| Segment | Span | Words | Spoken time @ 145 wpm |
|---|---|---|---|
| Title + opener | 0:00–0:15 | 38 | 15.7 s |
| Hardware + ESP32 | 0:15–0:35 | 47 | 19.4 s |
| Read mode + ensemble | 0:35–1:05 | 71 | 29.4 s |
| Speech-to-Sign + Vosk | 1:05–1:25 | 47 | 19.4 s |
| Trainer mode | 1:25–1:50 | 51 | 21.1 s |
| Three-bug story | 1:50–2:25 | 82 | 33.9 s |
| Pi latency + INT8 | 2:25–2:50 | 63 | 26.1 s |
| End card | 2:50–3:00 | 12 | 5.0 s |
| **Total** | | **411** | **2:50.0** |

5-second buffer to 3:00 hard cap. Read aloud once with a stopwatch
before recording — if you land >2:55, drop the trainer-mode "with
audio feedback so you don't even have to look at the screen" clause
to claw back ~5 s.

## Recording-day notes

- **Mic**: a phone earbud mic in a quiet bedroom is fine. Avoid the
  Pi room (HDMI fan + speaker buzz).
- **Levels**: peak ~−6 dB, no clipping. Re-record any segment that
  hits 0 dB.
- **Take 3 of each segment minimum** — costs nothing, gives editor a
  choice.
- Save raw recordings as `recording/voice/segment_NN_takeM.wav`. The
  editor picks the best take per segment and slots them into the
  timeline.

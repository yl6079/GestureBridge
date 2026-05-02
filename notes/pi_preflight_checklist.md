# Pi pre-flight checklist — before recording the demo video

> Run through this **before any camera rolls.**  Every item should
> tick green.  If anything fails, fix it first — don't film around it.
>
> Estimated time: 10 minutes.  Companion documents:
> [`report/video_plan_v2.md`](../report/video_plan_v2.md) (beat sheet)
> and [`report/video_narration.md`](../report/video_narration.md)
> (voice-over).

---

## A. Software is current

- [ ] On the Pi: `cd ~/gesture-bridge && git pull origin shufeng`
  succeeds and the working tree is clean.
- [ ] `git log --oneline -3` shows commits authored *today* (so we
  know our latest narration script and PNGs are present, even though
  recording doesn't depend on them).

## B. Vosk speech-to-sign is ready

- [ ] `ls models/vosk-model-small-en-us-0.15` (or wherever
  `scripts/fetch_vosk_small.sh` puts it) shows the unpacked Vosk
  model directory.  If missing, run `bash scripts/fetch_vosk_small.sh`.
- [ ] Vosk model size is non-zero (small model is ~40 MB).

## C. App boots cleanly

- [ ] `bash start_gesturebridge.sh` brings up the kiosk web UI on
  `http://127.0.0.1:8080` with no Python tracebacks in the launcher
  terminal.
- [ ] All three mode tabs (Read / Speech-to-Sign / Trainer) load
  without 500 errors in the browser console.
- [ ] Camera preview shows live C270 video, not a frozen frame or a
  black square.

## D. Read mode hits confidence under recording lighting

Sign each letter clearly, hold for ~2 s, watch the on-screen
**ensemble** confidence number:

- [ ] **A** — confidence ≥ 0.4 stably
- [ ] **B** — confidence ≥ 0.4 stably
- [ ] **C** — confidence ≥ 0.4 stably
- [ ] **L** — confidence ≥ 0.4 stably
- [ ] **Y** — confidence ≥ 0.4 stably

If any letter sits below 0.4, **adjust the lighting before
recording** (overhead lamp, kill backlight from window).  Don't
lower the threshold.

## E. Speech-to-Sign actually resolves words

- [ ] Press the record button → say **"L"** clearly → release →
  reference image of L appears within 2 seconds.
- [ ] Repeat with **"B"** — reference image of B appears.
- [ ] No internet check: temporarily turn off the Pi's Wi-Fi and
  retry one of the above.  Vosk should still resolve.

## F. Trainer mode rounds work

- [ ] Click "Trainer" → a target letter appears.
- [ ] Sign the target → speaker says **"true"**, screen shows ✓.
- [ ] Deliberately sign the wrong letter → speaker says **"false"**,
  screen shows ✗.
- [ ] Run three rounds in a row without the app hanging.

## G. ESP32 wake / sleep behavior

- [ ] Walk away from the PIR for 15 s → app idles to standby.
- [ ] Walk back in front of the PIR → app re-launches within 2 s.
- [ ] Onboard ESP32 LED behaves as expected (per
  `esp32_camera.ino` debounce logic).

## H. Audio levels for the camera mic

- [ ] Phone (or whatever camera you're using) records the Pi
  speaker's TTS at a level you can hear in the playback, but **not
  clipping**.  If the speaker is too loud, turn down the Pi's
  output volume rather than moving the camera.
- [ ] No HDMI fan whine in the recording — if there is, point the
  camera mic away from the Pi.

## I. Recording-room logistics

- [ ] Background is plain (a wall, a curtain, or the lab desk).
  Avoid moving people, monitors playing video, etc.
- [ ] Pi LCD is at a clean angle for the side-on phone shot — both
  the kiosk UI and the hand should be visible in the same frame.
- [ ] Phone has at least 30 % battery and 4 GB free storage.
- [ ] Tripod or a book stack to keep the phone steady — avoid
  hand-held.

---

## If something fails

| Symptom | Most likely cause | Fix |
|---|---|---|
| Confidence stuck at 0.08 on every letter | Old (leaky) model is loaded | `ls -la artifacts/asl29/tflite/model_fp32.tflite` — must be the new one (~3.7 MB) and dated after Apr 30. Re-run `bash scripts/deploy_to_pi.sh` from Mac if not. |
| `nothing` predicted no matter what you sign | MediaPipe didn't detect a hand | Check lighting, move closer to camera. The kiosk should also show a small "no hand" indicator. |
| Vosk crashes on first record | Model dir not unpacked | `tar xzf` or re-run `fetch_vosk_small.sh`. |
| Web UI 404s on a sign asset | Asset rename mismatch | `ls assets/signs/A.jpg` should exist; `A1.jpg` should not. If `A1.jpg` exists, you're on a stale branch — `git pull origin shufeng`. |
| TTS doesn't speak | USB speaker not the default | Run `bash scripts/set_default_mic_c270.sh` (or its TTS equivalent) and re-launch. |

---

## Once everything ticks green

→ Start with the **hardware wide shot** (10 s static), then the **PIR
walk-up** (one take), then the three modes per
[`report/video_plan_v2.md`](../report/video_plan_v2.md).  Voice-over
recorded **separately** in a quiet room afterwards.

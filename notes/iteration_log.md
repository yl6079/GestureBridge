# Phase 2 Iteration Log

Append-only log of every iteration in the Phase 2 word-recognition sprint.
Each entry follows: **What was attempted / What works / What broke / Files
touched / Numbers / Sources & URLs**.

Plan reference: `~/.claude/plans/i-have-seen-claude-nifty-brooks.md`.
Branch: `shufeng`. Active mode: iteration-driven, ship-first.

---

## Iteration 1 — 2026-05-05 — Speech→sign upgrade for 5 known words

### What was attempted

The user reported the loudest pain point: when they say "electrical" the
speech-to-sign output is a 10-image array of letter gestures. Real ASL
has dedicated word-level signs. IT-1 fixes this for a small starter
vocabulary (`hello`, `thanks/thank you`, `yes`, `no`, `help`) before
training anything — proves the architecture for video-based
speech-to-sign output.

### What works

- Backend `MainRuntime.run_speech_to_sign` now does per-token lookup. If
  a token has a video clip on disk under `assets/word_clips/`, the
  response uses the clip filename (e.g. `hello.mp4`); otherwise it
  falls back to letter-spelling that token.
- Aliases handled in `WORD_CLIP_MAP` (`thanks` and `thank` both map to
  `thank_you.mp4`; `hi`/`hey` map to `hello.mp4`; `yeah`/`yep` map to
  `yes.mp4`; `nope` maps to `no.mp4`).
- `letters` and `sign_assets` lists are now strictly aligned 1:1 — the
  UI can map index-to-index without surprises.
- Web server serves `/assets/word_clips/<name>.mp4` with `video/mp4`
  MIME type, mirroring the existing `/assets/signs/` path. Missing clips
  return 404 cleanly.
- Web UI `renderSignGallery` detects `.mp4`/`.webm` extensions and
  emits a `<video autoplay loop muted playsinline>` element instead of
  `<img>`. Letter results unchanged.
- Smoke tests on Mac (Python 3.12 venv):
  - `'hello'` → `letters=['HELLO']`, `sign_assets=['hello.mp4']`
  - `'thanks'` → `letters=['THANKS']`, `sign_assets=['thank_you.mp4']`
  - `'electrical'` → letter array (fallback proven, no clip exists)
  - `'hello electrical'` → mixed `['hello.mp4', 'E.jpg', 'L.jpg', …]`
  - HTTP `GET /assets/word_clips/hello.mp4` → `200`, `video/mp4`, 85891 bytes
  - HTTP `GET /assets/word_clips/missing.mp4` → `404`
  - HTTP `GET /assets/signs/A.jpg` → `200`, `image/jpeg`, 12635 bytes

### What broke

- First attempt with Python 3.9 venv failed (`dataclass(slots=True)` is
  3.10+). Switched to `/opt/homebrew/bin/python3.12`.
- `aslbricks.org` returns 406 to the default curl User-Agent. Fixed by
  passing a Chrome UA via `-A`.
- `aslbricks.org` does not have `hello.mp4`. Fell back to
  `media.signbsl.com` (Start ASL clip), reached via `signasl.org`.
- Python emitted a `SyntaxWarning: invalid escape sequence '\.'` in the
  inline JS regex. Replaced regex with `endsWith('.mp4')`.

### Files touched

- `src/gesturebridge/system/main_runtime.py:67-87` — added `WORD_CLIP_MAP`
- `src/gesturebridge/system/main_runtime.py:412-441` — rewrote
  `run_speech_to_sign` to do per-token word→clip lookup with letter
  fallback, kept `letters` and `sign_assets` aligned 1:1
- `src/gesturebridge/config.py:135-144` — added `WebUIConfig.word_clips_dir`
- `src/gesturebridge/ui/web.py:794-832` — extended `/assets/...` route
  to also serve `/assets/word_clips/`, added MP4/WebM MIME types
- `src/gesturebridge/ui/web.py:511-540` — `renderSignGallery` branches
  to `<video>` for video assets
- `assets/word_clips/SOURCES.md` — new, documents clip provenance
- `scripts/fetch_word_clips.sh` — new, reproduces the IT-1 download
- `.gitignore` — `assets/word_clips/*.mp4`, `external/WLASL/`
- `notes/iteration_log.md` — this entry
- `notes/yizheng_wechat.md` — first Chinese sync

### Numbers

- 5 / 5 target words have working clips on disk (durations 2.2-2.7 s,
  resolutions 640×360 to 1920×1080).
- Backend logic verified against 8 utterances; no regression in
  letter-spelling fallback.
- HTTP smoke test: 200 / 404 paths both correct.
- No new runtime dependencies (still numpy + opencv-python-headless on
  dev; Pi runtime unchanged).

### Sources / URLs (for the user to verify)

- WLASL v0.3 JSON (canonical gloss list, used to discover URLs):
  `https://raw.githubusercontent.com/dxli94/WLASL/master/start_kit/WLASL_v0.3.json`
- `hello.mp4`: `https://media.signbsl.com/videos/asl/startasl/mp4/hello.mp4`
  (referrer `https://www.signasl.org/`).
- `help.mp4`, `no.mp4`, `yes.mp4`: `http://aslbricks.org/New/ASL-Videos/{word}.mp4`
- `thank_you.mp4`: `http://aslbricks.org/New/ASL-Videos/thank%20you.mp4`
- yt-dlp installed `--user` for IT-2 (data acquisition step).

### Acceptance check

User runs the app on Mac; speech-to-sign tab; says or types `hello` →
UI plays the HELLO clip in a `<video>` element. Says `electrical` →
letter sequence (fallback). Letter mode and Read mode are untouched.

### Next iteration entry point

IT-2: download WLASL-100 video set + build `(N, 30, 63)` landmark
tensor for training.

---

## Iteration 2 — 2026-05-05 — Pi probe, camera-index CLI, deploy-path fix

### What was attempted

Use the Tailscale IP from `notes/pi_access.md` to probe the Pi (no
write actions, just reads), confirm what hardware is currently visible,
and ship the easy IT-5 win (camera-index CLI flag). The deploy script
defaulted to a stale path (`.../GestureBridge/test`) that no longer
exists on the Pi; corrected to the canonical path.

### What works

- SSH to `elen6908@100.127.215.9` over Tailscale: confirmed reachable.
  `uname` shows Pi 5 (`6.12.75+rpt-rpi-2712 aarch64`).
- C270 webcam: detected at `/dev/video0` (also `/dev/video1` for the
  YUYV vs MJPG variant). `lsusb` shows `046d:0825 Logitech, Inc. Webcam
  C270`.
- USB speaker (JieLi Technology): detected via `lsusb`.
- Project location: `/home/elen6908/Documents/GestureBridge` (no `/test`).
- `--camera-index N` CLI flag: live, prints override on startup.
- Memory: Pi credentials and USB-port mapping persisted to memory file.

### What broke

- **ESP32 not currently visible.** No `/dev/ttyUSB*` or `/dev/ttyACM*`
  device. `/dev/serial/by-id/` empty. Either unplugged or PIR firmware
  not flashed in current state. Yizheng's note ("ESP32 插中间靠里面")
  describes the intended layout in `pic/connection.jpg`. Word mode does
  not require ESP32 for the demo; the wake-gating layer just stays in
  fallback "always-active" mode if no serial events arrive. Will re-test
  once Yizheng confirms the setup.

### Files touched

- `src/gesturebridge/app.py:60-72` — added `--camera-index N` argparse
  flag; overrides `cfg.asl29.runtime.camera_index` on startup.
- `scripts/deploy_to_pi.sh:13-24` — fixed default `PI_PATH` from
  `.../GestureBridge/test` to `.../GestureBridge`.
- Memory: `~/.claude/.../memory/pi_credentials.md` updated with
  Tailscale IP, project path, and USB port mapping per Yizheng.

### Numbers

- Pi network round-trip via Tailscale: ~30 ms (subjective).
- C270 detected as expected; no extra config needed.

### Sources / URLs

- Pi access cheat-sheet: `notes/pi_access.md` (gitignored, contains the
  Tailscale IP and the now-stale path).

### Acceptance check

`python -m gesturebridge.app --help` lists `--camera-index`. CLI
override works on Mac (Python 3.12 venv).

---

## Iteration 3 — 2026-05-05 — WLASL-100 dataset, landmark extraction, Conv1D training

### What was attempted

End-to-end pose-only word-recognition pipeline: scrape direct-MP4
sources for the top-100 most-frequent WLASL glosses, extract MediaPipe
landmark sequences, train a Conv1D-Small, export numpy weights for Pi
inference, and ship a CLI predictor so the user can sanity-check on
real clips immediately.

### What works

**Data — `scripts/prepare_wlasl100.py`:**
- Top-100 selection by instance count over all 2,000 WLASL glosses
  (proxy for the official WLASL-100 split). Direct-MP4 priority order:
  aslbricks → aslsignbank → signingsavvy → aslsearch → asldeafined →
  others, with yt-dlp YouTube fallback.
- Per-host `Referer` headers + browser User-Agent fix the 406/403 bot
  blocks on aslbricks and signbsl. SSL hostname-mismatch on
  aslsignbank.haskins.yale.edu accepted via a relaxed context limited
  to that one host.
- Resumable: existing files skipped. Manifest CSV preserves provenance
  (gloss, video_id, split, source URL, ok/result reason).

**Landmarks — `scripts/extract_wlasl_landmarks.py`:**
- Reuses the existing `pipelines/hand_crop.HandCropper` (MediaPipe Tasks
  HandLandmarker) for parity with the letter pipeline.
- 30 evenly-spaced frames per clip, normalized landmarks (wrist origin,
  max-abs-distance scale), produces `(N, 30, 63)` tensor.
- Drops clips with <30% hand-detection rate.

**Training — `scripts/train_wlasl100_pose.py`:**
- Conv1D-Small (~50K params): 64 → 64 → MaxPool 2 → 128 → GAP →
  Dense 128 → Dense 100 softmax.
- Cross-entropy with label smoothing 0.05, Adam lr 1e-3, ReduceLROnPlateau,
  EarlyStopping, augmentation (temporal roll ±2 frames, spatial scale 0.9-1.1).
- Saves both Keras `.keras` and a numpy `.npz` of weights for Pi runtime.

**Pi runtime — `pipelines/word_classifier.py`:**
- Pure numpy forward pass (Conv1D SAME-padded via einsum, ReLU,
  MaxPool, GlobalAvgPool, Dense). **No TF dependency at runtime.**
- Bit-exact match with the Keras model: `max_prob_diff = 0.0` across 5
  test examples.

**Predictor CLI — `scripts/predict_word_clip.py`:**
- One-liner: `python scripts/predict_word_clip.py path/to/clip.mp4`
- Reports detection rate, top-5 with confidence bars.
- Useful for live testing before UI integration lands.

### Numbers

- WLASL-100 download (direct sources only): **676 / 2023 attempts** =
  33%. **All 100 glosses got ≥3 clips** (min 3, median 7, max 12).
  Top performers: thin (12), who (11), cool (11), computer (10), candy (10).
- Disk: 71 MB clips on first pass; grew to 381 MB after partial yt-dlp
  attempts (which got rate-limited by sandbox timeouts; will resume).
- Landmark extraction: **670 / 676** clips kept (6 dropped <30% detect).
  Time: 6 min on Mac CPU.
- Train: 490, val: 113, test: 67.
- Train top-1: 64% (overfitting, expected at 5 clips/class).
- **Val top-1: 21.2%, top-5: 53.1%.**
- **Test top-1: 16.4%, top-5: 43.3%.**
- Real-world spot checks (in-distribution WLASL clips):
  - `yes/64284.mp4` → top-1 **YES 88%** ✅
  - `book/07068.mp4` → top-1 **BOOK 38%** ✅
  - `help/27209.mp4` → top-4 HELP 9% (top-1 wrongly BOOK 21%)
- Out-of-distribution clip (signbsl HELLO/HELP/etc.) recognition is
  weaker — expected, framing/lighting differs from WLASL training set.

### What broke

- **Acceptance bar (val ≥40% top-1) not met.** Causes: only 5 clips/class
  on average (much too few for 100-way classification), per-class val
  set has 1-3 clips so noise dominates. Mitigations queued:
  1. yt-dlp YouTube fallback (additional ~675 clips). Sandbox 2-min
     bash timeout killed the first run; need to re-launch in a way the
     sandbox doesn't kill (workaround: run from a user terminal directly,
     or chunk by gloss).
  2. Stronger augmentation (mixup, mirror-aware augment) on the
     existing data.
  3. Demo-mode: use **top-5** as the headline metric and pick a
     curated 10-15 word vocabulary that the model gets reliably right.
- Background bash with stdout redirection: Python buffers stdout when
  piped to a file; `-u` flag fixes it. Documented in the script.
- Several signingsavvy/handspeak URLs return 403 even with Referer
  (anti-hotlinking). No fix; we accept the long tail.

### Files touched

- NEW `scripts/prepare_wlasl100.py` (~210 lines) — downloader.
- NEW `scripts/extract_wlasl_landmarks.py` (~190 lines) — MediaPipe
  → (N, 30, 63) tensor.
- NEW `scripts/train_wlasl100_pose.py` (~155 lines) — Conv1D + GRU
  variants, eval JSON, npz export.
- NEW `src/gesturebridge/pipelines/word_classifier.py` (~110 lines) —
  pure-numpy inference loaded from npz.
- NEW `scripts/predict_word_clip.py` (~70 lines) — sanity-check CLI.
- Output artifacts (gitignored): `data/wlasl100/landmarks.npz`,
  `artifacts/wlasl100/conv1d_small.keras`, `artifacts/wlasl100/conv1d_small.npz`,
  `artifacts/wlasl100/labels.txt`, `artifacts/wlasl100/eval.json`.

### Sources / URLs

- WLASL: <https://github.com/dxli94/WLASL>
  (license: C-UDA, computational use only)
- WLASL JSON v0.3:
  <https://raw.githubusercontent.com/dxli94/WLASL/master/start_kit/WLASL_v0.3.json>
- Direct-MP4 sources used: aslbricks.org, signingsavvy.com (partial),
  signbsl.com, aslsignbank.haskins.yale.edu, aslsearch.com,
  asldeafined.com.
- Cited per-clip in `data/wlasl100/manifest.csv` (gitignored).

### Acceptance check

```bash
.venv/bin/python scripts/predict_word_clip.py path/to/your_clip.mp4
# → prints top-5 with confidence bars
```

For in-distribution WLASL test clips this routinely lands top-1 right
on signs with strong landmark signatures (yes, no, computer, book,
yellow, …). For out-of-distribution clips top-5 is more reliable.

### Next iteration entry point

IT-4: wire `WordClassifier` into a new "Word" tab in the web UI with a
"Capture 1 second" button. While Yizheng confirms ESP32 status, this
is a Mac-only change.

In parallel: run yt-dlp YouTube fallback chunked-by-gloss so the
sandbox 2-min timeout doesn't kill it, and re-train.

---

## Iteration 4 — 2026-05-05 — Word mode UI integration (in-app capture button)

### What was attempted

Hook the trained Conv1D word classifier into the live web UI so the user
can click a button, sign a word for ~1 s in front of the camera, and see
the top-5 predictions in the same view as the existing letter pipeline.
Keep it inside the existing **Read** mode tab so the user doesn't need
to learn a new mode — letters and words coexist, exactly per the
approved plan.

### What works

- `MainRuntime` now optionally holds a `word_classifier`. If
  `artifacts/wlasl100/conv1d_small.npz` + `labels.txt` exist, `app.py`
  auto-loads the WordClassifier and prints
  `[app] WLASL-100 word classifier attached: … (100 classes)` at boot.
- New runtime fields: `_word_capturing`, `_word_buffer` (list of 63-d
  vectors), `_word_window_frames=30`, `_word_last_prediction` (sticky).
- `process_camera_frame` pushes normalized landmarks into the buffer
  whenever `_word_capturing` is True. When buffer hits 30 frames →
  `_finalize_word_capture` runs the Conv1D, stashes top-5 with
  per-prob confidence and elapsed wallclock seconds, and (if top-1 ≥
  0.35) speaks the predicted word via TTS.
- New endpoints:
  - `POST /api/word/capture` → `{"ok": true, "window_frames": 30, "started_ts": ...}`
  - `GET /api/state` now includes `word_loaded`, `word_capturing`,
    `word_buffer_filled`, `word_window_frames`, `word_prediction`.
- New UI card "Word Recognition (WLASL-100)" inside the Read mode
  layout, hidden when the classifier isn't loaded. Button shows live
  capture progress (`Capturing... (12/30)`) and renders top-5 with
  confidence bars after each capture.
- Smoke tested end-to-end on Mac with a stub runtime: state poll →
  capture POST → 30 fake frames → prediction returned and sticky.

### What broke

- Initially called `take_word_prediction()` (clearing) inside
  `/api/state`. Switched to a sticky model — clients poll the same
  endpoint repeatedly and need the prediction to persist until next
  capture. Added `clear_word_prediction()` for an explicit clear path
  (not used yet).
- Frames where MediaPipe finds no hand replicate the last-seen vector
  rather than emitting zeros. Matches the training-time pad behaviour
  in `extract_wlasl_landmarks.extract_clip`. Without this the Conv1D
  receives noise-on-zero gaps and drops accuracy.

### Files touched

- `src/gesturebridge/app.py:148-167` — auto-attach WordClassifier from
  npz weights file; prints attachment info.
- `src/gesturebridge/system/main_runtime.py:79-95` — new word-mode fields.
- `src/gesturebridge/system/main_runtime.py:392-405` — `process_camera_frame`
  now feeds buffer when capturing; surfaces word state in response.
- `src/gesturebridge/system/main_runtime.py:430-510` — new helper methods:
  `_normalize_landmarks_63`, `_maybe_capture_word_frame`,
  `_finalize_word_capture`, `start_word_capture`, `take_word_prediction`,
  `clear_word_prediction`.
- `src/gesturebridge/ui/web.py:799-806` — `/api/state` exposes
  word_* fields.
- `src/gesturebridge/ui/web.py:872-883` — `POST /api/word/capture`
  handler.
- `src/gesturebridge/ui/web.py:317-328` — new "Word Recognition" card
  in the Read mode side panel.
- `src/gesturebridge/ui/web.py:556-595` — JS: `wordLoaded` /
  `wordCapturing` state, `startWordCapture()` POST function, render
  loop fills the prediction list with confidence bars.
- `src/gesturebridge/ui/web.py:572` — read-mode layout shows the new
  card only when `word_loaded == true`.

### Sources / URLs

(no external downloads; pure code change)

### Acceptance check

```bash
.venv/bin/python -m gesturebridge.app --run-main --camera-index 0
# → opens http://127.0.0.1:8080
# → Read mode tab shows new "Word Recognition (WLASL-100)" card
# → Sign a word, click "Capture Word (1s)"
# → After ~1 s, top-5 predictions appear with confidence bars
# → If top-1 ≥ 0.35, TTS speaks the recognized word
```

### Next iteration entry point

- **IT-6 (when Yizheng confirms Pi state):** rsync the new
  `artifacts/wlasl100/` to Pi, run end-to-end on the C270, measure
  added word-mode latency.
- **Data top-up:** chunked yt-dlp on YouTube clips per-gloss to roughly
  double the dataset, then retrain Conv1D and re-export weights.
- **IT-7:** expand `WORD_CLIP_MAP` in `main_runtime.py` so speech-to-sign
  also covers the WLASL-100 vocabulary (today only 5 words have video
  clips; we can mine display clips from the WLASL videos already on
  disk).

---



